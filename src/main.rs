mod analysis;
mod input;
mod parquet_output;
mod perfetto;

use analysis::*;
use input::*;
use parquet_output::*;
use perfetto::*;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use datafusion::arrow::array::{
    Array, Float64Array, Int64Array, StringArray, StringViewArray, UInt32Array, UInt64Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::Serialize;
use serde_json::{Value, json};

const GLOBAL_ID_RADIX: i64 = 0x1000000;
const GPU_PROCESS_ID_BASE: i64 = 2_000_000_000;
const LAUNCH_FLOW_ID_BASE: u64 = 1_u64 << 51;
const MEMCPY_FLOW_ID_BASE: u64 = 2_u64 << 51;
const CORE_LAUNCH_FLOW_ID_BASE: u64 = 3_u64 << 51;
const PCIE_USAGE_FLOW_ID_BASE: u64 = 4_u64 << 51;

#[derive(Parser, Debug)]
#[command(about = "Convert Nsight Parquet tables to Perfetto JSON with DataFusion")]
struct Args {
    #[arg(long)]
    parquet_dir: PathBuf,
    #[arg(long)]
    report: String,
    #[arg(long)]
    output_json: PathBuf,
    #[arg(long)]
    output_parquet: PathBuf,
    #[arg(long)]
    output_dependencies: PathBuf,
}

struct TableAvailability {
    kernels: bool,
    runtime: bool,
    memcpy: bool,
    nvtx: bool,
    strings: bool,
}

#[derive(Debug)]
struct Kernel {
    start: i64,
    end: i64,
    device: i64,
    context: i64,
    stream: i64,
    correlation: i64,
    global_pid: i64,
    name: String,
    grid: [i64; 3],
    block: [i64; 3],
    sequence: u64,
    event_id: String,
    launch_call: Option<usize>,
    nvtx_regions: Vec<String>,
}

#[derive(Debug)]
struct Memcpy {
    start: i64,
    end: i64,
    device: i64,
    context: i64,
    stream: i64,
    correlation: i64,
    global_pid: i64,
    bytes: u64,
    copy_kind: i64,
    src_kind: i64,
    dst_kind: i64,
    src_device: i64,
    src_context: i64,
    dst_device: i64,
    dst_context: i64,
    graph_node: i64,
    virtual_address: String,
    copy_count: u64,
    event_id: String,
    launch_call: Option<usize>,
}

#[derive(Debug)]
struct RuntimeCall {
    start: i64,
    end: i64,
    global_tid: i64,
    correlation: i64,
    name: String,
    event_id: Option<String>,
}

#[derive(Debug)]
struct NvtxRange {
    start: i64,
    end: i64,
    name: String,
    pid: i64,
    tid: i64,
    kernel_bounds: BTreeMap<i64, (i64, i64)>,
}

struct TraceRow {
    report: String,
    event_type: String,
    cat: String,
    name: String,
    ph: String,
    ts_us: f64,
    dur_us: Option<f64>,
    aligned_ts_us: f64,
    pid: String,
    tid: String,
    args_json: String,
    event_id: Option<String>,
    launch_event_id: Option<String>,
    stream_id: Option<u64>,
    correlation_id: Option<u32>,
    stream_sequence: Option<u64>,
    depends_on_event_id: Option<String>,
    dependency_type: Option<String>,
}

struct DependencyRow {
    report: String,
    stream_id: u64,
    stream_sequence: u64,
    predecessor_event_id: String,
    predecessor_kernel: String,
    predecessor_ts_us: f64,
    predecessor_dur_us: f64,
    successor_event_id: String,
    successor_kernel: String,
    successor_ts_us: f64,
    successor_dur_us: f64,
    gap_us: f64,
    dependency_type: String,
}

fn prepare_output(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if path.is_dir() {
        fs::remove_dir_all(path)?;
    } else if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> Result<()> {
    let args = Args::parse();
    let ctx = SessionContext::new();
    let tables = register_tables(&ctx, &args.parquet_dir).await?;
    if !tables.strings {
        eprintln!("warning: StringIds.parquet is absent; using numeric fallback event names");
    }

    let mut kernels = if tables.kernels {
        load_kernels(&ctx).await?
    } else {
        eprintln!("warning: CUPTI_ACTIVITY_KIND_KERNEL.parquet is absent");
        Vec::new()
    };
    let mut nvtx = if tables.nvtx {
        load_nvtx(&ctx).await?
    } else {
        eprintln!("warning: NVTX_EVENTS.parquet is absent; exporting CUDA timelines without NVTX");
        Vec::new()
    };
    let mut runtime = if tables.runtime {
        load_runtime(&ctx).await?
    } else {
        eprintln!(
            "warning: CUPTI_ACTIVITY_KIND_RUNTIME.parquet is absent; API flows are unavailable"
        );
        Vec::new()
    };
    let mut memcpys = if tables.memcpy {
        load_memcpys(&ctx).await?
    } else {
        Vec::new()
    };
    let (launch_dependencies, memcpy_launch_dependencies) = link_runtime_calls_to_gpu_activities(
        &args.report,
        &mut kernels,
        &mut memcpys,
        &mut runtime,
    );
    let cuda_sync_api_count = mark_cuda_sync_calls(&args.report, &mut runtime);
    if kernels.is_empty() && memcpys.is_empty() {
        for (call_idx, call) in runtime.iter_mut().enumerate() {
            let pid = source_pid(call.global_tid);
            let tid = call.global_tid % GLOBAL_ID_RADIX;
            call.event_id = Some(format!(
                "{}:cuda_api_unlinked:{pid}:{tid}:{}:{call_idx}",
                args.report, call.correlation
            ));
        }
    }
    project_nvtx_to_kernels(&mut kernels, &mut nvtx, &runtime);
    assign_kernel_ids(&args.report, &mut kernels);

    let origin_ns = kernels
        .iter()
        .map(|k| k.start)
        .chain(memcpys.iter().map(|copy| copy.start))
        .chain(nvtx.iter().map(|n| n.start))
        .chain(
            runtime
                .iter()
                .filter(|call| call.event_id.is_some())
                .map(|call| call.start),
        )
        .min()
        .context("trace contains no CUDA kernel, memcpy, linked API, or NVTX ranges")?;
    let measured_batch_anchor = nvtx
        .iter()
        .filter(|n| {
            n.name.starts_with("CriticalPath/MeasuredBatch/") && n.name.ends_with("/batch_0")
        })
        .map(|n| n.start)
        .min();
    let anchor_ns = measured_batch_anchor.unwrap_or(origin_ns);
    let alignment_anchor = if measured_batch_anchor.is_some() {
        "critical_path_batch_0"
    } else {
        eprintln!(
            "warning: no CriticalPath/MeasuredBatch/.../batch_0 NVTX anchor; using first trace event"
        );
        "first_trace_event"
    };

    let projected_nvtx: usize = nvtx.iter().map(|n| n.kernel_bounds.len()).sum();
    let (trace_rows, dependencies, json_events, pcie_usage_launch_dependencies) =
        emit_outputs(EmitInput {
            report: &args.report,
            output_json: &args.output_json,
            kernels: &kernels,
            memcpys: &memcpys,
            runtime: &runtime,
            nvtx: &nvtx,
            origin_ns,
            anchor_ns,
        })?;
    let trace_row_count = trace_rows.len();
    let cuda_api_count = runtime
        .iter()
        .filter(|call| call.event_id.is_some())
        .count();
    let dependency_count = dependencies.len();
    let h2d_count = memcpys
        .iter()
        .filter(|copy| matches!(copy.copy_kind, 1 | 11))
        .count();
    let d2h_count = memcpys
        .iter()
        .filter(|copy| matches!(copy.copy_kind, 2 | 12))
        .count();
    let d2d_count = memcpys
        .iter()
        .filter(|copy| matches!(copy.copy_kind, 3 | 8 | 13))
        .count();
    write_parquet(&ctx, trace_rows_batch(trace_rows)?, &args.output_parquet).await?;
    write_parquet(
        &ctx,
        dependency_batch(dependencies)?,
        &args.output_dependencies,
    )
    .await?;

    println!(
        "report={} kernels={} cuda_api={} cuda_sync_api={} launch_dependencies={} core_launch_dependencies={} memcpy={} h2d={} d2h={} d2d={} memcpy_launch_dependencies={} pcie_usage_launch_dependencies={} nvtx={} nvtx_kernel={} stream_dependencies={} json_events={} parquet_rows={} anchor_ns={} alignment_anchor={}",
        args.report,
        kernels.len(),
        cuda_api_count,
        cuda_sync_api_count,
        launch_dependencies,
        launch_dependencies,
        memcpys.len(),
        h2d_count,
        d2h_count,
        d2d_count,
        memcpy_launch_dependencies,
        pcie_usage_launch_dependencies,
        nvtx.len(),
        projected_nvtx,
        dependency_count,
        json_events,
        trace_row_count,
        anchor_ns,
        alignment_anchor,
    );
    Ok(())
}
