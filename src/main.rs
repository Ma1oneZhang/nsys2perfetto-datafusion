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
use serde::Serialize;
use serde_json::{Value, json};

const GLOBAL_ID_RADIX: i64 = 0x1000000;
const GPU_PROCESS_ID_BASE: i64 = 2_000_000_000;
const LAUNCH_FLOW_ID_BASE: u64 = 1_u64 << 51;
const MEMCPY_FLOW_ID_BASE: u64 = 2_u64 << 51;

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

#[derive(Serialize)]
struct TraceEvent {
    name: String,
    ph: String,
    cat: String,
    ts: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    dur: Option<f64>,
    tid: i64,
    pid: i64,
    args: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bp: Option<String>,
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

struct JsonArrayWriter {
    writer: BufWriter<File>,
    first: bool,
}

impl JsonArrayWriter {
    fn create(path: &Path) -> Result<Self> {
        prepare_output(path)?;
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(b"[")?;
        Ok(Self {
            writer,
            first: true,
        })
    }

    fn event(&mut self, event: &TraceEvent) -> Result<()> {
        if !self.first {
            self.writer.write_all(b",")?;
        }
        self.first = false;
        serde_json::to_writer(&mut self.writer, event)?;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.writer.write_all(b"]\n")?;
        self.writer.flush()?;
        Ok(())
    }
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

fn i64_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column {name}"))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| anyhow!("column {name} is not Int64"))
}

fn string_at(batch: &RecordBatch, name: &str, row: usize) -> Result<String> {
    let array = batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column {name}"))?;
    if let Some(strings) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(strings.value(row).to_owned());
    }
    if let Some(strings) = array.as_any().downcast_ref::<StringViewArray>() {
        return Ok(strings.value(row).to_owned());
    }
    bail!("column {name} is neither Utf8 nor Utf8View")
}

async fn register_tables(ctx: &SessionContext, dir: &Path) -> Result<bool> {
    let tables = [
        ("kernels", "CUPTI_ACTIVITY_KIND_KERNEL.parquet"),
        ("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME.parquet"),
        ("nvtx", "NVTX_EVENTS.parquet"),
        ("strings", "StringIds.parquet"),
    ];
    for (name, file) in tables {
        let path = dir.join(file);
        if !path.is_file() {
            bail!(
                "required Nsight Parquet table is missing: {}",
                path.display()
            );
        }
        ctx.register_parquet(
            name,
            path.to_str().context("non-UTF8 Parquet path")?,
            ParquetReadOptions::default(),
        )
        .await?;
    }
    let memcpy_path = dir.join("CUPTI_ACTIVITY_KIND_MEMCPY.parquet");
    if memcpy_path.is_file() {
        ctx.register_parquet(
            "memcpy",
            memcpy_path.to_str().context("non-UTF8 Parquet path")?,
            ParquetReadOptions::default(),
        )
        .await?;
        Ok(true)
    } else {
        Ok(false)
    }
}

async fn load_kernels(ctx: &SessionContext) -> Result<Vec<Kernel>> {
    let sql = r#"
        SELECT
            CAST(k.start AS BIGINT) AS start_ns,
            CAST(k."end" AS BIGINT) AS end_ns,
            CAST(k."deviceId" AS BIGINT) AS device_id,
            CAST(k."contextId" AS BIGINT) AS context_id,
            CAST(k."streamId" AS BIGINT) AS stream_id,
            CAST(k."correlationId" AS BIGINT) AS correlation_id,
            CAST(k."globalPid" AS BIGINT) AS global_pid,
            COALESCE(s.value, CONCAT('kernel_', CAST(k."shortName" AS VARCHAR))) AS kernel_name,
            CAST(k."gridX" AS BIGINT) AS grid_x,
            CAST(k."gridY" AS BIGINT) AS grid_y,
            CAST(k."gridZ" AS BIGINT) AS grid_z,
            CAST(k."blockX" AS BIGINT) AS block_x,
            CAST(k."blockY" AS BIGINT) AS block_y,
            CAST(k."blockZ" AS BIGINT) AS block_z
        FROM kernels k
        LEFT JOIN strings s ON CAST(k."shortName" AS BIGINT) = CAST(s.id AS BIGINT)
        ORDER BY start_ns, end_ns, correlation_id
    "#;
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut kernels = Vec::new();
    for batch in batches {
        let start = i64_col(&batch, "start_ns")?;
        let end = i64_col(&batch, "end_ns")?;
        let device = i64_col(&batch, "device_id")?;
        let context = i64_col(&batch, "context_id")?;
        let stream = i64_col(&batch, "stream_id")?;
        let correlation = i64_col(&batch, "correlation_id")?;
        let global_pid = i64_col(&batch, "global_pid")?;
        let gx = i64_col(&batch, "grid_x")?;
        let gy = i64_col(&batch, "grid_y")?;
        let gz = i64_col(&batch, "grid_z")?;
        let bx = i64_col(&batch, "block_x")?;
        let by = i64_col(&batch, "block_y")?;
        let bz = i64_col(&batch, "block_z")?;
        for row in 0..batch.num_rows() {
            kernels.push(Kernel {
                start: start.value(row),
                end: end.value(row),
                device: device.value(row),
                context: context.value(row),
                stream: stream.value(row),
                correlation: correlation.value(row),
                global_pid: global_pid.value(row),
                name: string_at(&batch, "kernel_name", row)?,
                grid: [gx.value(row), gy.value(row), gz.value(row)],
                block: [bx.value(row), by.value(row), bz.value(row)],
                sequence: 0,
                event_id: String::new(),
                launch_call: None,
                nvtx_regions: Vec::new(),
            });
        }
    }
    Ok(kernels)
}

async fn load_nvtx(ctx: &SessionContext) -> Result<Vec<NvtxRange>> {
    // Matches nsys2json: only NvtxPushPopRange (eventType 59), with StringIds fallback.
    let sql = r#"
        SELECT
            CAST(n.start AS BIGINT) AS start_ns,
            CAST(n."end" AS BIGINT) AS end_ns,
            COALESCE(n.text, s.value, 'NVTX') AS nvtx_name,
            CAST(n."globalTid" AS BIGINT) AS global_tid
        FROM nvtx n
        LEFT JOIN strings s ON CAST(n."textId" AS BIGINT) = CAST(s.id AS BIGINT)
        WHERE CAST(n."eventType" AS BIGINT) = 59 AND n."end" IS NOT NULL
        ORDER BY start_ns, end_ns
    "#;
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut ranges = Vec::new();
    for batch in batches {
        let start = i64_col(&batch, "start_ns")?;
        let end = i64_col(&batch, "end_ns")?;
        let global_tid = i64_col(&batch, "global_tid")?;
        for row in 0..batch.num_rows() {
            let gid = global_tid.value(row);
            ranges.push(NvtxRange {
                start: start.value(row),
                end: end.value(row),
                name: string_at(&batch, "nvtx_name", row)?,
                pid: (gid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX,
                tid: gid % GLOBAL_ID_RADIX,
                kernel_bounds: BTreeMap::new(),
            });
        }
    }
    Ok(ranges)
}

async fn load_runtime(ctx: &SessionContext) -> Result<Vec<RuntimeCall>> {
    let sql = r#"
        SELECT
            CAST(r.start AS BIGINT) AS start_ns,
            CAST(r."end" AS BIGINT) AS end_ns,
            CAST(r."globalTid" AS BIGINT) AS global_tid,
            CAST(r."correlationId" AS BIGINT) AS correlation_id,
            COALESCE(s.value, CONCAT('cuda_api_', CAST(r."nameId" AS VARCHAR))) AS runtime_name
        FROM runtime r
        LEFT JOIN strings s ON CAST(r."nameId" AS BIGINT) = CAST(s.id AS BIGINT)
        ORDER BY start_ns, end_ns, correlation_id
    "#;
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut calls = Vec::new();
    for batch in batches {
        let start = i64_col(&batch, "start_ns")?;
        let end = i64_col(&batch, "end_ns")?;
        let global_tid = i64_col(&batch, "global_tid")?;
        let correlation = i64_col(&batch, "correlation_id")?;
        for row in 0..batch.num_rows() {
            calls.push(RuntimeCall {
                start: start.value(row),
                end: end.value(row),
                global_tid: global_tid.value(row),
                correlation: correlation.value(row),
                name: string_at(&batch, "runtime_name", row)?,
                event_id: None,
            });
        }
    }
    Ok(calls)
}

async fn load_memcpys(ctx: &SessionContext) -> Result<Vec<Memcpy>> {
    let sql = r#"
        SELECT
            CAST(start AS BIGINT) AS start_ns,
            CAST("end" AS BIGINT) AS end_ns,
            CAST("deviceId" AS BIGINT) AS device_id,
            CAST("contextId" AS BIGINT) AS context_id,
            CAST("streamId" AS BIGINT) AS stream_id,
            CAST("correlationId" AS BIGINT) AS correlation_id,
            CAST("globalPid" AS BIGINT) AS global_pid,
            CAST(bytes AS BIGINT) AS bytes,
            CAST("copyKind" AS BIGINT) AS copy_kind,
            CAST("srcKind" AS BIGINT) AS src_kind,
            CAST("dstKind" AS BIGINT) AS dst_kind,
            COALESCE(CAST("srcDeviceId" AS BIGINT), -1) AS src_device_id,
            COALESCE(CAST("srcContextId" AS BIGINT), -1) AS src_context_id,
            COALESCE(CAST("dstDeviceId" AS BIGINT), -1) AS dst_device_id,
            COALESCE(CAST("dstContextId" AS BIGINT), -1) AS dst_context_id,
            COALESCE(CAST("graphNodeId" AS BIGINT), -1) AS graph_node_id,
            COALESCE(CAST("virtualAddress" AS VARCHAR), '0') AS virtual_address,
            COALESCE(CAST("copyCount" AS BIGINT), 1) AS copy_count
        FROM memcpy
        WHERE CAST("copyKind" AS BIGINT) IN (1, 2, 3, 8, 11, 12, 13)
        ORDER BY start_ns, end_ns, correlation_id
    "#;
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut copies = Vec::new();
    for batch in batches {
        let start = i64_col(&batch, "start_ns")?;
        let end = i64_col(&batch, "end_ns")?;
        let device = i64_col(&batch, "device_id")?;
        let context = i64_col(&batch, "context_id")?;
        let stream = i64_col(&batch, "stream_id")?;
        let correlation = i64_col(&batch, "correlation_id")?;
        let global_pid = i64_col(&batch, "global_pid")?;
        let bytes = i64_col(&batch, "bytes")?;
        let copy_kind = i64_col(&batch, "copy_kind")?;
        let src_kind = i64_col(&batch, "src_kind")?;
        let dst_kind = i64_col(&batch, "dst_kind")?;
        let src_device = i64_col(&batch, "src_device_id")?;
        let src_context = i64_col(&batch, "src_context_id")?;
        let dst_device = i64_col(&batch, "dst_device_id")?;
        let dst_context = i64_col(&batch, "dst_context_id")?;
        let graph_node = i64_col(&batch, "graph_node_id")?;
        let copy_count = i64_col(&batch, "copy_count")?;
        for row in 0..batch.num_rows() {
            copies.push(Memcpy {
                start: start.value(row),
                end: end.value(row),
                device: device.value(row),
                context: context.value(row),
                stream: stream.value(row),
                correlation: correlation.value(row),
                global_pid: global_pid.value(row),
                bytes: bytes.value(row) as u64,
                copy_kind: copy_kind.value(row),
                src_kind: src_kind.value(row),
                dst_kind: dst_kind.value(row),
                src_device: src_device.value(row),
                src_context: src_context.value(row),
                dst_device: dst_device.value(row),
                dst_context: dst_context.value(row),
                graph_node: graph_node.value(row),
                virtual_address: string_at(&batch, "virtual_address", row)?,
                copy_count: copy_count.value(row) as u64,
                event_id: String::new(),
                launch_call: None,
            });
        }
    }
    Ok(copies)
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum BoundaryKind {
    ApiEnd,
    NvtxEnd,
    ApiStart,
    NvtxStart,
}

fn link_runtime_calls_to_gpu_activities(
    report: &str,
    kernels: &mut [Kernel],
    memcpys: &mut [Memcpy],
    runtime: &mut [RuntimeCall],
) -> (usize, usize) {
    let mut calls_by_process_correlation: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (idx, call) in runtime.iter().enumerate() {
        let pid = (call.global_tid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX;
        calls_by_process_correlation
            .entry((pid, call.correlation))
            .or_default()
            .push(idx);
    }

    let mut kernel_links = 0;
    for kernel in kernels {
        let pid = (kernel.global_pid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX;
        let Some(candidates) = calls_by_process_correlation.get(&(pid, kernel.correlation)) else {
            continue;
        };
        let Some(&call_idx) = candidates.iter().min_by_key(|&&idx| {
            let call = &runtime[idx];
            (call.start > kernel.start, call.start.abs_diff(kernel.start))
        }) else {
            continue;
        };

        if runtime[call_idx].event_id.is_none() {
            let call = &runtime[call_idx];
            let tid = call.global_tid % GLOBAL_ID_RADIX;
            runtime[call_idx].event_id = Some(format!(
                "{report}:cuda_api:{pid}:{tid}:{}:{call_idx}",
                call.correlation
            ));
        }
        kernel.launch_call = Some(call_idx);
        kernel_links += 1;
    }

    let mut memcpy_links = 0;
    for (copy_idx, copy) in memcpys.iter_mut().enumerate() {
        copy.event_id = format!(
            "{report}:cuda_memcpy:{}:{}:{}:{copy_idx}",
            copy.device, copy.context, copy.stream
        );
        let pid = (copy.global_pid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX;
        let Some(candidates) = calls_by_process_correlation.get(&(pid, copy.correlation)) else {
            continue;
        };
        let Some(&call_idx) = candidates.iter().min_by_key(|&&idx| {
            let call = &runtime[idx];
            (call.start > copy.start, call.start.abs_diff(copy.start))
        }) else {
            continue;
        };
        if runtime[call_idx].event_id.is_none() {
            let call = &runtime[call_idx];
            let tid = call.global_tid % GLOBAL_ID_RADIX;
            runtime[call_idx].event_id = Some(format!(
                "{report}:cuda_api:{pid}:{tid}:{}:{call_idx}",
                call.correlation
            ));
        }
        copy.launch_call = Some(call_idx);
        memcpy_links += 1;
    }
    (kernel_links, memcpy_links)
}

fn project_nvtx_to_kernels(
    kernels: &mut [Kernel],
    nvtx: &mut [NvtxRange],
    runtime: &[RuntimeCall],
) {
    let mut kernels_by_process_correlation: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (idx, kernel) in kernels.iter().enumerate() {
        let pid = (kernel.global_pid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX;
        kernels_by_process_correlation
            .entry((pid, kernel.correlation))
            .or_default()
            .push(idx);
    }

    let process_devices: BTreeSet<(i64, i64)> = kernels
        .iter()
        .map(|kernel| {
            (
                (kernel.global_pid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX,
                kernel.device,
            )
        })
        .collect();

    // nsys2json overlaps all NVTX and CUDA API intervals on a device, then
    // follows correlationId to a kernel. Grouping by process as well prevents
    // unrelated processes on the same GPU from contaminating one another,
    // while deliberately preserving its cross-thread overlap semantics.
    for (pid, device) in process_devices {
        let mut boundaries: Vec<(i64, BoundaryKind, usize)> = Vec::new();
        for (idx, range) in nvtx
            .iter()
            .enumerate()
            .filter(|(_, range)| range.pid == pid)
        {
            boundaries.push((range.start, BoundaryKind::NvtxStart, idx));
            boundaries.push((range.end, BoundaryKind::NvtxEnd, idx));
        }
        for (idx, call) in runtime.iter().enumerate().filter(|(_, call)| {
            let call_pid = (call.global_tid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX;
            call_pid == pid
                && kernels_by_process_correlation
                    .get(&(pid, call.correlation))
                    .is_some_and(|indices| {
                        indices
                            .iter()
                            .any(|&kernel_idx| kernels[kernel_idx].device == device)
                    })
        }) {
            boundaries.push((call.start, BoundaryKind::ApiStart, idx));
            boundaries.push((call.end, BoundaryKind::ApiEnd, idx));
        }
        boundaries.sort_unstable_by_key(|&(time, kind, idx)| (time, kind, idx));

        let mut active_nvtx = BTreeSet::new();
        for (_, kind, idx) in boundaries {
            match kind {
                BoundaryKind::NvtxStart => {
                    active_nvtx.insert(idx);
                }
                BoundaryKind::NvtxEnd => {
                    active_nvtx.remove(&idx);
                }
                BoundaryKind::ApiEnd => {}
                BoundaryKind::ApiStart => {
                    let call = &runtime[idx];
                    let Some(kernel_indices) =
                        kernels_by_process_correlation.get(&(pid, call.correlation))
                    else {
                        continue;
                    };

                    for &kernel_idx in kernel_indices {
                        if kernels[kernel_idx].device != device {
                            continue;
                        }
                        let kernel_start = kernels[kernel_idx].start;
                        let kernel_end = kernels[kernel_idx].end;
                        for &nvtx_idx in &active_nvtx {
                            let range = &mut nvtx[nvtx_idx];
                            range
                                .kernel_bounds
                                .entry(device)
                                .and_modify(|bounds| {
                                    bounds.0 = bounds.0.min(kernel_start);
                                    bounds.1 = bounds.1.max(kernel_end);
                                })
                                .or_insert((kernel_start, kernel_end));
                            let region_name = range.name.clone();
                            if !kernels[kernel_idx].nvtx_regions.contains(&region_name) {
                                kernels[kernel_idx].nvtx_regions.push(region_name);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_kernel(pid: i64, device: i64, correlation: i64, start: i64) -> Kernel {
        Kernel {
            start,
            end: start + 5,
            device,
            context: device,
            stream: device,
            correlation,
            global_pid: pid * GLOBAL_ID_RADIX,
            name: format!("kernel_{device}"),
            grid: [1, 1, 1],
            block: [1, 1, 1],
            sequence: 0,
            event_id: String::new(),
            launch_call: None,
            nvtx_regions: Vec::new(),
        }
    }

    #[test]
    fn projects_one_process_to_multiple_devices() {
        let pid = 42;
        let tid = 7;
        let mut kernels = vec![test_kernel(pid, 0, 10, 15), test_kernel(pid, 1, 11, 25)];
        let mut nvtx = vec![NvtxRange {
            start: 0,
            end: 100,
            name: "multi_gpu_range".into(),
            pid,
            tid,
            kernel_bounds: BTreeMap::new(),
        }];
        let mut runtime = vec![
            RuntimeCall {
                start: 10,
                end: 12,
                global_tid: pid * GLOBAL_ID_RADIX + tid,
                correlation: 10,
                name: "cudaLaunchKernel".into(),
                event_id: None,
            },
            RuntimeCall {
                start: 20,
                end: 22,
                // Match nsys2json: NVTX/API interval overlap is per process and
                // device, so an API call on another thread is still projected.
                global_tid: pid * GLOBAL_ID_RADIX + tid + 1,
                correlation: 11,
                name: "cudaLaunchKernel".into(),
                event_id: None,
            },
        ];

        let mut memcpys = Vec::new();
        assert_eq!(
            link_runtime_calls_to_gpu_activities(
                "report",
                &mut kernels,
                &mut memcpys,
                &mut runtime
            ),
            (2, 0)
        );
        project_nvtx_to_kernels(&mut kernels, &mut nvtx, &runtime);

        assert_eq!(kernels[0].launch_call, Some(0));
        assert_eq!(kernels[1].launch_call, Some(1));
        assert!(runtime.iter().all(|call| call.event_id.is_some()));
        assert_eq!(nvtx[0].kernel_bounds.get(&0), Some(&(15, 20)));
        assert_eq!(nvtx[0].kernel_bounds.get(&1), Some(&(25, 30)));
        assert_eq!(kernels[0].nvtx_regions, ["multi_gpu_range"]);
        assert_eq!(kernels[1].nvtx_regions, ["multi_gpu_range"]);

        runtime[1].end = kernels[1].start + 10;
        let flow_start = flow_start_ns(&runtime[1], kernels[1].start);
        assert!(flow_start >= runtime[1].start);
        assert!(flow_start < kernels[1].start);
    }
}

fn assign_kernel_ids(report: &str, kernels: &mut [Kernel]) {
    let mut sequence: HashMap<(i64, i64, i64, i64), u64> = HashMap::new();
    for idx in 0..kernels.len() {
        let source_pid = (kernels[idx].global_pid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX;
        let key = (
            source_pid,
            kernels[idx].device,
            kernels[idx].context,
            kernels[idx].stream,
        );
        let seq = sequence.entry(key).or_default();
        *seq += 1;
        kernels[idx].sequence = *seq;
        kernels[idx].event_id = format!(
            "{report}:cuda:{source_pid}:{}:{}:{}:{}",
            kernels[idx].device, kernels[idx].context, kernels[idx].stream, seq
        );
    }
}

fn ns_to_us(ns: i64) -> f64 {
    ns as f64 / 1000.0
}

fn flow_start_ns(call: &RuntimeCall, activity_start: i64) -> i64 {
    (call.end - 1).min(activity_start - 1).max(call.start)
}

fn source_pid(global_pid: i64) -> i64 {
    (global_pid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX
}

fn copy_direction(copy_kind: i64) -> &'static str {
    match copy_kind {
        1 => "H2D",
        2 => "D2H",
        3 | 8 => "D2D",
        11 => "Unified H2D",
        12 => "Unified D2H",
        13 => "Unified D2D",
        _ => "Unknown",
    }
}

fn memory_kind(kind: i64) -> &'static str {
    match kind {
        0 => "Pageable",
        1 => "Pinned",
        2 => "Device",
        3 => "Array",
        4 => "Managed",
        5 => "Device Static",
        6 => "Managed Static",
        _ => "Unknown",
    }
}

fn emit_outputs(
    report: &str,
    output_json: &Path,
    kernels: &[Kernel],
    memcpys: &[Memcpy],
    runtime: &[RuntimeCall],
    nvtx: &[NvtxRange],
    origin_ns: i64,
    anchor_ns: i64,
) -> Result<(Vec<TraceRow>, Vec<DependencyRow>, usize)> {
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum DeviceSliceKey {
        Runtime(usize, i64),
        Nvtx(usize, i64),
        ProjectedNvtx(usize, i64),
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum DeviceTrackKind {
        ProjectedNvtx,
        Runtime,
        Nvtx,
    }

    let mut writer = JsonArrayWriter::create(output_json)?;
    let mut rows =
        Vec::with_capacity(kernels.len() + memcpys.len() + runtime.len() + nvtx.len() * 2);
    let dependencies = Vec::new();
    let mut json_event_count = 0usize;

    // A CUDA device ID is process-local. Keep the source process in every GPU
    // track key so two processes' device 0 never collapse onto one device.
    let gpu_keys: BTreeSet<(i64, i64)> = kernels
        .iter()
        .map(|kernel| (source_pid(kernel.global_pid), kernel.device))
        .chain(
            memcpys
                .iter()
                .map(|copy| (source_pid(copy.global_pid), copy.device)),
        )
        .collect();
    let gpu_processes: BTreeMap<(i64, i64), i64> = gpu_keys
        .iter()
        .enumerate()
        .map(|(idx, &key)| (key, GPU_PROCESS_ID_BASE + idx as i64))
        .collect();
    let stream_keys: BTreeSet<(i64, i64, i64, i64)> = kernels
        .iter()
        .map(|kernel| {
            (
                source_pid(kernel.global_pid),
                kernel.device,
                kernel.context,
                kernel.stream,
            )
        })
        .chain(memcpys.iter().map(|copy| {
            (
                source_pid(copy.global_pid),
                copy.device,
                copy.context,
                copy.stream,
            )
        }))
        .collect();
    let stream_tracks: BTreeMap<(i64, i64, i64, i64), i64> = stream_keys
        .iter()
        .enumerate()
        .map(|(idx, &key)| (key, idx as i64 + 1))
        .collect();
    let mut call_devices: BTreeMap<usize, BTreeSet<i64>> = BTreeMap::new();
    for kernel in kernels {
        if let Some(call_idx) = kernel.launch_call {
            call_devices
                .entry(call_idx)
                .or_default()
                .insert(kernel.device);
        }
    }
    for copy in memcpys {
        if let Some(call_idx) = copy.launch_call {
            call_devices
                .entry(call_idx)
                .or_default()
                .insert(copy.device);
        }
    }
    let process_devices: BTreeMap<i64, BTreeSet<i64>> =
        gpu_keys
            .iter()
            .fold(BTreeMap::new(), |mut devices, &(pid, device)| {
                devices.entry(pid).or_default().insert(device);
                devices
            });

    // Place CUDA API tracks under each CUDA device process. Complete CUDA API
    // events can overlap, so allocate the minimum number of adjacent lanes.
    // NVTX tracks are allocated separately below and use B/E stack events to
    // preserve push/pop hierarchy on one track per source thread.
    let mut device_intervals: BTreeMap<
        (i64, i64, DeviceTrackKind, i64),
        Vec<(i64, i64, DeviceSliceKey)>,
    > = BTreeMap::new();
    for (&call_idx, devices) in &call_devices {
        let call = &runtime[call_idx];
        let pid = source_pid(call.global_tid);
        let tid = call.global_tid % GLOBAL_ID_RADIX;
        for &device in devices {
            device_intervals
                .entry((pid, device, DeviceTrackKind::Runtime, tid))
                .or_default()
                .push((
                    call.start,
                    call.end,
                    DeviceSliceKey::Runtime(call_idx, device),
                ));
        }
    }
    let mut device_slice_tracks = BTreeMap::new();
    let mut device_track_metadata = Vec::new();
    let projected_track_keys: BTreeSet<(i64, i64, i64)> = nvtx
        .iter()
        .flat_map(|range| {
            range
                .kernel_bounds
                .keys()
                .map(move |&device| (range.pid, device, range.tid))
        })
        .collect();
    let projected_tracks: BTreeMap<(i64, i64, i64), i64> = projected_track_keys
        .iter()
        .enumerate()
        .map(|(idx, &key)| (key, 1_000_000_000_i64 + idx as i64))
        .collect();
    for (&(pid, device, source_tid), &tid) in &projected_tracks {
        device_track_metadata.push((
            pid,
            device,
            DeviceTrackKind::ProjectedNvtx,
            tid,
            source_tid,
            None,
        ));
    }
    for (range_idx, range) in nvtx.iter().enumerate() {
        for &device in range.kernel_bounds.keys() {
            device_slice_tracks.insert(
                DeviceSliceKey::ProjectedNvtx(range_idx, device),
                projected_tracks[&(range.pid, device, range.tid)],
            );
        }
    }

    let mut next_device_track = 1_100_000_000_i64;
    for (&(pid, device, kind, source_tid), intervals) in &mut device_intervals {
        intervals.sort_unstable_by_key(|&(start, end, key)| (start, end, key));
        let mut lane_ends: Vec<i64> = Vec::new();
        let mut lane_tids: Vec<i64> = Vec::new();
        for &(start, end, key) in intervals.iter() {
            let lane = lane_ends
                .iter()
                .position(|&lane_end| lane_end <= start)
                .unwrap_or_else(|| {
                    let lane = lane_ends.len();
                    lane_ends.push(i64::MIN);
                    lane_tids.push(next_device_track);
                    device_track_metadata.push((
                        pid,
                        device,
                        kind,
                        next_device_track,
                        source_tid,
                        Some(lane),
                    ));
                    next_device_track += 1;
                    lane
                });
            lane_ends[lane] = end;
            device_slice_tracks.insert(key, lane_tids[lane]);
        }
    }

    let mut nvtx_track_keys = BTreeSet::new();
    for range in nvtx {
        let range_devices: Vec<i64> = if range.kernel_bounds.is_empty() {
            process_devices
                .get(&range.pid)
                .into_iter()
                .flatten()
                .copied()
                .collect()
        } else {
            range.kernel_bounds.keys().copied().collect()
        };
        nvtx_track_keys.extend(
            range_devices
                .into_iter()
                .map(|device| (range.pid, device, range.tid)),
        );
    }
    let nvtx_tracks: BTreeMap<(i64, i64, i64), i64> = nvtx_track_keys
        .iter()
        .enumerate()
        .map(|(idx, &key)| (key, 1_200_000_000_i64 + idx as i64))
        .collect();
    for (&(pid, device, source_tid), &tid) in &nvtx_tracks {
        device_track_metadata.push((pid, device, DeviceTrackKind::Nvtx, tid, source_tid, None));
    }
    for (range_idx, range) in nvtx.iter().enumerate() {
        let range_devices: Vec<i64> = if range.kernel_bounds.is_empty() {
            process_devices
                .get(&range.pid)
                .into_iter()
                .flatten()
                .copied()
                .collect()
        } else {
            range.kernel_bounds.keys().copied().collect()
        };
        for device in range_devices {
            device_slice_tracks.insert(
                DeviceSliceKey::Nvtx(range_idx, device),
                nvtx_tracks[&(range.pid, device, range.tid)],
            );
        }
    }

    for (&(source_pid, device), &pid) in &gpu_processes {
        for (name, args) in [
            (
                "process_name",
                json!({"name": format!("CUDA Device {device} / Source PID {source_pid}")}),
            ),
            (
                "process_sort_index",
                json!({"sort_index": -10_000 + pid - GPU_PROCESS_ID_BASE}),
            ),
        ] {
            writer.event(&TraceEvent {
                name: name.into(),
                ph: "M".into(),
                cat: "__metadata".into(),
                ts: 0.0,
                dur: None,
                tid: 0,
                pid,
                args,
                id: None,
                bp: None,
            })?;
            json_event_count += 1;
        }
    }
    for (&(source_pid, device, context, stream), &tid) in &stream_tracks {
        for (name, args) in [
            (
                "thread_name",
                json!({"name": format!("CUDA HW Context {context} / Stream {stream}")}),
            ),
            ("thread_sort_index", json!({"sort_index": tid})),
        ] {
            writer.event(&TraceEvent {
                name: name.into(),
                ph: "M".into(),
                cat: "__metadata".into(),
                ts: 0.0,
                dur: None,
                tid,
                pid: gpu_processes[&(source_pid, device)],
                args,
                id: None,
                bp: None,
            })?;
            json_event_count += 1;
        }
    }
    for (source_pid, device, kind, tid, source_tid, lane) in device_track_metadata {
        let track_kind = match kind {
            DeviceTrackKind::ProjectedNvtx => "NVTX Kernel",
            DeviceTrackKind::Runtime => "CUDA API",
            DeviceTrackKind::Nvtx => "NVTX Thread",
        };
        let track_name = match lane {
            Some(lane) => format!("{track_kind} {source_tid} / Lane {}", lane + 1),
            None => format!("{track_kind} {source_tid}"),
        };
        writer.event(&TraceEvent {
            name: "thread_name".into(),
            ph: "M".into(),
            cat: "__metadata".into(),
            ts: 0.0,
            dur: None,
            tid,
            pid: gpu_processes[&(source_pid, device)],
            args: json!({"name": track_name}),
            id: None,
            bp: None,
        })?;
        json_event_count += 1;
        writer.event(&TraceEvent {
            name: "thread_sort_index".into(),
            ph: "M".into(),
            cat: "__metadata".into(),
            ts: 0.0,
            dur: None,
            tid,
            pid: gpu_processes[&(source_pid, device)],
            args: json!({"sort_index": tid}),
            id: None,
            bp: None,
        })?;
        json_event_count += 1;
    }

    for (call_idx, call) in runtime
        .iter()
        .enumerate()
        .filter(|(_, call)| call.event_id.is_some())
    {
        let pid_number = (call.global_tid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX;
        let tid_number = call.global_tid % GLOBAL_ID_RADIX;
        let event_id = call
            .event_id
            .as_ref()
            .context("linked CUDA API has no event ID")?;
        let ts_us = ns_to_us(call.start - origin_ns);
        let dur_us = ns_to_us(call.end - call.start);
        let args = json!({
            "correlationId": call.correlation,
            "eventId": event_id,
            "cpuStartNs": call.start,
            "cpuEndNs": call.end,
            "cpuDurationNs": call.end - call.start,
        });
        for &device in &call_devices[&call_idx] {
            let mut device_args = args.clone();
            device_args["deviceId"] = json!(device);
            writer.event(&TraceEvent {
                name: call.name.clone(),
                ph: "X".into(),
                cat: "cuda_api".into(),
                ts: ts_us,
                dur: Some(dur_us),
                tid: device_slice_tracks[&DeviceSliceKey::Runtime(call_idx, device)],
                pid: gpu_processes[&(pid_number, device)],
                args: device_args.clone(),
                id: None,
                bp: None,
            })?;
            json_event_count += 1;
            rows.push(TraceRow {
                report: report.into(),
                event_type: "cuda_api".into(),
                cat: "cuda_api".into(),
                name: call.name.clone(),
                ph: "X".into(),
                ts_us,
                dur_us: Some(dur_us),
                aligned_ts_us: ns_to_us(call.start - anchor_ns),
                pid: format!("CUDA Device {device} / Source PID {pid_number}"),
                tid: format!("CUDA API Thread {tid_number}"),
                args_json: device_args.to_string(),
                event_id: Some(event_id.clone()),
                launch_event_id: None,
                stream_id: None,
                correlation_id: Some(call.correlation as u32),
                stream_sequence: None,
                depends_on_event_id: None,
                dependency_type: None,
            });
        }
    }

    for (kernel_idx, kernel) in kernels.iter().enumerate() {
        let kernel_source_pid = source_pid(kernel.global_pid);
        let launch_event_id = kernel
            .launch_call
            .and_then(|idx| runtime[idx].event_id.as_ref().cloned());
        let args = json!({
            "correlationId": kernel.correlation,
            "contextId": kernel.context,
            "streamId": kernel.stream,
            "streamSequence": kernel.sequence,
            "eventId": kernel.event_id,
            "launchEventId": launch_event_id,
            "gpuStartNs": kernel.start,
            "gpuEndNs": kernel.end,
            "gpuDurationNs": kernel.end - kernel.start,
            "sourceProcessId": kernel_source_pid,
            "deviceId": kernel.device,
            "grid": kernel.grid,
            "block": kernel.block,
            "NVTXRegions": kernel.nvtx_regions,
        });
        let ts_us = ns_to_us(kernel.start - origin_ns);
        let dur_us = ns_to_us(kernel.end - kernel.start);
        let pid_label = format!(
            "CUDA Device {} / Source PID {}",
            kernel.device, kernel_source_pid
        );
        let tid_label = format!("Context {} / Stream {}", kernel.context, kernel.stream);
        let gpu_pid = gpu_processes[&(kernel_source_pid, kernel.device)];
        let gpu_tid = stream_tracks[&(
            kernel_source_pid,
            kernel.device,
            kernel.context,
            kernel.stream,
        )];
        writer.event(&TraceEvent {
            name: kernel.name.clone(),
            ph: "X".into(),
            cat: "cuda".into(),
            ts: ts_us,
            dur: Some(dur_us),
            tid: gpu_tid,
            pid: gpu_pid,
            args: args.clone(),
            id: None,
            bp: None,
        })?;
        json_event_count += 1;
        rows.push(TraceRow {
            report: report.into(),
            event_type: "cuda_kernel".into(),
            cat: "cuda".into(),
            name: kernel.name.clone(),
            ph: "X".into(),
            ts_us,
            dur_us: Some(dur_us),
            aligned_ts_us: ns_to_us(kernel.start - anchor_ns),
            pid: pid_label.clone(),
            tid: tid_label.clone(),
            args_json: args.to_string(),
            event_id: Some(kernel.event_id.clone()),
            launch_event_id: launch_event_id.clone(),
            stream_id: Some(kernel.stream as u64),
            correlation_id: Some(kernel.correlation as u32),
            stream_sequence: Some(kernel.sequence),
            depends_on_event_id: None,
            dependency_type: None,
        });

        if let Some(call_idx) = kernel.launch_call {
            let call = &runtime[call_idx];
            let call_event_id = call
                .event_id
                .as_ref()
                .context("kernel launch call has no event ID")?;
            let call_pid = gpu_pid;
            let call_tid = device_slice_tracks[&DeviceSliceKey::Runtime(call_idx, kernel.device)];
            let flow_id = LAUNCH_FLOW_ID_BASE + kernel_idx as u64;
            let flow_args = json!({
                "from": call_event_id,
                "to": kernel.event_id,
                "correlationId": kernel.correlation,
                "deviceId": kernel.device,
                "streamId": kernel.stream,
                "cpuCallStartNs": call.start,
                "cpuCallEndNs": call.end,
                "gpuKernelStartNs": kernel.start,
                "gpuKernelEndNs": kernel.end,
                "launchLatencyUs": ns_to_us((kernel.start - call.end).max(0)),
                "kernelStartMinusApiEndUs": ns_to_us(kernel.start - call.end),
            });
            let flow_start_ns = flow_start_ns(call, kernel.start);
            writer.event(&TraceEvent {
                name: "cuda_kernel_launch".into(),
                ph: "s".into(),
                cat: "cuda_launch_dependency".into(),
                ts: ns_to_us(flow_start_ns - origin_ns),
                dur: None,
                tid: call_tid,
                pid: call_pid,
                args: flow_args.clone(),
                id: Some(flow_id),
                bp: None,
            })?;
            writer.event(&TraceEvent {
                name: "cuda_kernel_launch".into(),
                ph: "f".into(),
                cat: "cuda_launch_dependency".into(),
                ts: ns_to_us((kernel.start + 1).min(kernel.end) - origin_ns),
                dur: None,
                tid: gpu_tid,
                pid: gpu_pid,
                args: flow_args,
                id: Some(flow_id),
                bp: Some("e".into()),
            })?;
            json_event_count += 2;
        }
    }

    for (copy_idx, copy) in memcpys.iter().enumerate() {
        let copy_source_pid = source_pid(copy.global_pid);
        let gpu_pid = gpu_processes[&(copy_source_pid, copy.device)];
        let gpu_tid = stream_tracks[&(copy_source_pid, copy.device, copy.context, copy.stream)];
        let direction = copy_direction(copy.copy_kind);
        let duration_ns = copy.end - copy.start;
        let bandwidth_gbps = if duration_ns > 0 {
            copy.bytes as f64 / duration_ns as f64
        } else {
            0.0
        };
        let launch_event_id = copy
            .launch_call
            .and_then(|idx| runtime[idx].event_id.as_ref().cloned());
        let args = json!({
            "direction": direction, "copyKindId": copy.copy_kind,
            "bytes": copy.bytes, "durationNs": duration_ns, "bandwidthGBps": bandwidth_gbps,
            "sourceProcessId": copy_source_pid, "deviceId": copy.device,
            "contextId": copy.context, "streamId": copy.stream,
            "correlationId": copy.correlation, "eventId": copy.event_id,
            "launchEventId": launch_event_id,
            "srcMemoryKindId": copy.src_kind, "srcMemoryKind": memory_kind(copy.src_kind),
            "dstMemoryKindId": copy.dst_kind, "dstMemoryKind": memory_kind(copy.dst_kind),
            "srcDeviceId": copy.src_device, "srcContextId": copy.src_context,
            "dstDeviceId": copy.dst_device, "dstContextId": copy.dst_context,
            "graphNodeId": copy.graph_node, "virtualAddress": copy.virtual_address,
            "copyCount": copy.copy_count, "gpuStartNs": copy.start, "gpuEndNs": copy.end,
        });
        let ts_us = ns_to_us(copy.start - origin_ns);
        let dur_us = ns_to_us(duration_ns);
        writer.event(&TraceEvent {
            name: format!("CUDA Memcpy {direction}"),
            ph: "X".into(),
            cat: "cuda_memcpy".into(),
            ts: ts_us,
            dur: Some(dur_us),
            tid: gpu_tid,
            pid: gpu_pid,
            args: args.clone(),
            id: None,
            bp: None,
        })?;
        json_event_count += 1;
        rows.push(TraceRow {
            report: report.into(),
            event_type: format!(
                "cuda_memcpy_{}",
                direction.to_ascii_lowercase().replace(' ', "_")
            ),
            cat: "cuda_memcpy".into(),
            name: format!("CUDA Memcpy {direction}"),
            ph: "X".into(),
            ts_us,
            dur_us: Some(dur_us),
            aligned_ts_us: ns_to_us(copy.start - anchor_ns),
            pid: format!("CUDA Device {} / Source PID {copy_source_pid}", copy.device),
            tid: format!("Context {} / Stream {}", copy.context, copy.stream),
            args_json: args.to_string(),
            event_id: Some(copy.event_id.clone()),
            launch_event_id: launch_event_id.clone(),
            stream_id: Some(copy.stream as u64),
            correlation_id: Some(copy.correlation as u32),
            stream_sequence: None,
            depends_on_event_id: None,
            dependency_type: None,
        });
        if let Some(call_idx) = copy.launch_call {
            let call = &runtime[call_idx];
            let call_event_id = call
                .event_id
                .as_ref()
                .context("memcpy API call has no event ID")?;
            let call_pid = gpu_pid;
            let call_tid = device_slice_tracks[&DeviceSliceKey::Runtime(call_idx, copy.device)];
            let flow_id = MEMCPY_FLOW_ID_BASE + copy_idx as u64;
            let flow_args = json!({
                "from": call_event_id, "to": copy.event_id, "direction": direction,
                "correlationId": copy.correlation, "deviceId": copy.device,
                "streamId": copy.stream, "bytes": copy.bytes,
                "cpuCallStartNs": call.start, "cpuCallEndNs": call.end,
                "gpuCopyStartNs": copy.start, "gpuCopyEndNs": copy.end,
            });
            writer.event(&TraceEvent {
                name: format!(
                    "cuda_memcpy_{}",
                    direction.to_ascii_lowercase().replace(' ', "_")
                ),
                ph: "s".into(),
                cat: "cuda_memcpy_dependency".into(),
                ts: ns_to_us(flow_start_ns(call, copy.start) - origin_ns),
                dur: None,
                tid: call_tid,
                pid: call_pid,
                args: flow_args.clone(),
                id: Some(flow_id),
                bp: None,
            })?;
            writer.event(&TraceEvent {
                name: format!(
                    "cuda_memcpy_{}",
                    direction.to_ascii_lowercase().replace(' ', "_")
                ),
                ph: "f".into(),
                cat: "cuda_memcpy_dependency".into(),
                ts: ns_to_us((copy.start + 1).min(copy.end) - origin_ns),
                dur: None,
                tid: gpu_tid,
                pid: gpu_pid,
                args: flow_args,
                id: Some(flow_id),
                bp: Some("e".into()),
            })?;
            json_event_count += 2;
        }
    }

    // Emit NVTX as timestamp-sorted begin/end stacks. For equal timestamps,
    // close inner ranges first and open outer ranges first. Perfetto can then
    // reconstruct the original push/pop nesting instead of flattening ranges.
    let mut nvtx_boundaries = Vec::new();
    for (range_idx, range) in nvtx.iter().enumerate() {
        let range_devices: Vec<i64> = if range.kernel_bounds.is_empty() {
            process_devices
                .get(&range.pid)
                .into_iter()
                .flatten()
                .copied()
                .collect()
        } else {
            range.kernel_bounds.keys().copied().collect()
        };
        for device in range_devices {
            let pid = gpu_processes[&(range.pid, device)];
            let tid = device_slice_tracks[&DeviceSliceKey::Nvtx(range_idx, device)];
            nvtx_boundaries.push((
                pid,
                tid,
                range.start,
                true,
                range.start,
                range.end,
                range_idx,
                device,
                false,
            ));
            nvtx_boundaries.push((
                pid,
                tid,
                range.end,
                false,
                range.start,
                range.end,
                range_idx,
                device,
                false,
            ));
        }
        for (&device, &(start, end)) in &range.kernel_bounds {
            let pid = gpu_processes[&(range.pid, device)];
            let tid = device_slice_tracks[&DeviceSliceKey::ProjectedNvtx(range_idx, device)];
            nvtx_boundaries.push((pid, tid, start, true, start, end, range_idx, device, true));
            nvtx_boundaries.push((pid, tid, end, false, start, end, range_idx, device, true));
        }
    }
    nvtx_boundaries.sort_unstable_by(|a, b| {
        (a.0, a.1, a.2)
            .cmp(&(b.0, b.1, b.2))
            .then_with(|| match (a.3, b.3) {
                (false, true) => std::cmp::Ordering::Less,
                (true, false) => std::cmp::Ordering::Greater,
                (true, true) => b.5.cmp(&a.5),
                (false, false) => b.4.cmp(&a.4),
            })
            .then_with(|| a.6.cmp(&b.6))
    });
    for (pid, tid, time, is_start, _, _, range_idx, device, projected) in nvtx_boundaries {
        let range = &nvtx[range_idx];
        writer.event(&TraceEvent {
            name: range.name.clone(),
            ph: if is_start { "B" } else { "E" }.into(),
            cat: if projected { "nvtx-kernel" } else { "nvtx" }.into(),
            ts: ns_to_us(time - origin_ns),
            dur: None,
            tid,
            pid,
            args: json!({
                "sourcePid": range.pid,
                "sourceTid": range.tid,
                "deviceId": device,
            }),
            id: None,
            bp: None,
        })?;
        json_event_count += 1;
    }

    for range in nvtx {
        let ts_us = ns_to_us(range.start - origin_ns);
        let dur_us = ns_to_us(range.end - range.start);
        let args = json!({"sourcePid": range.pid, "sourceTid": range.tid});
        let range_devices: Vec<i64> = if range.kernel_bounds.is_empty() {
            process_devices
                .get(&range.pid)
                .into_iter()
                .flatten()
                .copied()
                .collect()
        } else {
            range.kernel_bounds.keys().copied().collect()
        };
        for device in range_devices {
            let mut device_args = args.clone();
            device_args["deviceId"] = json!(device);
            rows.push(TraceRow {
                report: report.into(),
                event_type: "nvtx".into(),
                cat: "nvtx".into(),
                name: range.name.clone(),
                ph: "X".into(),
                ts_us,
                dur_us: Some(dur_us),
                aligned_ts_us: ns_to_us(range.start - anchor_ns),
                pid: format!("CUDA Device {device} / Source PID {}", range.pid),
                tid: format!("NVTX Thread {}", range.tid),
                args_json: device_args.to_string(),
                event_id: None,
                launch_event_id: None,
                stream_id: None,
                correlation_id: None,
                stream_sequence: None,
                depends_on_event_id: None,
                dependency_type: None,
            });
        }

        for (&device, &(kernel_start, kernel_end)) in &range.kernel_bounds {
            let projected_ts_us = ns_to_us(kernel_start - origin_ns);
            let projected_dur_us = ns_to_us(kernel_end - kernel_start);
            rows.push(TraceRow {
                report: report.into(),
                event_type: "nvtx_kernel".into(),
                cat: "nvtx-kernel".into(),
                name: range.name.clone(),
                ph: "X".into(),
                ts_us: projected_ts_us,
                dur_us: Some(projected_dur_us),
                aligned_ts_us: ns_to_us(kernel_start - anchor_ns),
                pid: format!("CUDA Device {device} / Source PID {}", range.pid),
                tid: format!("NVTX Kernel Thread {}", range.tid),
                args_json:
                    json!({"sourcePid": range.pid, "sourceTid": range.tid, "deviceId": device})
                        .to_string(),
                event_id: None,
                launch_event_id: None,
                stream_id: None,
                correlation_id: None,
                stream_sequence: None,
                depends_on_event_id: None,
                dependency_type: None,
            });
        }
    }
    writer.finish()?;
    Ok((rows, dependencies, json_event_count))
}

fn trace_rows_batch(rows: Vec<TraceRow>) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("report", DataType::Utf8, false),
        Field::new("event_type", DataType::Utf8, false),
        Field::new("cat", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("ph", DataType::Utf8, false),
        Field::new("ts_us", DataType::Float64, false),
        Field::new("dur_us", DataType::Float64, true),
        Field::new("aligned_ts_us", DataType::Float64, false),
        Field::new("pid", DataType::Utf8, false),
        Field::new("tid", DataType::Utf8, false),
        Field::new("args_json", DataType::Utf8, false),
        Field::new("event_id", DataType::Utf8, true),
        Field::new("launch_event_id", DataType::Utf8, true),
        Field::new("stream_id", DataType::UInt64, true),
        Field::new("correlation_id", DataType::UInt32, true),
        Field::new("stream_sequence", DataType::UInt64, true),
        Field::new("depends_on_event_id", DataType::Utf8, true),
        Field::new("dependency_type", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.report.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.event_type.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.cat.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.name.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter().map(|r| Some(r.ph.as_str())).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| Some(r.ts_us)).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.dur_us).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|r| Some(r.aligned_ts_us))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.pid.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.tid.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.args_json.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| r.event_id.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| r.launch_event_id.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                rows.iter().map(|r| r.stream_id).collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                rows.iter().map(|r| r.correlation_id).collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                rows.iter().map(|r| r.stream_sequence).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| r.depends_on_event_id.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| r.dependency_type.as_deref())
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(Into::into)
}

fn dependency_batch(rows: Vec<DependencyRow>) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("report", DataType::Utf8, false),
        Field::new("stream_id", DataType::UInt64, false),
        Field::new("stream_sequence", DataType::UInt64, false),
        Field::new("predecessor_event_id", DataType::Utf8, false),
        Field::new("predecessor_kernel", DataType::Utf8, false),
        Field::new("predecessor_ts_us", DataType::Float64, false),
        Field::new("predecessor_dur_us", DataType::Float64, false),
        Field::new("successor_event_id", DataType::Utf8, false),
        Field::new("successor_kernel", DataType::Utf8, false),
        Field::new("successor_ts_us", DataType::Float64, false),
        Field::new("successor_dur_us", DataType::Float64, false),
        Field::new("gap_us", DataType::Float64, false),
        Field::new("dependency_type", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.report.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                rows.iter().map(|r| Some(r.stream_id)).collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                rows.iter()
                    .map(|r| Some(r.stream_sequence))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.predecessor_event_id.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.predecessor_kernel.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|r| Some(r.predecessor_ts_us))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|r| Some(r.predecessor_dur_us))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.successor_event_id.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.successor_kernel.as_str()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|r| Some(r.successor_ts_us))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|r| Some(r.successor_dur_us))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| Some(r.gap_us)).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| Some(r.dependency_type.as_str()))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(Into::into)
}

async fn write_parquet(ctx: &SessionContext, batch: RecordBatch, path: &Path) -> Result<()> {
    prepare_output(path)?;
    ctx.read_batch(batch)?
        .write_parquet(
            path.to_str().context("non-UTF8 output path")?,
            DataFrameWriteOptions::new().with_single_file_output(true),
            None,
        )
        .await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let ctx = SessionContext::new();
    let has_memcpy = register_tables(&ctx, &args.parquet_dir).await?;

    let (kernels_result, nvtx_result, runtime_result) =
        tokio::join!(load_kernels(&ctx), load_nvtx(&ctx), load_runtime(&ctx),);
    let mut kernels = kernels_result?;
    let mut nvtx = nvtx_result?;
    let mut runtime = runtime_result?;
    let mut memcpys = if has_memcpy {
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
        .context("trace contains no kernel or NVTX ranges")?;
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
    let (trace_rows, dependencies, json_events) = emit_outputs(
        &args.report,
        &args.output_json,
        &kernels,
        &memcpys,
        &runtime,
        &nvtx,
        origin_ns,
        anchor_ns,
    )?;
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
        "report={} kernels={} cuda_api={} launch_dependencies={} memcpy={} h2d={} d2h={} d2d={} memcpy_launch_dependencies={} nvtx={} nvtx_kernel={} stream_dependencies={} json_events={} parquet_rows={} anchor_ns={} alignment_anchor={}",
        args.report,
        kernels.len(),
        cuda_api_count,
        launch_dependencies,
        memcpys.len(),
        h2d_count,
        d2h_count,
        d2d_count,
        memcpy_launch_dependencies,
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
