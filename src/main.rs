use std::collections::{BTreeSet, HashMap};
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
    predecessor: Option<usize>,
    nvtx_regions: Vec<String>,
}

#[derive(Debug)]
struct RuntimeCall {
    start: i64,
    end: i64,
    global_tid: i64,
    correlation: i64,
    device: i64,
}

#[derive(Debug)]
struct NvtxRange {
    start: i64,
    end: i64,
    name: String,
    pid: i64,
    tid: i64,
    device: i64,
    kernel_bounds: Option<(i64, i64)>,
}

#[derive(Serialize)]
struct TraceEvent {
    name: String,
    ph: String,
    cat: String,
    ts: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    dur: Option<f64>,
    tid: String,
    pid: String,
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

async fn register_tables(ctx: &SessionContext, dir: &Path) -> Result<()> {
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
    Ok(())
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
                predecessor: None,
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
                device: -1,
                kernel_bounds: None,
            });
        }
    }
    Ok(ranges)
}

async fn load_runtime(ctx: &SessionContext) -> Result<Vec<RuntimeCall>> {
    let sql = r#"
        SELECT
            CAST(start AS BIGINT) AS start_ns,
            CAST("end" AS BIGINT) AS end_ns,
            CAST("globalTid" AS BIGINT) AS global_tid,
            CAST("correlationId" AS BIGINT) AS correlation_id
        FROM runtime
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
                device: -1,
            });
        }
    }
    Ok(calls)
}

fn link_processes_to_devices(
    kernels: &[Kernel],
    nvtx: &mut [NvtxRange],
    runtime: &mut [RuntimeCall],
) -> Result<HashMap<i64, i64>> {
    let mut pid_to_device = HashMap::new();
    for kernel in kernels {
        let pid = (kernel.global_pid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX;
        if let Some(previous) = pid_to_device.insert(pid, kernel.device) {
            if previous != kernel.device {
                bail!(
                    "process {pid} is associated with devices {previous} and {}",
                    kernel.device
                );
            }
        }
    }
    for range in nvtx {
        range.device = *pid_to_device
            .get(&range.pid)
            .ok_or_else(|| anyhow!("NVTX process {} has no CUDA device", range.pid))?;
    }
    for call in runtime {
        let pid = (call.global_tid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX;
        call.device = *pid_to_device
            .get(&pid)
            .ok_or_else(|| anyhow!("CUDA Runtime process {pid} has no CUDA device"))?;
    }
    Ok(pid_to_device)
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum BoundaryKind {
    ApiEnd,
    NvtxEnd,
    ApiStart,
    NvtxStart,
}

fn project_nvtx_to_kernels(
    kernels: &mut [Kernel],
    nvtx: &mut [NvtxRange],
    runtime: &[RuntimeCall],
) {
    let mut kernel_by_correlation = HashMap::new();
    for (idx, kernel) in kernels.iter().enumerate() {
        kernel_by_correlation.insert((kernel.device, kernel.correlation), idx);
    }

    let devices: BTreeSet<i64> = kernels.iter().map(|k| k.device).collect();
    for device in devices {
        let mut boundaries: Vec<(i64, BoundaryKind, usize)> = Vec::new();
        for (idx, range) in nvtx.iter().enumerate().filter(|(_, n)| n.device == device) {
            boundaries.push((range.start, BoundaryKind::NvtxStart, idx));
            boundaries.push((range.end, BoundaryKind::NvtxEnd, idx));
        }
        for (idx, call) in runtime
            .iter()
            .enumerate()
            .filter(|(_, r)| r.device == device)
        {
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
                    let Some(&kernel_idx) = kernel_by_correlation.get(&(device, call.correlation))
                    else {
                        continue;
                    };
                    let kernel_start = kernels[kernel_idx].start;
                    let kernel_end = kernels[kernel_idx].end;
                    for &nvtx_idx in &active_nvtx {
                        let range = &mut nvtx[nvtx_idx];
                        range.kernel_bounds = Some(match range.kernel_bounds {
                            None => (kernel_start, kernel_end),
                            Some((start, end)) => (start.min(kernel_start), end.max(kernel_end)),
                        });
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

fn assign_stream_dependencies(report: &str, kernels: &mut [Kernel]) {
    let mut previous: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut sequence: HashMap<(i64, i64, i64), u64> = HashMap::new();
    for idx in 0..kernels.len() {
        let key = (
            kernels[idx].device,
            kernels[idx].context,
            kernels[idx].stream,
        );
        let seq = sequence.entry(key).or_default();
        *seq += 1;
        kernels[idx].sequence = *seq;
        kernels[idx].event_id = format!(
            "{report}:cuda:{}:{}:{}:{}",
            kernels[idx].device, kernels[idx].context, kernels[idx].stream, seq
        );
        kernels[idx].predecessor = previous.insert(key, idx);
    }
}

fn ns_to_us(ns: i64) -> f64 {
    ns as f64 / 1000.0
}

fn emit_outputs(
    report: &str,
    output_json: &Path,
    kernels: &[Kernel],
    nvtx: &[NvtxRange],
    origin_ns: i64,
    anchor_ns: i64,
) -> Result<(Vec<TraceRow>, Vec<DependencyRow>, usize)> {
    let mut writer = JsonArrayWriter::create(output_json)?;
    let mut rows = Vec::with_capacity(kernels.len() + nvtx.len() * 2);
    let mut dependencies = Vec::with_capacity(kernels.len());
    let mut json_event_count = 0usize;

    for kernel in kernels {
        let predecessor_id = kernel.predecessor.map(|idx| kernels[idx].event_id.clone());
        let args = json!({
            "correlationId": kernel.correlation,
            "contextId": kernel.context,
            "streamId": kernel.stream,
            "streamSequence": kernel.sequence,
            "eventId": kernel.event_id,
            "dependsOnEventId": predecessor_id,
            "dependencyType": predecessor_id.as_ref().map(|_| "same_stream_order"),
            "grid": kernel.grid,
            "block": kernel.block,
            "NVTXRegions": kernel.nvtx_regions,
        });
        let ts_us = ns_to_us(kernel.start - origin_ns);
        let dur_us = ns_to_us(kernel.end - kernel.start);
        let pid = format!("Device {}", kernel.device);
        let tid = format!("Stream {}", kernel.stream);
        writer.event(&TraceEvent {
            name: kernel.name.clone(),
            ph: "X".into(),
            cat: "cuda".into(),
            ts: ts_us,
            dur: Some(dur_us),
            tid: tid.clone(),
            pid: pid.clone(),
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
            pid: pid.clone(),
            tid: tid.clone(),
            args_json: args.to_string(),
            event_id: Some(kernel.event_id.clone()),
            stream_id: Some(kernel.stream as u64),
            correlation_id: Some(kernel.correlation as u32),
            stream_sequence: Some(kernel.sequence),
            depends_on_event_id: predecessor_id.clone(),
            dependency_type: predecessor_id.as_ref().map(|_| "same_stream_order".into()),
        });

        if let Some(previous_idx) = kernel.predecessor {
            let previous = &kernels[previous_idx];
            // Legacy Chrome/Perfetto flow IDs must be numeric. Keep the richer
            // string event IDs in args and Parquet, but use this compact ID for s/f.
            let flow_id = ((kernel.device as u64 & 0xffff) << 48)
                | ((kernel.stream as u64 & 0xffff) << 32)
                | (kernel.sequence & 0xffff_ffff);
            let flow_args = json!({
                "from": previous.event_id,
                "to": kernel.event_id,
                "streamId": kernel.stream,
            });
            writer.event(&TraceEvent {
                name: "same_stream_order".into(),
                ph: "s".into(),
                cat: "cuda_dependency".into(),
                // Flow starts must fall inside the predecessor slice. Chrome/Perfetto
                // slices are half-open, so binding at exactly `end` drops the flow.
                ts: ns_to_us((previous.end - 1).max(previous.start) - origin_ns),
                dur: None,
                tid: format!("Stream {}", previous.stream),
                pid: format!("Device {}", previous.device),
                args: flow_args.clone(),
                id: Some(flow_id),
                bp: None,
            })?;
            writer.event(&TraceEvent {
                name: "same_stream_order".into(),
                ph: "f".into(),
                cat: "cuda_dependency".into(),
                ts: ns_to_us(kernel.start - origin_ns),
                dur: None,
                tid: tid.clone(),
                pid: pid.clone(),
                args: flow_args,
                id: Some(flow_id),
                bp: Some("e".into()),
            })?;
            json_event_count += 2;
            dependencies.push(DependencyRow {
                report: report.into(),
                stream_id: kernel.stream as u64,
                stream_sequence: kernel.sequence,
                predecessor_event_id: previous.event_id.clone(),
                predecessor_kernel: previous.name.clone(),
                predecessor_ts_us: ns_to_us(previous.start - origin_ns),
                predecessor_dur_us: ns_to_us(previous.end - previous.start),
                successor_event_id: kernel.event_id.clone(),
                successor_kernel: kernel.name.clone(),
                successor_ts_us: ts_us,
                successor_dur_us: dur_us,
                gap_us: ns_to_us((kernel.start - previous.end).max(0)),
                dependency_type: "same_stream_order".into(),
            });
        }
    }

    for range in nvtx {
        let pid = format!("Device {}", range.device);
        let tid = format!("NVTX Thread {}", range.tid);
        let ts_us = ns_to_us(range.start - origin_ns);
        let dur_us = ns_to_us(range.end - range.start);
        let args = json!({"sourcePid": range.pid, "sourceTid": range.tid});
        writer.event(&TraceEvent {
            name: range.name.clone(),
            ph: "X".into(),
            cat: "nvtx".into(),
            ts: ts_us,
            dur: Some(dur_us),
            tid: tid.clone(),
            pid: pid.clone(),
            args: args.clone(),
            id: None,
            bp: None,
        })?;
        json_event_count += 1;
        rows.push(TraceRow {
            report: report.into(),
            event_type: "nvtx".into(),
            cat: "nvtx".into(),
            name: range.name.clone(),
            ph: "X".into(),
            ts_us,
            dur_us: Some(dur_us),
            aligned_ts_us: ns_to_us(range.start - anchor_ns),
            pid: pid.clone(),
            tid,
            args_json: args.to_string(),
            event_id: None,
            stream_id: None,
            correlation_id: None,
            stream_sequence: None,
            depends_on_event_id: None,
            dependency_type: None,
        });

        if let Some((kernel_start, kernel_end)) = range.kernel_bounds {
            let projected_tid = format!("NVTX Kernel Thread {}", range.tid);
            let projected_ts_us = ns_to_us(kernel_start - origin_ns);
            let projected_dur_us = ns_to_us(kernel_end - kernel_start);
            writer.event(&TraceEvent {
                name: range.name.clone(),
                ph: "X".into(),
                cat: "nvtx-kernel".into(),
                ts: projected_ts_us,
                dur: Some(projected_dur_us),
                tid: projected_tid.clone(),
                pid: pid.clone(),
                args: json!({}),
                id: None,
                bp: None,
            })?;
            json_event_count += 1;
            rows.push(TraceRow {
                report: report.into(),
                event_type: "nvtx_kernel".into(),
                cat: "nvtx-kernel".into(),
                name: range.name.clone(),
                ph: "X".into(),
                ts_us: projected_ts_us,
                dur_us: Some(projected_dur_us),
                aligned_ts_us: ns_to_us(kernel_start - anchor_ns),
                pid: pid.clone(),
                tid: projected_tid,
                args_json: "{}".into(),
                event_id: None,
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
    register_tables(&ctx, &args.parquet_dir).await?;

    let (kernels_result, nvtx_result, runtime_result) =
        tokio::join!(load_kernels(&ctx), load_nvtx(&ctx), load_runtime(&ctx),);
    let mut kernels = kernels_result?;
    let mut nvtx = nvtx_result?;
    let mut runtime = runtime_result?;
    link_processes_to_devices(&kernels, &mut nvtx, &mut runtime)?;
    project_nvtx_to_kernels(&mut kernels, &mut nvtx, &runtime);
    assign_stream_dependencies(&args.report, &mut kernels);

    let origin_ns = kernels
        .iter()
        .map(|k| k.start)
        .chain(nvtx.iter().map(|n| n.start))
        .min()
        .context("trace contains no kernel or NVTX ranges")?;
    let anchor_ns = nvtx
        .iter()
        .filter(|n| {
            n.name.starts_with("CriticalPath/MeasuredBatch/") && n.name.ends_with("/batch_0")
        })
        .map(|n| n.start)
        .min()
        .context("no CriticalPath/MeasuredBatch/.../batch_0 NVTX anchor")?;

    let projected_nvtx = nvtx.iter().filter(|n| n.kernel_bounds.is_some()).count();
    let (trace_rows, dependencies, json_events) = emit_outputs(
        &args.report,
        &args.output_json,
        &kernels,
        &nvtx,
        origin_ns,
        anchor_ns,
    )?;
    let trace_row_count = trace_rows.len();
    let dependency_count = dependencies.len();
    write_parquet(&ctx, trace_rows_batch(trace_rows)?, &args.output_parquet).await?;
    write_parquet(
        &ctx,
        dependency_batch(dependencies)?,
        &args.output_dependencies,
    )
    .await?;

    println!(
        "report={} kernels={} nvtx={} nvtx_kernel={} dependencies={} json_events={} parquet_rows={} anchor_ns={}",
        args.report,
        kernels.len(),
        nvtx.len(),
        projected_nvtx,
        dependency_count,
        json_events,
        trace_row_count,
        anchor_ns,
    );
    Ok(())
}
