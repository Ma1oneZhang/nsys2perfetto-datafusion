use super::*;

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

pub(super) async fn register_tables(ctx: &SessionContext, dir: &Path) -> Result<TableAvailability> {
    let tables = [
        ("kernels", "CUPTI_ACTIVITY_KIND_KERNEL.parquet"),
        ("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME.parquet"),
        ("strings", "StringIds.parquet"),
        ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY.parquet"),
        ("nvtx", "NVTX_EVENTS.parquet"),
    ];
    let mut availability = TableAvailability {
        kernels: false,
        runtime: false,
        memcpy: false,
        nvtx: false,
        strings: false,
    };
    for (name, file) in tables {
        let path = dir.join(file);
        if !path.is_file() {
            continue;
        }
        ctx.register_parquet(
            name,
            path.to_str().context("non-UTF8 Parquet path")?,
            ParquetReadOptions::default(),
        )
        .await?;
        match name {
            "kernels" => availability.kernels = true,
            "runtime" => availability.runtime = true,
            "strings" => availability.strings = true,
            "memcpy" => availability.memcpy = true,
            "nvtx" => availability.nvtx = true,
            _ => unreachable!(),
        }
    }
    if !availability.strings {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        ctx.register_batch("strings", RecordBatch::new_empty(schema))?;
    }
    Ok(availability)
}

pub(super) async fn load_kernels(ctx: &SessionContext) -> Result<Vec<Kernel>> {
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

pub(super) async fn load_nvtx(ctx: &SessionContext) -> Result<Vec<NvtxRange>> {
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

pub(super) async fn load_runtime(ctx: &SessionContext) -> Result<Vec<RuntimeCall>> {
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

pub(super) async fn load_memcpys(ctx: &SessionContext) -> Result<Vec<Memcpy>> {
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
