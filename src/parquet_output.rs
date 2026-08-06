use super::*;

pub(super) fn trace_rows_batch(rows: Vec<TraceRow>) -> Result<RecordBatch> {
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
        Field::new("device_id", DataType::Int64, true),
        Field::new("metric_id", DataType::UInt32, true),
        Field::new("metric_value", DataType::Int64, true),
        Field::new("metric_unit", DataType::Utf8, true),
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
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.device_id).collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                rows.iter().map(|r| r.metric_id).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.metric_value).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| r.metric_unit.as_deref())
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(Into::into)
}

pub(super) fn dependency_batch(rows: Vec<DependencyRow>) -> Result<RecordBatch> {
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

pub(super) async fn write_parquet(
    ctx: &SessionContext,
    batch: RecordBatch,
    path: &Path,
) -> Result<()> {
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
