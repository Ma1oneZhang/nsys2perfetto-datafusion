# nsys2perfetto-datafusion

Convert native NVIDIA Nsight Systems Parquet exports into a Perfetto-compatible
timeline, an aligned event table, and an explicit CUDA stream dependency table.
The converter reads Parquet with Apache DataFusion and does not use SQLite.

## Features

- CUDA kernel slices with device, context, stream, correlation, grid, and block metadata
- CPU CUDA Runtime launch slices linked to GPU kernel execution with Perfetto flows
- NVTX push/pop ranges projected through same-thread CUDA Runtime launches by default
- Multi-GPU processes, with projected NVTX tracks separated by CUDA device
- Numeric Perfetto flow events between consecutive kernels on each CUDA stream
- Aligned event Parquet for DuckDB/DataFusion queries
- A separate dependency Parquet with predecessor/successor kernel IDs and stream gaps

## Input

First export an `.nsys-rep` with a recent Nsight Systems CLI:

```bash
nsys export \
  --type=parquetdir \
  --ts-normalize=true \
  --force-overwrite=true \
  --output=/tmp/report-parquet \
  report.nsys-rep
```

The Parquet directory must include the native Nsight string, CUDA kernel, CUDA
Runtime, and NVTX tables used by the trace.

`aligned_ts_us` uses the first `CriticalPath/MeasuredBatch/.../batch_0` NVTX
range when present, and otherwise falls back to the first trace event.

## Run

Clone this repository and call the binary through Cargo:

```bash
cargo run --locked --release -- \
  --parquet-dir /tmp/report-parquet \
  --report report \
  --output-json /tmp/report.perfetto.json \
  --output-parquet /tmp/report.perfetto.parquet \
  --output-dependencies /tmp/report.kernel_dependencies.parquet
```

Push/pop NVTX ranges are thread-local, so the default
`--nvtx-projection thread` mode only associates a range with CUDA Runtime
launches on the same Nsight `globalTid`. Use `--nvtx-projection process` only
to reproduce the legacy process-wide `nsys2json` overlap behavior.

Open the JSON at [ui.perfetto.dev](https://ui.perfetto.dev/). The JSON uses
Chrome Trace Event format and emits `s`/`f` flow pairs with numeric IDs so the
dependency arrows are accepted by Perfetto.

Every matched kernel launch emits a `cuda_launch_dependency` flow from the CPU
CUDA Runtime API slice to the corresponding GPU kernel. Matching uses the
Nsight `(PID, correlationId)` relationship. Same-stream kernel ordering remains
available separately as `cuda_dependency` flows.

## Stream dependencies

Dependencies are ordering edges between consecutive kernels on the same CUDA
stream. Each kernel event includes its stream ID and sequence number. The event
Parquet contains `depends_on_event_id`, while the dependency Parquet records the
predecessor and successor event IDs, kernel names, timestamps, durations, and
the inter-kernel gap.

## Requirements

- Rust/Cargo 1.88 or newer
- Native Parquet output from NVIDIA Nsight Systems

## License

Apache-2.0
