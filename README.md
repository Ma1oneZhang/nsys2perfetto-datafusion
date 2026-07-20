# nsys2perfetto-datafusion

Convert native NVIDIA Nsight Systems Parquet exports into a Perfetto-compatible
timeline and an aligned event table.
The converter reads Parquet with Apache DataFusion and does not use SQLite.

## Features

- CUDA kernel slices with device, context, stream, correlation, grid, and block metadata
- CPU CUDA Runtime launch slices linked to GPU kernel execution with Perfetto flows
- H2D, D2H, and D2D memcpy slices with byte count, memory kinds, bandwidth,
  source/destination device/context details, and API-to-copy flows
- Explicit per-source-process `CUDA HW deviceId` tracks with context/stream lanes showing
  the CUPTI kernel start, end, and duration
- NVTX push/pop ranges and NVTX-to-kernel projection using CUDA Runtime overlap
- CUDA API, CPU NVTX, and GPU-projected NVTX slices merged onto their source CPU thread
- Process-aware multi-GPU tracks so process-local device IDs cannot be conflated
- Aligned event Parquet for DuckDB/DataFusion queries

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

Open the JSON at [ui.perfetto.dev](https://ui.perfetto.dev/). The JSON uses
Chrome Trace Event format and emits `s`/`f` flow pairs with numeric IDs so the
dependency arrows are accepted by Perfetto.

Every matched kernel launch emits a `cuda_launch_dependency` flow from the CPU
CUDA Runtime API slice to the corresponding GPU kernel. Matching uses the
Nsight `(PID, correlationId)` relationship. Memcpy API calls use the same
relationship to link to H2D, D2H, and D2D hardware intervals. Consecutive
kernels on a stream are intentionally not connected.

The Chrome JSON uses numeric process and thread IDs plus `process_name`,
`thread_name`, and sort-index metadata. As a result, CUDA hardware execution
tracks are grouped and displayed before host launch tracks in Perfetto. Select
either a host launch slice or a GPU kernel slice to display their connecting
flow arrow.

The dependency Parquet argument remains for CLI compatibility and produces an
empty, schema-valid table; only observed API-to-GPU correlation flows are shown.

## Requirements

- Rust/Cargo 1.88 or newer
- Native Parquet output from NVIDIA Nsight Systems

## License

Apache-2.0
