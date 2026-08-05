# nsys2perfetto-datafusion

Convert native NVIDIA Nsight Systems Parquet exports into a Perfetto-compatible
timeline and an aligned event table.
The converter reads Parquet with Apache DataFusion and does not use SQLite.

## Features

- CUDA kernel slices with device, context, stream, correlation, grid, and block metadata
- CPU CUDA Runtime launch slices linked to GPU kernel execution with Perfetto flows
- Visible `cudaDeviceSynchronize` and `cudaStreamSynchronize` Runtime slices,
  including versioned and per-thread-default-stream API variants
- H2D, D2H, and D2D memcpy slices with byte count, memory kinds, bandwidth,
  source/destination device/context details, and API-to-copy flows
- Memcpy remains visible on its CUDA HW context/stream and is also projected to
  one per-device `PCIe Usage` lane for combined H2D/D2H occupancy; D2D uses a
  separate `GPU Copy D2D` lane
- Every kernel remains visible on its CUDA HW context/stream and is also
  projected once to an overlap-safe per-device `CUDA Core Timeline`; copy
  activity is never included on this kernel-only timeline
- Every matched CUDA launch has independent API flow arrows to both its
  original stream kernel and its `CUDA Core Timeline` projection
- Every matched H2D/D2H API has independent flow arrows to both its original
  stream memcpy and every corresponding per-device `PCIe Usage` projection;
  D2D remains excluded from PCIe flows
- Explicit per-source-process `CUDA HW deviceId` tracks with context/stream lanes showing
  the CUPTI kernel start, end, and duration
- Overlap-safe CUDA HW context/stream lanes preserve raw CUPTI intervals even
  when legacy instrumentation reports slightly crossing complete events
- NVTX push/pop ranges and NVTX-to-kernel projection using CUDA Runtime overlap
- Device-centric Perfetto hierarchy: every CUDA device contains its HW
  context/stream, NVTX Kernel, CUDA API, and NVTX Thread child tracks
- NVTX and projected NVTX emitted as timestamp-sorted begin/end stacks so
  Perfetto preserves their push/pop parent-child hierarchy
- Projected NVTX keeps nested ranges on one stack and moves only partially
  crossing kernel envelopes to adjacent lanes
- Overlap-safe CUDA API lanes so Perfetto drops no complete-event slices
- Tracks with the same source thread ID grouped as NVTX Kernel, NVTX Thread,
  then CUDA API
- Process-aware multi-GPU tracks so process-local device IDs cannot be conflated
- Dynamic device discovery with no fixed GPU-count limit; GPU-less NVTX/runtime
  processes inherit all devices observed in the trace instead of becoming
  `Device -1`
- Streaming gzip output when `--output-json` ends in `.gz`
- An eight-worker Tokio multi-thread runtime for Parquet parsing and conversion
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
Runtime, memcpy, or NVTX tables used by the trace. Timeline tables are optional
independently: missing categories are skipped, and a missing `StringIds` table
uses stable numeric fallback names. Conversion fails only when no available
table can produce a timeline event.

`aligned_ts_us` uses the first `CriticalPath/MeasuredBatch/.../batch_0` NVTX
range when present, and otherwise falls back to the first trace event.

## Run

Clone this repository and call the binary through Cargo:

```bash
cargo run --locked --release -- \
  --parquet-dir /tmp/report-parquet \
  --report report \
  --output-json /tmp/report.perfetto.json.gz \
  --output-parquet /tmp/report.perfetto.parquet \
  --output-dependencies /tmp/report.kernel_dependencies.parquet
```

Open the JSON or JSON gzip directly at [ui.perfetto.dev](https://ui.perfetto.dev/).
The `.json.gz` is a single compressed JSON stream, not an archive, and contains
no Parquet output. The JSON uses
Chrome Trace Event format and emits `s`/`f` flow pairs with numeric IDs so the
dependency arrows are accepted by Perfetto. Flow IDs are namespaced by the
report name, preventing duplicate flow starts when traces from distinct reports
are merged.

Every matched kernel launch emits a `cuda_launch_dependency` flow from the CPU
CUDA Runtime API slice to the corresponding GPU kernel. Matching uses the
Nsight `(PID, correlationId)` relationship. Memcpy API calls use the same
relationship to link to H2D, D2H, and D2D hardware intervals. Consecutive
kernels on a stream are intentionally not connected.

The Chrome JSON uses numeric process and thread IDs plus `process_name`,
`thread_name`, and sort-index metadata. CUDA APIs and NVTX ranges are associated
with their device through CUPTI correlation and placed under that device rather
than under a separate host process. Select either an API slice or a GPU kernel
or memcpy slice to display their connecting flow arrow.

The dependency Parquet argument remains for CLI compatibility and produces an
empty, schema-valid table; only observed API-to-GPU correlation flows are shown.

## Requirements

- Rust/Cargo 1.88 or newer
- Native Parquet output from NVIDIA Nsight Systems

## Code organization

- `main.rs`: CLI orchestration, shared data models, and conversion summary
- `input.rs`: DataFusion table registration and native Nsight Parquet loading
- `analysis.rs`: CUDA API/activity linking, NVTX projection, device assignment,
  lane allocation, and unit tests
- `perfetto.rs`: track planning plus Chrome Trace JSON events and flows
- `parquet_output.rs`: Arrow schemas and aligned Parquet writers

## License

Apache-2.0
