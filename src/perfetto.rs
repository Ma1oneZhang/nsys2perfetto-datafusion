use super::*;

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

struct JsonArrayWriter {
    writer: JsonOutput,
    first: bool,
}

enum JsonOutput {
    Plain(BufWriter<File>),
    Gzip(Box<GzEncoder<BufWriter<File>>>),
}

impl Write for JsonOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(writer) => writer.write(buf),
            Self::Gzip(writer) => writer.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(writer) => writer.flush(),
            Self::Gzip(writer) => writer.flush(),
        }
    }
}

impl JsonArrayWriter {
    fn create(path: &Path) -> Result<Self> {
        prepare_output(path)?;
        let file = BufWriter::new(File::create(path)?);
        let mut writer = if path.extension().is_some_and(|extension| extension == "gz") {
            JsonOutput::Gzip(Box::new(GzEncoder::new(file, Compression::default())))
        } else {
            JsonOutput::Plain(file)
        };
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
        match self.writer {
            JsonOutput::Plain(mut writer) => writer.flush()?,
            JsonOutput::Gzip(writer) => {
                let mut file = (*writer).finish()?;
                file.flush()?;
            }
        }
        Ok(())
    }
}

pub(super) struct EmitInput<'a> {
    pub(super) report: &'a str,
    pub(super) output_json: &'a Path,
    pub(super) kernels: &'a [Kernel],
    pub(super) memcpys: &'a [Memcpy],
    pub(super) runtime: &'a [RuntimeCall],
    pub(super) nvtx: &'a [NvtxRange],
    pub(super) origin_ns: i64,
    pub(super) anchor_ns: i64,
}

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

type KernelUsageIntervals = BTreeMap<(i64, i64), Vec<(i64, i64, usize)>>;
type DeviceIntervals = BTreeMap<(i64, i64, DeviceTrackKind, i64), Vec<(i64, i64, DeviceSliceKey)>>;

pub(super) fn emit_outputs(
    input: EmitInput<'_>,
) -> Result<(Vec<TraceRow>, Vec<DependencyRow>, usize, usize)> {
    let EmitInput {
        report,
        output_json,
        kernels,
        memcpys,
        runtime,
        nvtx,
        origin_ns,
        anchor_ns,
    } = input;

    let mut writer = JsonArrayWriter::create(output_json)?;
    let mut rows =
        Vec::with_capacity(kernels.len() + memcpys.len() + runtime.len() + nvtx.len() * 2);
    let dependencies = Vec::new();
    let mut json_event_count = 0usize;
    let mut pcie_usage_launch_dependencies = 0usize;

    // A CUDA device ID is process-local. Keep the source process in every GPU
    // track key so two processes' device 0 never collapse onto one device.
    let extra_pids = runtime
        .iter()
        .filter(|call| call.event_id.is_some())
        .map(|call| source_pid(call.global_tid))
        .chain(nvtx.iter().map(|range| range.pid));
    let (process_devices, memcpy_devices) =
        resolve_device_assignments(kernels, memcpys, extra_pids);
    let gpu_keys: BTreeSet<(i64, i64)> = process_devices
        .iter()
        .flat_map(|(&pid, devices)| devices.iter().map(move |&device| (pid, device)))
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
        .chain(memcpys.iter().zip(&memcpy_devices).map(|(copy, devices)| {
            (
                source_pid(copy.global_pid),
                primary_copy_device(copy, devices),
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
    let copy_track_keys: BTreeSet<(i64, i64, &'static str)> = memcpys
        .iter()
        .zip(&memcpy_devices)
        .flat_map(|(copy, devices)| {
            let pid = source_pid(copy.global_pid);
            let track_name = copy_track_name(copy.copy_kind);
            devices.iter().map(move |&device| (pid, device, track_name))
        })
        .collect();
    let copy_tracks: BTreeMap<(i64, i64, &'static str), i64> = copy_track_keys
        .iter()
        .enumerate()
        .map(|(idx, &key)| (key, 500_000_000_i64 + idx as i64))
        .collect();
    let mut kernel_usage_intervals = KernelUsageIntervals::new();
    for (kernel_idx, kernel) in kernels.iter().enumerate() {
        kernel_usage_intervals
            .entry((source_pid(kernel.global_pid), kernel.device))
            .or_default()
            .push((kernel.start, kernel.end, kernel_idx));
    }
    let mut kernel_usage_tracks = BTreeMap::new();
    let mut kernel_usage_track_metadata = Vec::new();
    let mut next_kernel_usage_track = 400_000_000_i64;
    for (&(pid, device), intervals) in &mut kernel_usage_intervals {
        let assignments = allocate_interval_lanes(intervals);
        let lane_count = assignments
            .iter()
            .map(|&(_, lane)| lane + 1)
            .max()
            .unwrap_or(0);
        let lane_tids = (0..lane_count)
            .map(|lane| {
                let tid = next_kernel_usage_track;
                next_kernel_usage_track += 1;
                kernel_usage_track_metadata.push((pid, device, tid, lane));
                tid
            })
            .collect::<Vec<_>>();
        for (kernel_idx, lane) in assignments {
            kernel_usage_tracks.insert(kernel_idx, lane_tids[lane]);
        }
    }
    let mut call_devices: BTreeMap<usize, BTreeSet<i64>> = BTreeMap::new();
    for kernel in kernels {
        if let Some(call_idx) = kernel.launch_call {
            call_devices
                .entry(call_idx)
                .or_default()
                .insert(kernel.device);
        }
    }
    for (copy, devices) in memcpys.iter().zip(&memcpy_devices) {
        if let Some(call_idx) = copy.launch_call {
            call_devices.entry(call_idx).or_default().extend(devices);
        }
    }
    for (call_idx, call) in runtime.iter().enumerate() {
        if call.event_id.is_some() && !call_devices.contains_key(&call_idx) {
            call_devices.entry(call_idx).or_default().insert(-1);
        }
    }
    // Place CUDA API tracks under each CUDA device process. Complete CUDA API
    // events can overlap, so allocate the minimum number of adjacent lanes.
    // NVTX tracks are allocated separately below and use B/E stack events to
    // preserve push/pop hierarchy on one track per source thread.
    let mut device_intervals = DeviceIntervals::new();
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
    for &(source_pid, device, tid, lane) in &kernel_usage_track_metadata {
        let track_name = if lane == 0 {
            "CUDA Core Timeline".into()
        } else {
            format!("CUDA Core Timeline / Lane {}", lane + 1)
        };
        for (name, args) in [
            ("thread_name", json!({"name": track_name})),
            (
                "thread_sort_index",
                json!({"sort_index": -300 + lane as i64}),
            ),
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
    for (&(source_pid, device, track_name), &tid) in &copy_tracks {
        let sort_index = match track_name {
            "PCIe Usage" => -200,
            "GPU Copy D2D" => -190,
            _ => -180,
        };
        for (name, args) in [
            ("thread_name", json!({"name": track_name})),
            ("thread_sort_index", json!({"sort_index": sort_index})),
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
    let source_thread_keys: BTreeSet<(i64, i64, i64)> = device_track_metadata
        .iter()
        .map(|&(pid, device, _, _, source_tid, _)| (pid, device, source_tid))
        .collect();
    let mut source_thread_ranks = BTreeMap::new();
    let mut previous_device = None;
    let mut rank = 0_i64;
    for &(pid, device, source_tid) in &source_thread_keys {
        if previous_device != Some((pid, device)) {
            previous_device = Some((pid, device));
            rank = 0;
        }
        source_thread_ranks.insert((pid, device, source_tid), rank);
        rank += 1;
    }

    for (source_pid, device, kind, tid, source_tid, lane) in device_track_metadata {
        let track_kind = match kind {
            DeviceTrackKind::ProjectedNvtx => "NVTX Kernel",
            DeviceTrackKind::Runtime => "CUDA API",
            DeviceTrackKind::Nvtx => "NVTX Thread",
        };
        let kind_order = match kind {
            DeviceTrackKind::ProjectedNvtx => 0_i64,
            DeviceTrackKind::Nvtx => 1_i64,
            DeviceTrackKind::Runtime => 2_i64,
        };
        let lane_order = lane.unwrap_or(0) as i64;
        let thread_rank = source_thread_ranks[&(source_pid, device, source_tid)];
        let sort_index = 1_000_000 + thread_rank * 100 + kind_order * 20 + lane_order;
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
            args: json!({"sort_index": sort_index}),
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
        writer.event(&TraceEvent {
            name: kernel.name.clone(),
            ph: "X".into(),
            cat: "cuda_kernel_usage".into(),
            ts: ts_us,
            dur: Some(dur_us),
            tid: kernel_usage_tracks[&kernel_idx],
            pid: gpu_pid,
            args: json!({
                "deviceId": kernel.device,
                "contextId": kernel.context,
                "streamId": kernel.stream,
                "correlationId": kernel.correlation,
                "eventId": kernel.event_id,
            }),
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
                args: flow_args.clone(),
                id: Some(flow_id),
                bp: Some("e".into()),
            })?;
            let core_flow_id = CORE_LAUNCH_FLOW_ID_BASE + kernel_idx as u64;
            let mut core_flow_args = flow_args.clone();
            core_flow_args["targetTrack"] = json!("CUDA Core Timeline");
            writer.event(&TraceEvent {
                name: "cuda_core_kernel_launch".into(),
                ph: "s".into(),
                cat: "cuda_core_launch_dependency".into(),
                ts: ns_to_us(flow_start_ns - origin_ns),
                dur: None,
                tid: call_tid,
                pid: call_pid,
                args: core_flow_args.clone(),
                id: Some(core_flow_id),
                bp: None,
            })?;
            writer.event(&TraceEvent {
                name: "cuda_core_kernel_launch".into(),
                ph: "f".into(),
                cat: "cuda_core_launch_dependency".into(),
                ts: ns_to_us((kernel.start + 1).min(kernel.end) - origin_ns),
                dur: None,
                tid: kernel_usage_tracks[&kernel_idx],
                pid: gpu_pid,
                args: core_flow_args,
                id: Some(core_flow_id),
                bp: Some("e".into()),
            })?;
            json_event_count += 4;
        }
    }

    for (copy_idx, (copy, devices)) in memcpys.iter().zip(&memcpy_devices).enumerate() {
        let copy_source_pid = source_pid(copy.global_pid);
        let primary_device = primary_copy_device(copy, devices);
        let track_name = copy_track_name(copy.copy_kind);
        let gpu_pid = gpu_processes[&(copy_source_pid, primary_device)];
        let gpu_tid = stream_tracks[&(copy_source_pid, primary_device, copy.context, copy.stream)];
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
            "sourceProcessId": copy_source_pid, "deviceId": primary_device,
            "activityDeviceId": copy.device, "displayDeviceIds": devices,
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
        for &device in devices {
            writer.event(&TraceEvent {
                name: format!("CUDA Memcpy {direction}"),
                ph: "X".into(),
                cat: "cuda_copy_usage".into(),
                ts: ts_us,
                dur: Some(dur_us),
                tid: copy_tracks[&(copy_source_pid, device, track_name)],
                pid: gpu_processes[&(copy_source_pid, device)],
                args: json!({
                    "direction": direction,
                    "bytes": copy.bytes,
                    "bandwidthGBps": bandwidth_gbps,
                    "deviceId": device,
                    "eventId": copy.event_id,
                }),
                id: None,
                bp: None,
            })?;
            json_event_count += 1;
        }
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
            pid: format!("CUDA Device {primary_device} / Source PID {copy_source_pid}"),
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
            let call_tid = device_slice_tracks[&DeviceSliceKey::Runtime(call_idx, primary_device)];
            let flow_id = MEMCPY_FLOW_ID_BASE + copy_idx as u64;
            let flow_args = json!({
                "from": call_event_id, "to": copy.event_id, "direction": direction,
                "correlationId": copy.correlation, "deviceId": primary_device,
                "activityDeviceId": copy.device, "displayDeviceIds": devices,
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
                args: flow_args.clone(),
                id: Some(flow_id),
                bp: Some("e".into()),
            })?;
            json_event_count += 2;

            if is_pcie_copy(copy.copy_kind) {
                for &device in devices {
                    let pcie_flow_id =
                        PCIE_USAGE_FLOW_ID_BASE + pcie_usage_launch_dependencies as u64;
                    let mut pcie_flow_args = flow_args.clone();
                    pcie_flow_args["deviceId"] = json!(device);
                    pcie_flow_args["targetTrack"] = json!("PCIe Usage");
                    writer.event(&TraceEvent {
                        name: format!(
                            "pcie_usage_{}",
                            direction.to_ascii_lowercase().replace(' ', "_")
                        ),
                        ph: "s".into(),
                        cat: "pcie_usage_dependency".into(),
                        ts: ns_to_us(flow_start_ns(call, copy.start) - origin_ns),
                        dur: None,
                        tid: device_slice_tracks[&DeviceSliceKey::Runtime(call_idx, device)],
                        pid: gpu_processes[&(copy_source_pid, device)],
                        args: pcie_flow_args.clone(),
                        id: Some(pcie_flow_id),
                        bp: None,
                    })?;
                    writer.event(&TraceEvent {
                        name: format!(
                            "pcie_usage_{}",
                            direction.to_ascii_lowercase().replace(' ', "_")
                        ),
                        ph: "f".into(),
                        cat: "pcie_usage_dependency".into(),
                        ts: ns_to_us((copy.start + 1).min(copy.end) - origin_ns),
                        dur: None,
                        tid: copy_tracks[&(copy_source_pid, device, "PCIe Usage")],
                        pid: gpu_processes[&(copy_source_pid, device)],
                        args: pcie_flow_args,
                        id: Some(pcie_flow_id),
                        bp: Some("e".into()),
                    })?;
                    pcie_usage_launch_dependencies += 1;
                    json_event_count += 2;
                }
            }
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
    Ok((
        rows,
        dependencies,
        json_event_count,
        pcie_usage_launch_dependencies,
    ))
}
