use super::*;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum BoundaryKind {
    ApiEnd,
    NvtxEnd,
    ApiStart,
    NvtxStart,
}

pub(super) fn link_runtime_calls_to_gpu_activities(
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

fn is_cuda_sync_api(name: &str) -> bool {
    ["cudaDeviceSynchronize", "cudaStreamSynchronize"]
        .into_iter()
        .any(|prefix| {
            name == prefix
                || name
                    .strip_prefix(prefix)
                    .is_some_and(|suffix| suffix.starts_with('_'))
        })
}

pub(super) fn mark_cuda_sync_calls(report: &str, runtime: &mut [RuntimeCall]) -> usize {
    let mut sync_calls = 0;
    for (call_idx, call) in runtime.iter_mut().enumerate() {
        if !is_cuda_sync_api(&call.name) {
            continue;
        }
        sync_calls += 1;
        if call.event_id.is_none() {
            let pid = source_pid(call.global_tid);
            let tid = call.global_tid % GLOBAL_ID_RADIX;
            call.event_id = Some(format!(
                "{report}:cuda_api_sync:{pid}:{tid}:{}:{call_idx}",
                call.correlation
            ));
        }
    }
    sync_calls
}

pub(super) fn project_nvtx_to_kernels(
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

pub(super) fn assign_kernel_ids(report: &str, kernels: &mut [Kernel]) {
    let mut sequence: HashMap<(i64, i64, i64, i64), u64> = HashMap::new();
    for kernel in kernels {
        let source_pid = (kernel.global_pid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX;
        let key = (source_pid, kernel.device, kernel.context, kernel.stream);
        let seq = sequence.entry(key).or_default();
        *seq += 1;
        kernel.sequence = *seq;
        kernel.event_id = format!(
            "{report}:cuda:{source_pid}:{}:{}:{}:{}",
            kernel.device, kernel.context, kernel.stream, seq
        );
    }
}

pub(super) fn ns_to_us(ns: i64) -> f64 {
    ns as f64 / 1000.0
}

pub(super) fn flow_start_ns(call: &RuntimeCall, activity_start: i64) -> i64 {
    (call.end - 1).min(activity_start - 1).max(call.start)
}

pub(super) fn source_pid(global_pid: i64) -> i64 {
    (global_pid / GLOBAL_ID_RADIX) % GLOBAL_ID_RADIX
}

pub(super) fn copy_direction(copy_kind: i64) -> &'static str {
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

pub(super) fn memory_kind(kind: i64) -> &'static str {
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

pub(super) fn copy_track_name(copy_kind: i64) -> &'static str {
    match copy_kind {
        1 | 2 | 11 | 12 => "PCIe Usage",
        3 | 8 | 13 => "GPU Copy D2D",
        _ => "CUDA Copy Unknown",
    }
}

pub(super) fn is_pcie_copy(copy_kind: i64) -> bool {
    matches!(copy_kind, 1 | 2 | 11 | 12)
}

pub(super) fn copy_device_candidates(copy: &Memcpy) -> BTreeSet<i64> {
    let mut devices = BTreeSet::new();
    if copy.device >= 0 {
        devices.insert(copy.device);
    }
    match copy.copy_kind {
        1 | 11 if copy.dst_device >= 0 => {
            devices.insert(copy.dst_device);
        }
        2 | 12 if copy.src_device >= 0 => {
            devices.insert(copy.src_device);
        }
        3 | 8 | 13 => {
            if copy.src_device >= 0 {
                devices.insert(copy.src_device);
            }
            if copy.dst_device >= 0 {
                devices.insert(copy.dst_device);
            }
        }
        _ => {}
    }
    devices
}

pub(super) fn primary_copy_device(copy: &Memcpy, devices: &BTreeSet<i64>) -> i64 {
    if devices.contains(&copy.device) {
        copy.device
    } else {
        *devices
            .first()
            .expect("resolved memcpy device set is empty")
    }
}

pub(super) fn allocate_interval_lanes<K: Copy + Ord>(
    intervals: &mut [(i64, i64, K)],
) -> Vec<(K, usize)> {
    intervals.sort_unstable_by_key(|&(start, end, key)| (start, end, key));
    let mut lane_ends = Vec::new();
    intervals
        .iter()
        .map(|&(start, end, key)| {
            let lane = lane_ends
                .iter()
                .position(|&lane_end| lane_end <= start)
                .unwrap_or_else(|| {
                    lane_ends.push(i64::MIN);
                    lane_ends.len() - 1
                });
            lane_ends[lane] = end;
            (key, lane)
        })
        .collect()
}

/// Allocate lanes for begin/end events while retaining all hierarchy that can
/// be represented by a stack. Nested and disjoint intervals share a lane;
/// partially crossing intervals move to an adjacent lane.
pub(super) fn allocate_laminar_lanes<K: Copy + Ord>(
    intervals: &mut [(i64, i64, K)],
) -> Vec<(K, usize)> {
    intervals.sort_unstable_by(|a, b| {
        (a.0, std::cmp::Reverse(a.1), a.2).cmp(&(b.0, std::cmp::Reverse(b.1), b.2))
    });

    let mut lane_stacks: Vec<Vec<i64>> = Vec::new();
    intervals
        .iter()
        .map(|&(start, end, key)| {
            let lane = lane_stacks
                .iter_mut()
                .position(|stack| {
                    while stack.last().is_some_and(|&active_end| active_end <= start) {
                        stack.pop();
                    }
                    stack.last().is_none_or(|&parent_end| end <= parent_end)
                })
                .unwrap_or_else(|| {
                    lane_stacks.push(Vec::new());
                    lane_stacks.len() - 1
                });
            lane_stacks[lane].push(end);
            (key, lane)
        })
        .collect()
}

pub(super) fn resolve_device_assignments(
    kernels: &[Kernel],
    memcpys: &[Memcpy],
    extra_pids: impl IntoIterator<Item = i64>,
) -> (BTreeMap<i64, BTreeSet<i64>>, Vec<BTreeSet<i64>>) {
    let mut process_devices: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    let mut global_devices = BTreeSet::new();

    for kernel in kernels.iter().filter(|kernel| kernel.device >= 0) {
        let pid = source_pid(kernel.global_pid);
        process_devices
            .entry(pid)
            .or_default()
            .insert(kernel.device);
        global_devices.insert(kernel.device);
    }
    for copy in memcpys {
        let pid = source_pid(copy.global_pid);
        for device in copy_device_candidates(copy) {
            process_devices.entry(pid).or_default().insert(device);
            global_devices.insert(device);
        }
    }

    let copy_devices = memcpys
        .iter()
        .map(|copy| {
            let pid = source_pid(copy.global_pid);
            let candidates = copy_device_candidates(copy);
            if !candidates.is_empty() {
                candidates
            } else if let Some(devices) = process_devices.get(&pid).filter(|set| !set.is_empty()) {
                devices.clone()
            } else if !global_devices.is_empty() {
                global_devices.clone()
            } else {
                BTreeSet::from([-1])
            }
        })
        .collect::<Vec<_>>();

    for (copy, devices) in memcpys.iter().zip(&copy_devices) {
        process_devices
            .entry(source_pid(copy.global_pid))
            .or_default()
            .extend(devices);
    }
    for pid in extra_pids {
        if process_devices.get(&pid).is_some_and(|set| !set.is_empty()) {
            continue;
        }
        process_devices.insert(
            pid,
            if global_devices.is_empty() {
                BTreeSet::from([-1])
            } else {
                global_devices.clone()
            },
        );
    }
    (process_devices, copy_devices)
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

    #[test]
    fn assigns_unknown_process_to_every_discovered_device() {
        let kernels = (0..4)
            .map(|device| test_kernel(42, device, 10 + device, 15 + device))
            .collect::<Vec<_>>();
        let (process_devices, copy_devices) = resolve_device_assignments(&kernels, &[], [7_i64]);

        assert!(copy_devices.is_empty());
        assert_eq!(
            process_devices[&7],
            BTreeSet::from([0_i64, 1_i64, 2_i64, 3_i64])
        );
    }

    #[test]
    fn combines_host_transfers_on_pcie_usage_track() {
        assert_eq!(copy_track_name(1), "PCIe Usage");
        assert_eq!(copy_track_name(2), "PCIe Usage");
        assert_eq!(copy_track_name(11), "PCIe Usage");
        assert_eq!(copy_track_name(12), "PCIe Usage");
        assert_eq!(copy_track_name(8), "GPU Copy D2D");
        assert!([1, 2, 11, 12].into_iter().all(is_pcie_copy));
        assert!([3, 8, 13].into_iter().all(|kind| !is_pcie_copy(kind)));
    }

    #[test]
    fn allocates_overlapping_kernel_timeline_lanes_without_duplication() {
        let mut intervals = vec![(0, 10, 0), (5, 7, 1), (7, 12, 2), (12, 15, 3)];
        let assignments = allocate_interval_lanes(&mut intervals);

        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments, vec![(0, 0), (1, 1), (2, 1), (3, 0)]);
    }

    #[test]
    fn keeps_nested_intervals_together_and_separates_partial_crossings() {
        let mut intervals = vec![
            (0, 10, "outer"),
            (2, 5, "nested"),
            (4, 12, "crossing"),
            (12, 15, "disjoint"),
            (0, 8, "same_start_inner"),
        ];
        let assignments = allocate_laminar_lanes(&mut intervals);

        assert_eq!(
            assignments,
            vec![
                ("outer", 0),
                ("same_start_inner", 0),
                ("nested", 0),
                ("crossing", 1),
                ("disjoint", 0),
            ]
        );
    }

    #[test]
    fn marks_device_and_stream_synchronization_calls_as_visible() {
        let mut runtime = vec![
            RuntimeCall {
                start: 10,
                end: 20,
                global_tid: 7 * GLOBAL_ID_RADIX + 11,
                correlation: 1,
                name: "cudaDeviceSynchronize".into(),
                event_id: None,
            },
            RuntimeCall {
                start: 30,
                end: 40,
                global_tid: 7 * GLOBAL_ID_RADIX + 11,
                correlation: 2,
                name: "cudaStreamSynchronize_ptsz".into(),
                event_id: None,
            },
            RuntimeCall {
                start: 50,
                end: 60,
                global_tid: 7 * GLOBAL_ID_RADIX + 11,
                correlation: 3,
                name: "cudaLaunchKernel".into(),
                event_id: None,
            },
        ];

        assert_eq!(mark_cuda_sync_calls("report", &mut runtime), 2);
        assert!(runtime[0].event_id.is_some());
        assert!(runtime[1].event_id.is_some());
        assert!(runtime[2].event_id.is_none());
    }
}
