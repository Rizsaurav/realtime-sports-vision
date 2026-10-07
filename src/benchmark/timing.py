"""Latency / energy / contention measurement for the benchmark.

Method: pre-decode N frames, warm up, then time `repeats` full passes with CUDA
synchronisation around the detector; report percentiles pooled over repeats and
the spread between repeats. A background NVML sampler plus foreign-PID checks
flag runs that shared the GPU, because on a shared machine a clean-looking
number can be silently wrong. Energy comes from Zeus (NVML energy counter).
"""

import os
import statistics
import threading
import time

import cv2
import numpy as np

# SW power cap | HW slowdown | SW thermal | HW thermal | HW power brake
_BAD_THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80


def _phys_gpu_index():
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return int(first) if first.isdigit() else 0


def _nvml_handle(index):
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception:
        return None, None


def foreign_pids(index):
    nv, h = _nvml_handle(index)
    if nv is None:
        return set()
    try:
        return {p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h)} - {os.getpid()}
    except Exception:
        return set()


class GpuSampler:
    def __init__(self, index, period=0.2):
        self.nv, self.h = _nvml_handle(index)
        self.period, self.rows, self._stop = period, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                u = self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu
                w = self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
                c = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
                try:
                    thr = self.nv.nvmlDeviceGetCurrentClocksThrottleReasons(self.h)
                except Exception:
                    thr = 0
                self.rows.append((u, w, c, thr))
            except Exception:
                pass
            time.sleep(self.period)

    def start(self):
        if self.nv is not None:
            self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self.nv is not None:
            self._t.join(timeout=2)
        if not self.rows:
            return {}
        a = np.array(self.rows, dtype=float)
        return {"gpu_util_mean": round(a[:, 0].mean(), 1),
                "gpu_util_max": float(a[:, 0].max()),
                "power_w_mean": round(a[:, 1].mean(), 1),
                "sm_clock_min_mhz": float(a[:, 2].min()),
                "throttled": bool(int(np.bitwise_or.reduce(a[:, 3].astype(int))) & _BAD_THROTTLE_MASK)}


def wait_for_idle_gpu(index, max_wait_s=120, util_thr=5, need=5):
    """Block until `need` consecutive NVML samples show util < util_thr (others idle)."""
    nv, h = _nvml_handle(index)
    if nv is None:
        return {"idle_gate_ok": None, "idle_gate_waited_s": 0.0}
    t0, ok_run, powers = time.time(), 0, []
    while time.time() - t0 < max_wait_s:
        u = nv.nvmlDeviceGetUtilizationRates(h).gpu
        powers.append(nv.nvmlDeviceGetPowerUsage(h) / 1000.0)
        ok_run = ok_run + 1 if u < util_thr else 0
        if ok_run >= need:
            return {"idle_gate_ok": True, "idle_gate_waited_s": round(time.time() - t0, 1),
                    "idle_power_w": round(statistics.mean(powers[-need:]), 1)}
        time.sleep(1.0)
    return {"idle_gate_ok": False, "idle_gate_waited_s": round(time.time() - t0, 1)}


def _sync(device):
    if device != "cpu":
        import torch
        torch.cuda.synchronize()


def benchmark_pipeline(detector, make_tracker, frame_paths, device,
                       n_frames=500, warmup=50, repeats=5):
    on_gpu = device != "cpu"
    gi = _phys_gpu_index() if on_gpu else None
    out = {}

    t0 = time.perf_counter()
    imgs = [cv2.imread(str(p)) for p in frame_paths[:n_frames]]
    out["decode_ms_mean"] = round((time.perf_counter() - t0) * 1000 / len(imgs), 3)
    n = len(imgs)

    if on_gpu:
        out.update(wait_for_idle_gpu(gi))
        pids_before = foreign_pids(gi)
        sampler = GpuSampler(gi).start()
        try:
            from zeus.monitor import ZeusMonitor
            zeus = ZeusMonitor(gpu_indices=[gi], sync_execution_with="torch")
        except Exception:
            zeus = None

    trk = make_tracker()
    for img in imgs[:warmup]:
        trk.update(detector.infer(img), img)
    _sync(device)

    det_ms, trk_ms, rep_medians, energies = [], [], [], []
    for r in range(repeats):
        trk = make_tracker()
        d_r, t_r = [], []
        if on_gpu and zeus:
            zeus.begin_window(f"rep{r}")
        for img in imgs:
            _sync(device)
            a = time.perf_counter()
            dets = detector.infer(img)
            _sync(device)
            b = time.perf_counter()
            trk.update(dets, img)
            c = time.perf_counter()
            d_r.append((b - a) * 1000)
            t_r.append((c - b) * 1000)
        if on_gpu and zeus:
            energies.append(zeus.end_window(f"rep{r}").total_energy)
        det_ms += d_r
        trk_ms += t_r
        rep_medians.append(statistics.median(np.add(d_r, t_r)))

    total = np.add(det_ms, trk_ms)
    p50, p95, p99 = (float(np.percentile(total, q)) for q in (50, 95, 99))
    cv = float(np.std(rep_medians) / np.mean(rep_medians) * 100)
    out.update({
        "fps": round(1000.0 / p50, 2),
        "fps_e2e_with_decode": round(1000.0 / (p50 + out["decode_ms_mean"]), 2),
        "latency_p50_ms": round(p50, 3), "latency_p95_ms": round(p95, 3),
        "latency_p99_ms": round(p99, 3),
        "latency_detect_p50_ms": round(float(np.percentile(det_ms, 50)), 3),
        "latency_track_p50_ms": round(float(np.percentile(trk_ms, 50)), 3),
        "latency_repeat_medians_ms": [round(float(m), 3) for m in rep_medians],
        "latency_repeat_cv_pct": round(cv, 2),
        "timing_frames": n, "timing_repeats": repeats, "timing_warmup": warmup,
    })

    contended = cv > 5.0 or (p99 / p50) > 1.5
    if on_gpu:
        stats = sampler.stop()
        new_pids = foreign_pids(gi) - pids_before
        out.update(stats)
        out["new_foreign_pids"] = sorted(new_pids)
        contended = contended or bool(new_pids) or stats.get("throttled", False)
        if energies:
            out["energy_j_per_frame"] = round(statistics.median(energies) / n, 4)
            out["energy_note"] = ("whole-GPU NVML energy counter; includes any other "
                                  "tenants and idle draw (see idle_power_w)")
    out["contended"] = bool(contended)
    return out
