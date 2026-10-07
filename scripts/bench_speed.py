#!/usr/bin/env python3
"""Speed-only re-measurement: waits for an idle GPU, then times each config and
merges the speed fields into its existing results JSON (accuracy is untouched).

Needed on a shared GPU, where accuracy runs are valid under contention but
timing is not. Detectors are timed back to back in one process so they see the
same GPU conditions; the order is repeated twice (A B C A B C) to expose drift.

Usage: python scripts/bench_speed.py configs/baseline.yaml configs/baseline_yolo26m.yaml ...
       [--max-wait-min 120] [--device cuda]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import SportsMOT
from models.detector import Detector
from tracking.tracker import Tracker
from benchmark.timing import (_phys_gpu_index, benchmark_pipeline,
                              wait_for_idle_gpu)

SPEED_KEYS_DROP = {"idle_gate_waited_s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--max-wait-min", type=float, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    cfgs = [yaml.safe_load(open(c)) for c in args.configs]
    dets = {}
    for c in cfgs:
        d = c["detector"]
        dets[c["experiment"]] = Detector(
            d["name"], imgsz=d.get("imgsz", 640), conf=d.get("conf", 0.25),
            device=args.device, classes=d.get("classes"), iou=d.get("iou"))
    ds = SportsMOT(split=cfgs[0]["data"].get("split", "val"))
    frames = ds.frames(ds.sequences()[0])

    gi = _phys_gpu_index()
    deadline = time.time() + args.max_wait_min * 60
    best = {}
    for rnd in range(args.rounds):
        for c in cfgs:
            exp = c["experiment"]
            while True:
                gate = wait_for_idle_gpu(gi, max_wait_s=60)
                if gate["idle_gate_ok"]:
                    break
                if time.time() > deadline:
                    print(f"[speed] gave up waiting for an idle GPU ({exp})", flush=True)
                    return 1
                print(f"[speed] GPU busy, waiting... ({exp})", flush=True)
            s = benchmark_pipeline(dets[exp], lambda n=c["tracker"]["name"]: Tracker(n),
                                   frames, args.device, n_frames=500,
                                   warmup=50, repeats=5)
            s.update({k: v for k, v in gate.items() if k not in SPEED_KEYS_DROP})
            print(f"[speed] round {rnd} {exp}: fps={s['fps']} p50={s['latency_p50_ms']} "
                  f"cv={s['latency_repeat_cv_pct']}% contended={s['contended']}", flush=True)
            if not s["contended"] and (exp not in best or
                                       s["latency_p50_ms"] < best[exp]["latency_p50_ms"]):
                best[exp] = s

    for c in cfgs:
        exp, path = c["experiment"], Path(c["benchmark"]["save_to"])
        if exp not in best or not path.exists():
            print(f"[speed] NO clean timing for {exp}; JSON left unchanged", flush=True)
            continue
        r = json.loads(path.read_text())
        r.update(best[exp])
        r["speed_measured_by"] = "bench_speed.py (idle-gated, best clean of rounds)"
        path.write_text(json.dumps(r, indent=1))
        print(f"[speed] updated {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
