# RUNBOOK — zero to first run

All steps below are **CPU-only ($0)** until marked [GPU].

## 0. Setup (15 min, CPU)
```bash
git init && git add . && git commit -m "scaffold"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Data (1-2 hrs, CPU + network)
```bash
bash scripts/download_data.sh
```
Verify: open one SportsMOT clip's first frame, confirm `gt/gt.txt` parses via a 10-line python check.

## 2. Smoke test the plumbing (30 min, CPU) — DONE 2026-10-04

Proven working on this machine's CPU: `python scripts/smoke_cpu.py`
downloads a sample photo, synthesizes a 30-frame clip, runs
YOLO26n -> ByteTrack -> annotate, writes the clip + results JSON.
Result: 30/30 frames tracked, 9.7 FPS on CPU, p95 62.3ms.
ONNX export also proven on CPU (`engines/yolo26n.onnx`, 9.9MB).

Remaining TODOs below are for real data + Week 2 backbones.
Implement the TODOs in this order (smallest testable slice first):
1. `src/data/sportsmot.py` — sequences() + ground_truth()  [IMPLEMENTED]
2. `src/models/detector.py` — yolo26s branch only, `device="cpu"`  [IMPLEMENTED]
3. `src/tracking/tracker.py` — bytetrack branch  [IMPLEMENTED, boxmot]
4. `src/streaming/pipeline.py` — run() on 30 frames  [IMPLEMENTED]
5. `src/benchmark/run_study.py` — wire it together  [IMPLEMENTED]

Then:
```bash
python scripts/run_benchmark.py --config configs/baseline.yaml --mode smoke --cpu
```
Success = a `results/*.json` with FPS numbers (slow on CPU, that's fine) and no crashes.

## 3. UI skeleton (2-3 hrs, CPU)
Fill in `src/ui/app.py` per the comments. Point it at the smoke-test JSON so the
dashboard renders with real (slow) numbers. From here on, the UI is done and
never needs the GPU.

## 4. ONNX export prep (1 hr, CPU)
Implement `Detector.to_onnx()` and run `scripts/export_onnx.py` on CPU.
If export works on CPU, Session B can't get stuck on it.

## 5. [GPU] Session A — baselines (end of Week 1)
On the H100: `run_benchmark.py --config configs/baseline.yaml` (full mode).
Also run the yolo26m + RT-DETR variants. Save every JSON. **Then turn the GPU off.**

## 6. [GPU] Sessions B/C/D — Weeks 2-4
See `docs/GPU_RUN_PLAN.md`. Total budget: ~10-15 H100-hours.

## 7. Ship (CPU)
Demo video edit, benchmark report, blog post, README polish. All CPU.
