# GPU RUN PLAN — the money doc

**Principle: the GPU is a measurement instrument, not a dev machine.**
You write code on CPU. You open the GPU for four batched sessions, take your
measurements, save the JSONs, and close it. The Streamlit UI replays cached
results, so demos never cost a cent.

## What NEVER needs GPU (do all of this on CPU, $0)

- venv setup, `pip install`, all code in `src/`
- `scripts/download_data.sh` (network-bound)
- annotation parsing, clip cutting with ffmpeg
- ONNX **export** (`export_onnx.py` runs on CPU; only the engine *build* needs GPU)
- Streamlit UI dev + all chart code (reads `results/*.json`)
- Smoke tests (`--mode smoke --cpu`, 30 frames)
- Writing the benchmark report and blog post

## The four GPU sessions (total ~10-15 H100-hours)

### Session A — Baselines (end of Week 1) · ~2-3 h
- `run_benchmark.py --config configs/baseline.yaml` (full mode, SportsMOT test)
- Repeat for yolo26m and RT-DETR detector variants
- Goal: the three baseline rows (FPS, p50/p95, mAP, MOTA) everything else is compared against
- **Do NOT proceed until baselines are saved.** Every later claim references them.

### Session B — TensorRT engine builds (start of Week 3) · ~3-4 h
- `python scripts/build_trt.py --config configs/tensorrt.yaml`
- Builds 9 engines: {yolo26s, mobilevit_s, tinyvit_5m} x {FP16, FP8, INT8}
- FP8/INT8 need the 500-image calibration set (prepared on CPU beforehand)
- Engines are H100-specific: **back them up** (`engines/`), they are reused forever
- If a build fails, debug the *export* on CPU first, then rebuild

### Session C — The tradeoff study (mid Week 3) · ~4-6 h
- Full cross product on real match video (SoccerNet clips):
  backbones {yolo26s, yolo26m, mobilevit_s, tinyvit_5m} x precisions {FP32-PT, FP16, FP8, INT8}
  x caching {off, k=3} x batch {1, 4}
- This single session produces every number in the report. Batch it; don't dribble.
- Save **everything** to `results/` with config hashes in filenames

### Session D — Demo capture (Week 4) · ~1-2 h
- Run the best config live on a broadcast clip, record the annotated output
  with the FPS counter overlay (raw vs tracked side-by-side)
- This 2-minute video is what goes on the site and in front of committees

## Cost-control rules

1. **Never iterate on GPU.** New code gets smoke-tested on CPU first, always.
2. **One question per session.** Session C answers the tradeoff study; don't also
   "quickly try" something else — that's how 15 hours becomes 40.
3. **Fine-tuning is OPTIONAL and deferred.** Start from pretrained weights. Only
   fine-tune on soccer data if the accuracy gap in Session A demands it, and
   budget it as a separate, deliberate session.
4. **LaunchPad is free until November** — prefer it for Sessions A/B; keep your
   H100 hours for C/D if you need to split.
5. After each session: `git commit` the configs + push `results/*.json`. Numbers
   you didn't save are numbers you paid for twice.
