# Sideline — Real-time Efficient Sports Vision

**One-liner:** professional-grade soccer tracking (players + ball, live) running in real time on a single commodity GPU, with a published latency/throughput/accuracy tradeoff study on an H100.

**The problem it solves:** pro tracking (Stats Perform, Second Spectrum) costs clubs tens to hundreds of thousands per season. Lower leagues, academies, and women's football get nothing. This pipeline delivers the same tracking intelligence at a fraction of the compute cost: efficient ViT backbones, frame-level feature caching, and TensorRT engines, all measured.

**The research contribution:** no existing end-to-end repo pairs an efficient-ViT backbone with streaming frame caching *and* publishes H100 numbers. We do three things nobody has put together: (1) backbone-swap experiment (CNN vs MobileViT/TinyViT) in a tracking pipeline, (2) YOLOV-style frame feature caching on that backbone, (3) TensorRT FP16/FP8/INT8 engines + the full FPS-vs-mAP/MOTA tradeoff study on real broadcast footage.

## Repo layout

```
configs/            experiment configs (one YAML per study arm)
src/
  data/             SoccerNet + SportsMOT loaders (CPU)
  models/           detector wrappers: YOLO26, RT-DETR, efficient-ViT variants
  tracking/         ByteTrack / OC-SORT wrappers
  streaming/        frame cache + real-time pipeline loop (the novel bit lives here)
  benchmark/        metrics (FPS, p50/p95 latency, mAP, MOTA, VRAM, watts) + study runner
  ui/               Streamlit demo app (replay-first: no GPU needed to browse results)
scripts/            download_data.sh, export_onnx.py, build_trt.py (GPU), run_benchmark.py
docs/
  RUNBOOK.md        zero to first run, step by step
  GPU_RUN_PLAN.md   THE MONEY DOC: exactly what needs GPU, when, and for how long
  TRADEOFF_STUDY.md what we measure, the experiment matrix, what each arm answers
```

## Quickstart (CPU only, $0)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU torch first
pip install -r requirements.txt
bash scripts/download_data.sh          # SoccerNet-Tracking + SportsMOT (~GBs, CPU/network)

# Prove the plumbing with zero dataset and zero GPU:
python scripts/smoke_cpu.py            # downloads a ~100KB sample photo + yolo26n weights once,
                                       # synthesizes a 30-frame clip, runs detect->track->annotate,
                                       # writes results/smoke_cpu/ + prints the latency report

# With real data:
python scripts/run_benchmark.py --config configs/baseline.yaml --mode smoke --cpu
streamlit run src/ui/app.py            # UI in replay mode, no GPU needed
```

**Status (2026-10-04):** everything above is implemented and smoke-tested on CPU
in this repo: SportsMOT parser, SoccerNet helpers, YOLO26/RT-DETR detector
wrapper, ByteTrack/OC-SORT wrapper (with built-in IoU fallback if boxmot is
missing), streaming pipeline with annotation + video writer, frame-cache
scaffold, mAP/MOTA/p50/p95 metrics, study runner, and the full Streamlit UI.
`results/*.json` ships with sample numbers so the UI renders immediately;
they are overwritten by real GPU measurements. Week 2 work (efficient-ViT
backbones in `src/models/efficient_vit.py`) is scaffolded, not yet implemented.

## GPU usage (read this before spending a dollar)

**Rule: the GPU is a measurement instrument, not a dev machine.** All code, configs, UI, and data prep happen on CPU. The GPU is powered on for four batched sessions only. Full plan in `docs/GPU_RUN_PLAN.md`; total budget is roughly **10-15 H100-hours** if you batch properly.

| Session | What | Est. H100 time |
|---|---|---|
| A | Baseline detector benchmarks (YOLO26-s/m, RT-DETR) on SportsMOT clips | 2-3 h |
| B | ONNX export + TensorRT engine builds (FP16/FP8/INT8) + calibration | 3-4 h |
| C | Full tradeoff study: backbones x precisions x caching on match video | 4-6 h |
| D | Demo video capture (real-time annotated broadcast clip) | 1-2 h |

Everything the UI shows comes from **cached results JSON** — you never re-run GPU to demo.

## 4-week roadmap

- **Week 1:** baseline. YOLO26 + ByteTrack on SportsMOT/SoccerNet-Tracking, PyTorch. Record mAP/MOTA + FPS baselines. (Mostly CPU dev; Session A at week's end.)
- **Week 2:** the novel bit. Efficient-ViT backbone swap + frame feature caching in `src/streaming/cache.py`. Measure the delta vs baseline.
- **Week 3:** TensorRT. Build FP16/FP8/INT8 engines (Sessions B+C), run the full tradeoff study, fill `docs/TRADEOFF_STUDY.md` results tables.
- **Week 4:** ship. Demo video, benchmark report with charts, technical blog post, clean README. This is what goes in the SOP.

## Outputs (what "done" looks like)

1. GitHub repo with reproducible benchmarks (one command replays the study from cached engines... or rebuilds them)
2. `results/` JSON: every FPS/latency/accuracy number, so nothing is ever re-run
3. Benchmark report (PDF-ish markdown) with tradeoff curves
4. 2-minute demo video: raw broadcast clip vs tracked output, side by side, with live FPS counter
5. Technical blog post on sauravrijal.onrender.com

## License note

ultralytics (YOLO) is AGPL-3.0. Fine for a portfolio/grad-school project and for publishing results. If this ever becomes a commercial product, the detector gets swapped. Your novel code (caching, pipeline, benchmark) is yours; license it MIT.
