# TRADEOFF STUDY — what we actually get to study

This is the scientific core of the project. Each arm below is one row in the
final report's tables and one point-cloud on the tradeoff curves.

## Metrics (every arm reports all of these)

| Metric | What it tells us | Tool |
|---|---|---|
| FPS | real-time viability (target: >=30) | pipeline latency trace |
| latency p50 / p95 | typical vs worst-case frame time (p95 is the honest number) | pipeline latency trace |
| mAP@0.5 | detection accuracy (players + ball) | SportsMOT gt |
| MOTA | tracking accuracy (ID switches kill this) | motmetrics |
| VRAM (MB) | deployment cost | pynvml |
| watts | power cost (ties to your inference-dyno story) | pynvml |

## Experiment arms

**Arm 0 — Baselines** (`configs/baseline.yaml`)
YOLO26-s, YOLO26-m, RT-DETR, PyTorch FP32, no caching.
*Answers: what does the standard stack get us on soccer footage?*

**Arm 1 — Efficient-ViT swap** (`configs/efficient_vit.yaml`, cache off)
Same pipeline, backbone -> MobileViT-S, then TinyViT-5M.
*Answers: how much accuracy do we trade for a lighter backbone?*

**Arm 2 — Frame caching** (same config, cache on, k=3)
YOLOV-style keyframe feature caching on each backbone.
*Answers: the headline result — FPS gained per point of MOTA lost. This curve
is the thing no public repo has published.*

**Arm 3 — Precision** (`configs/tensorrt.yaml`)
Each backbone x {FP16, FP8, INT8} via TensorRT, bs=1 and bs=4.
*Answers: on H100, is FP8 "free" accuracy? Where does INT8 break
(small ball detections are the canary)? bs=1 vs bs=4 separates
streaming latency from throughput ceiling.*

## The three curves in the report

1. **FPS vs MOTA**, points colored by backbone, shaped by precision — the money plot
2. **p95 latency vs batch size** — why bs=1 is the only honest streaming number
3. **Accuracy vs precision** per backbone — where quantization hurts and by how much

## What "done studying" means

Fill the results tables below with the Session C numbers. If Arm 2 shows
>=1.5x FPS for <2 points of MOTA loss on any backbone, that's the headline
for the blog post and the SOP. If INT8 collapses ball detection, that's
finding #2 (quantization sensitivity of small objects in sports).

### Results (fill after Session C)

| Arm | Backbone | Precision | Cache | FPS | p95 (ms) | mAP | MOTA | VRAM | W |
|---|---|---|---|---|---|---|---|---|---|
| 0 | yolo26s | FP32-PT | off | | | | | | |
| 0 | yolo26m | FP32-PT | off | | | | | | |
| 0 | rtdetr  | FP32-PT | off | | | | | | |
| 1 | mobilevit_s | FP32-PT | off | | | | | | |
| 1 | tinyvit_5m  | FP32-PT | off | | | | | | |
| 2 | mobilevit_s | FP32-PT | k=3 | | | | | | |
| 2 | tinyvit_5m  | FP32-PT | k=3 | | | | | | |
| 3 | ... | FP16/FP8/INT8 | ... | | | | | | |
