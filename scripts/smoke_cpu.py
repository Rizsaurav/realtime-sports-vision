#!/usr/bin/env python3
"""CPU smoke test: validates the ENTIRE pipeline for $0, no dataset needed.

  1. Downloads a small real photo (ultralytics docs image, ~100KB)
  2. Synthesizes a 30-frame "moving camera" clip from it with cv2
  3. Runs Detector (yolo26n, CPU) + Tracker through StreamingPipeline
  4. Writes annotated clip + results JSON, prints the latency report

If this passes, the plumbing is proven. GPU sessions then only change
the device and the config — zero code changes.

Usage: python scripts/smoke_cpu.py
"""
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.detector import Detector
from tracking.tracker import Tracker
from streaming.pipeline import StreamingPipeline

IMG_URL = "https://ultralytics.com/images/bus.jpg"
WORK = Path("results/smoke_cpu")


def make_clip(n_frames=30, w=640, h=480):
    WORK.mkdir(parents=True, exist_ok=True)
    img_path = WORK / "bus.jpg"
    if not img_path.exists():
        print(f"[smoke] downloading sample image (~100KB)...")
        urllib.request.urlretrieve(IMG_URL, img_path)
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (w * 2, h * 2))  # room to pan
    clip_path = WORK / "clip.mp4"
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             15.0, (w, h))
    for i in range(n_frames):
        x = int((w * i) / n_frames)  # slow pan = fake motion
        writer.write(img[0:h, x:x + w])
    writer.release()
    return clip_path


def main():
    clip = make_clip()
    print("[smoke] detector: yolo26n on CPU (weights auto-download ~5MB once)")
    detector = Detector("yolo26n", imgsz=320, conf=0.25, device="cpu")
    tracker = Tracker("bytetrack")  # falls back to built-in IoU if boxmot missing
    out = WORK / "clip_tracked.mp4"
    pipe = StreamingPipeline(detector, tracker, out_path=out)
    result = pipe.run(clip)
    pipe.save(result, WORK / "smoke_result.json")

    rep = pipe.latency_report()
    print(f"[smoke] frames={rep['frames']} fps={rep['fps']:.1f} "
          f"p50={rep['latency_p50_ms']:.1f}ms p95={rep['latency_p95_ms']:.1f}ms")
    print(f"[smoke] annotated clip: {out} ({out.stat().st_size // 1024} KB)")
    assert out.exists() and rep["frames"] == 30, "smoke test FAILED"
    print("[smoke] PASS — pipeline plumbing proven on CPU. "
          "GPU sessions change only device + config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
