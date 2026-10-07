#!/usr/bin/env python3
"""Fine-tune a YOLO26 detector on SportsMOT (single class: player).

Run on the GPU in tmux. Smoke-test the plumbing on CPU first with
--epochs 1 --fraction 0.01 --device cpu.

Usage:
  python scripts/finetune_detector.py --model yolo26s --epochs 30 \
      --data data/processed/sportsmot_yolo/data.yaml --device 0
Writes runs/finetune/<name>/weights/best.pt
"""
import argparse

from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s")
    ap.add_argument("--data", default="data/processed/sportsmot_yolo/data.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    model = YOLO(f"{args.model}.pt")
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, fraction=args.fraction,
                workers=args.workers, project=str(Path("runs/finetune").resolve()),
                name=args.name or f"{args.model}_sportsmot", exist_ok=True,
                patience=10, plots=False, seed=0, deterministic=False)


if __name__ == "__main__":
    main()
