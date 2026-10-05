#!/usr/bin/env python3
"""Export a detector to ONNX. CPU-OK (no GPU needed to export).

The TensorRT engine BUILD (scripts/build_trt.py) must run on the H100,
but export can happen on any machine. Do export early on CPU so GPU
Session B is pure engine-building, not debugging export code.

Usage: python scripts/export_onnx.py --detector yolo26s --out engines/yolo26s.onnx
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.detector import Detector


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    det = Detector(args.detector, device="cpu")
    out = det.to_onnx(args.out, opset=args.opset)
    print(f"[export] {args.detector} -> {out}")
