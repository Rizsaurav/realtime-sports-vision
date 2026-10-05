#!/usr/bin/env python3
"""Build TensorRT engines. *** GPU ONLY — run on the H100 (Session B). ***

TensorRT engines are architecture-specific: an engine built on the H100
runs ONLY on H100s. Build once, reuse for every later benchmark and demo.

Usage: python scripts/build_trt.py --config configs/tensorrt.yaml
Reads engines/backbones x precisions from the config and writes .engine files.
FP8/INT8 need calibration data (see config: calib_dataset, 500 images).
"""
import argparse
# TODO: tensorrt imports (only available on the GPU machine)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tensorrt.yaml")
    args = ap.parse_args()
    print(f"[trt] building engines from {args.config} — H100 required")
    raise NotImplementedError("implement with tensorrt.Builder on the GPU machine")
