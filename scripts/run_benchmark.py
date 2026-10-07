#!/usr/bin/env python3
"""Thin wrapper so every experiment is one command. See src/benchmark/run_study.py."""
import sys
sys.path.insert(0, "src")
from benchmark.run_study import run_study
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "smoke"])
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU regardless of mode (same as --mode smoke for device)")
    args = ap.parse_args()
    run_study(args.config, args.mode, force_cpu=args.cpu)
