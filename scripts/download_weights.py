#!/usr/bin/env python3
"""Download TinyViT-5M ImageNet-1k weights (safetensors, ~21MB) into weights/.

Uses urllib (proxy-env tolerant) so it works on machines where
huggingface_hub's httpx session can't parse the proxy config.
The file is gitignored; rerun on any fresh checkout/GPU box.

Usage: python scripts/download_weights.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.efficient_vit import WEIGHTS_URL, DEFAULT_WEIGHTS_PATH  # noqa: E402


def main():
    dest = DEFAULT_WEIGHTS_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[weights] already present: {dest} "
              f"({dest.stat().st_size / 1e6:.1f} MB)")
        return 0

    def hook(n, block, total):
        done = n * block / 1e6
        pct = f"{100 * n * block / total:.0f}%" if total > 0 else ""
        print(f"\r[weights] {done:.1f} MB {pct}", end="", flush=True)

    print(f"[weights] downloading TinyViT-5M in1k weights (~21MB)...")
    urllib.request.urlretrieve(WEIGHTS_URL, dest, hook)
    print(f"\n[weights] saved: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
