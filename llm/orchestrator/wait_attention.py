#!/usr/bin/env python3
"""Block until the digest has a NEW attention line, the night ends or the supervisor dies; then print and exit.
   python3 wait_attention.py <run root> [max_minutes]"""
import os
import sys
import time
from pathlib import Path

root, limit = Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 600
seen, t0 = None, time.time()


def lines():
    d = root / "digest.md"
    return {l for l in d.read_text().splitlines() if l.startswith(("ATTENTION", "NIGHT END"))} if d.exists() else set()


def supervisor_alive():
    try:
        os.kill(int(Path(root / "supervisor.pid").read_text()), 0)
        return True
    except Exception:
        return False


while time.time() - t0 < limit * 60:
    cur = lines()
    if seen is None:
        seen = cur                          # lines already present when we start are not news
    new = cur - seen
    if new or any(l.startswith("NIGHT END") for l in cur):
        print("\n".join(sorted(new or cur)))
        break
    if not supervisor_alive() and (root / "supervisor.pid").exists():
        print("ATTENTION: the supervisor process is not running")
        break
    time.sleep(60)
else:
    print("no new events within the time limit")
