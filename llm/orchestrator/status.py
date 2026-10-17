#!/usr/bin/env python3
"""A short summary of a night run (about 20 lines): python3 status.py <run root>"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

root = Path(sys.argv[1])


def alive(pidfile):
    try:
        os.kill(int(Path(root / pidfile).read_text()), 0)
        return True
    except Exception:
        return False


print(f"supervisor {'running' if alive('supervisor.pid') else 'STOPPED'} | wake-lock {'on' if alive('caffeinate.pid') else 'OFF'}")
try:
    s = json.loads((root / "status.json").read_text())
    for t, a in (s.get("agents") or {}).items():
        print(f"  {t}: {a.get('phase')} (attempt {a.get('attempt')}, {round((time.time() - a.get('since', time.time())) / 60)} min)")
except Exception:
    pass
def counters(port):
    try:
        text = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=3).read().decode()
    except Exception:
        return None
    gen = run = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:generation_tokens_total{"):
            gen += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_running{"):
            run += float(line.rsplit(" ", 1)[1])
    return gen, run


ports = [8000, 8001]
a = {p: counters(p) for p in ports}
time.sleep(2)
b = {p: counters(p) for p in ports}
parts, total = [], 0.0
for i, p in enumerate(ports):
    if a[p] and b[p]:
        r = (b[p][0] - a[p][0]) / 2.0
        total += r
        parts.append(f"GPU{i} {r:.0f} tok/s ({b[p][1]:.0f} running)")
    else:
        parts.append(f"GPU{i} down")
print("model servers: " + " | ".join(parts) + f" | total {total:.0f} tok/s")
digest = root / "digest.md"
print(digest.read_text() if digest.exists() else "(no digest yet)")
