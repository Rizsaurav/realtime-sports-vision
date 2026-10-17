#!/usr/bin/env python3
"""Sample a vLLM server's /metrics once per second, and report peak token throughput.

  python llm/pi/sampler.py run --port 8000 --out samples.csv        # runs until SIGTERM/Ctrl-C
  python llm/pi/sampler.py analyze samples.csv                      # prints the peaks as JSON

Columns: unix time, generation tokens total, prompt tokens total, requests running, requests waiting,
speculative tokens accepted, speculative tokens drafted. Peaks are computed from the counter deltas over the
actual elapsed time between samples (1 s and 5 s windows).
"""
import argparse
import csv
import json
import re
import signal
import sys
import time
import urllib.request

KEYS = ["vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:num_requests_running",
        "vllm:num_requests_waiting", "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total"]


def scrape(port):
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    out = dict.fromkeys(KEYS, 0.0)
    for line in text.splitlines():
        for k in KEYS:
            if line.startswith(k + "{") or line.startswith(k + " "):
                out[k] += float(line.rsplit(" ", 1)[1])
    return out


def run(port, path, interval):
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + KEYS)
        while not stop:
            try:
                m = scrape(port)
                w.writerow([f"{time.time():.3f}"] + [m[k] for k in KEYS])
                f.flush()
            except Exception:
                pass
            time.sleep(interval)


def analyze(path, since=None, until=None):
    rows = [r for r in csv.DictReader(open(path))
            if (since is None or float(r["t"]) >= since) and (until is None or float(r["t"]) <= until)]
    if len(rows) < 3:
        return {"samples": len(rows), "error": "too few samples in the window"}
    t = [float(r["t"]) for r in rows]
    g = [float(r[KEYS[0]]) for r in rows]
    p = [float(r[KEYS[1]]) for r in rows]
    run_ = [float(r[KEYS[2]]) for r in rows]

    def peak(series, window):
        """Highest rate over any span of at least `window` seconds (two-pointer over the samples)."""
        best, at, j = 0.0, None, 0
        for i in range(len(t)):
            while j + 1 < i and t[i] - t[j + 1] >= window - 0.05:
                j += 1
            if i > j and t[i] - t[j] >= window - 0.05:
                rate = (series[i] - series[j]) / (t[i] - t[j])
                if rate > best:
                    best, at = rate, t[i]
        return round(best, 1), at

    active = [i for i in range(1, len(t)) if g[i] > g[i - 1]]
    gen_total, prompt_total = g[-1] - g[0], p[-1] - p[0]
    span = sum(t[i] - t[i - 1] for i in active) or 1.0
    acc, drf = float(rows[-1][KEYS[4]]) - float(rows[0][KEYS[4]]), float(rows[-1][KEYS[5]]) - float(rows[0][KEYS[5]])
    return {"samples": len(rows), "peak_decode_tok_s_1s": peak(g, 1)[0], "peak_decode_tok_s_5s": peak(g, 5)[0],
            "mean_decode_tok_s_while_generating": round(gen_total / span, 1),
            "peak_concurrent_requests": max(run_) if run_ else 0, "generated_tokens": int(gen_total),
            "prompt_tokens_processed": int(prompt_total),
            "spec_acceptance_pct": round(100 * acc / drf, 1) if drf else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--out", required=True)
    r.add_argument("--interval", type=float, default=1.0)
    a = sub.add_parser("analyze")
    a.add_argument("csv")
    a.add_argument("--since", type=float, default=None, help="unix time")
    a.add_argument("--until", type=float, default=None, help="unix time")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.port, args.out, args.interval)
    else:
        print(json.dumps(analyze(args.csv, args.since, args.until), indent=1))
