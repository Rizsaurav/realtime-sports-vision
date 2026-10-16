#!/usr/bin/env python3
"""Aggregate and per-stream decode throughput at several concurrency levels (standard library only).

For each level N, N coding prompts are sent at the same time, each streamed; the aggregate rate is
total completion tokens divided by the time from the first request to the last token, and the
per-stream rate is the mean of each stream's own decode rate. With vLLM it also reports the MTP
draft acceptance rate for that level from /metrics.

  python llm/bench_concurrency.py --port 8000 --levels 1 2 4 8 --max-tokens 1024
"""
import argparse
import json
import re
import threading
import time
import urllib.request

from bench_decode import PROMPTS


def counters(base):
    try:
        text = urllib.request.urlopen(base + "/metrics", timeout=5).read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        m = re.match(r"(vllm:spec_decode_num_(?:accepted|draft)_tokens_total)\{[^}]*\}\s+([0-9.e+]+)", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def one(base, model, prompt, max_tokens, temperature, out, i):
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "top_k": 20, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    first, usage = None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                if first is None and ch.get("delta", {}).get("content"):
                    first = time.perf_counter()
    end = time.perf_counter()
    n = usage["completion_tokens"] if usage else 0
    out[i] = {"tokens": n, "per_stream": (n - 1) / (end - first) if first and n > 1 else 0.0, "end": end}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    one(base, args.model, "Say hello.", 32, args.temperature, {}, 0)
    print("level  aggregate tok/s  per-stream tok/s  acceptance")
    for n in args.levels:
        aggs, pers, accs = [], [], []
        for rep in range(args.repeats):
            c0 = counters(base)
            out, th = {}, []
            t0 = time.perf_counter()
            for i in range(n):
                p = PROMPTS[(i + rep) % len(PROMPTS)]
                th.append(threading.Thread(target=one, args=(base, args.model, p, args.max_tokens, args.temperature, out, i)))
            for t in th:
                t.start()
            for t in th:
                t.join()
            wall = max(o["end"] for o in out.values()) - t0
            aggs.append(sum(o["tokens"] for o in out.values()) / wall)
            pers.append(sum(o["per_stream"] for o in out.values()) / n)
            c1 = counters(base)
            d = c1.get("vllm:spec_decode_num_draft_tokens_total", 0) - c0.get("vllm:spec_decode_num_draft_tokens_total", 0)
            a = c1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - c0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
            if d:
                accs.append(100 * a / d)
        print(f"{n:5d}  {sum(aggs) / len(aggs):15.1f}  {sum(pers) / len(pers):16.1f}  "
              f"{(f'{sum(accs) / len(accs):.0f}%' if accs else '-'):>10}", flush=True)


if __name__ == "__main__":
    main()
