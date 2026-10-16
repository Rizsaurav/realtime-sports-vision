#!/usr/bin/env python3
"""Decode-speed benchmark for an OpenAI-compatible server (standard library only).

Runs a fixed set of coding prompts one at a time, streams each reply, and reports
time to first token and decode tokens per second (completion tokens after the first
token, over the time after the first token). With vLLM it also reports the MTP
acceptance rate from /metrics.

  python llm/bench_decode.py --port 8000 --max-tokens 1024
  python llm/bench_decode.py --port 8000 --thinking        # let the model reason first
"""
import argparse
import json
import re
import statistics
import time
import urllib.request

PROMPTS = [
    "Write a Python class LRUCache with get and put in O(1) using OrderedDict, with type hints and a docstring.",
    "Implement an asyncio rate limiter in Python that allows N calls per second across many coroutines, with a test.",
    "Write a SQL query and a short explanation: for each customer, the running total of their orders ordered by date, and flag the first order above 100 dollars.",
    "Refactor this into clean, idiomatic Python and explain each change:\n\ndef f(l):\n  r=[]\n  for i in range(len(l)):\n    if l[i]%2==0:\n      r.append(l[i]*l[i])\n  return r",
    "Write a TypeScript function debounce<T extends (...a: any[]) => void>(fn: T, ms: number) with correct typings and a Jest test.",
    "Find and fix the bug, then write a regression test:\n\ndef binary_search(a, x):\n  lo, hi = 0, len(a)\n  while lo < hi:\n    mid = (lo + hi) // 2\n    if a[mid] < x: lo = mid\n    else: hi = mid\n  return lo if lo < len(a) and a[lo] == x else -1",
]


def metrics(base):
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


def run(base, model, prompt, max_tokens, thinking, temperature):
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "top_k": 20, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, usage = None, None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    first = time.perf_counter()
    end = time.perf_counter()
    n = usage["completion_tokens"] if usage else 0
    decode = (n - 1) / (end - first) if first and n > 1 and end > first else 0.0
    return {"ttft_ms": (first - t0) * 1000 if first else 0.0, "tokens": n, "tok_s": decode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    for _ in range(args.warmup):
        run(base, args.model, "Say hello.", 32, False, args.temperature)
    m0 = metrics(base)
    rows = []
    for _ in range(args.rounds):
        for p in PROMPTS:
            r = run(base, args.model, p, args.max_tokens, args.thinking, args.temperature)
            rows.append(r)
            print(f"  {r['tokens']:5d} tokens  ttft {r['ttft_ms']:6.0f} ms  {r['tok_s']:6.1f} tok/s   {p[:50]!r}", flush=True)
    speeds = [r["tok_s"] for r in rows if r["tok_s"] > 0]
    m1 = metrics(base)
    print(f"decode tok/s: mean {statistics.mean(speeds):.1f}  median {statistics.median(speeds):.1f}  "
          f"min {min(speeds):.1f}  max {max(speeds):.1f}   ttft median {statistics.median(r['ttft_ms'] for r in rows):.0f} ms")
    acc = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    drf = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get("vllm:spec_decode_num_draft_tokens_total", 0)
    if drf:
        print(f"MTP acceptance: {acc:.0f}/{drf:.0f} draft tokens = {100 * acc / drf:.1f}%")


if __name__ == "__main__":
    main()
