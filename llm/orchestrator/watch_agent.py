#!/usr/bin/env python3
"""Live terminal view of one coding agent: its reasoning, its commands and file edits, and their results.

  python3 watch_agent.py <agent folder> [--replay N] [--once]

Follows the newest events_*.jsonl in the folder (the supervisor writes one per attempt, review and fix round) and
switches to the next one automatically. Pure display: it only reads files.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import deque

D, I, B, R = "\x1b[2m", "\x1b[3m", "\x1b[1m", "\x1b[0m"
CY, GR, RD, YL, MG = "\x1b[36m", "\x1b[90m", "\x1b[31m", "\x1b[33m", "\x1b[35m"
out = sys.stdout


def w(s):
    out.write(s)
    out.flush()


def summarize_args(name, a):
    a = a if isinstance(a, dict) else {}
    if name == "bash":
        return "$ " + str(a.get("command", ""))[:600]
    if name == "write":
        return f"write {a.get('path')}  ({len(str(a.get('content', '')))} chars)"
    if name == "edit":
        return f"edit {a.get('path')}"
    if name in ("read", "ls", "find", "grep"):
        return f"{name} " + " ".join(f"{v}" for v in a.values())[:200]
    return f"{name} {json.dumps(a)[:240]}"


def result_text(res):
    try:
        return "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
    except Exception:
        return str(res)[:300]


class View:
    def __init__(self, name):
        self.mode = None            # thinking | text | tool
        self.name, self.win, self.turn_chunks, self.ratio = name, deque(), 0, 1.0
        self.total_tokens, self.t_title, self.t_mark = 0, 0.0, time.time()

    def chunk(self):
        """One streamed fragment arrived: keep a 3-second window for a live tokens-per-second figure."""
        now = time.time()
        self.win.append(now)
        self.turn_chunks += 1
        while self.win and now - self.win[0] > 3:
            self.win.popleft()
        rate = len(self.win) / 3.0 * self.ratio
        if now - self.t_title > 1:
            self.t_title = now
            w(f"\x1b]0;agent: {self.name} | {rate:.0f} tok/s | {self.total_tokens + int(self.turn_chunks * self.ratio)} tokens\x07")
        if now - self.t_mark > 8:
            self.t_mark = now
            w(f"\n{YL}  ~ {rate:.0f} tok/s{R}\n")

    def switch(self, mode):
        if self.mode != mode:
            if self.mode in ("thinking", "text"):
                w("\n")
            self.mode = mode

    def event(self, e):
        t = e.get("type")
        if t == "message_update":
            a = e.get("assistantMessageEvent", {})
            at = a.get("type")
            if at == "thinking_start":
                self.switch("thinking")
                w(f"\n{MG}{B}reasoning{R}\n")
            elif at == "thinking_delta":
                self.chunk()
                w(f"{D}{I}{a.get('delta', '')}{R}")
            elif at == "thinking_end":
                w("\n")
            elif at == "text_start":
                self.switch("text")
                w("\n")
            elif at == "text_delta":
                self.chunk()
                w(a.get("delta", ""))
            elif at == "text_end":
                w("\n")
        elif t == "tool_execution_start":
            self.switch("tool")
            w(f"\n{CY}{B}> {summarize_args(e.get('toolName'), e.get('args'))}{R}\n")
        elif t == "tool_execution_end":
            txt = result_text(e.get("result", {})).strip()
            lines = txt.splitlines()
            shown = lines[:10]
            color = RD if e.get("isError") else GR
            for l in shown:
                w(f"{color}  {l[:170]}{R}\n")
            if len(lines) > len(shown):
                w(f"{GR}  ... {len(lines) - len(shown)} more lines{R}\n")
        elif t == "message_start":
            self.turn_chunks = 0
        elif t == "message_end" and e.get("message", {}).get("role") == "assistant":
            u = e["message"].get("usage", {})
            if self.turn_chunks > 20 and u.get("output"):          # calibrate chunks to real tokens
                self.ratio = 0.5 * self.ratio + 0.5 * min(8.0, u["output"] / self.turn_chunks)
            self.total_tokens += u.get("output", 0)
            self.turn_chunks = 0
            sec = (e["message"].get("durationMs") or 0) / 1000
            rate = f", {u.get('output', 0) / sec:.0f} tok/s" if sec > 1 and u.get("output") else ""
            w(f"{D}[turn done: {u.get('output', 0)} output tokens ({u.get('reasoning', 0)} reasoning){rate}]{R}\n")
        elif t == "agent_end":
            w(f"\n{YL}{B}-- agent finished --{R}\n")


def latest(folder):
    files = glob.glob(os.path.join(folder, "events_*.jsonl"))
    return max(files, key=os.path.getmtime) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--replay", type=int, default=40, help="show the last N events of the current file first")
    ap.add_argument("--once", action="store_true", help="print what exists and exit")
    args = ap.parse_args()
    name = os.path.basename(os.path.normpath(args.folder))
    w(f"\x1b]0;agent: {name}\x07")
    w(f"{B}agent: {name}{R}  {D}(Ctrl+C to close this view; the agent keeps running){R}\n")
    view, cur, pos, buf = View(name), None, 0, b""
    while True:
        nf = latest(args.folder)
        if nf and nf != cur:
            cur, pos, buf = nf, 0, b""
            w(f"\n{YL}{B}== {os.path.basename(cur)} =={R}\n")
            if args.replay and not args.once:      # start near the end of an already long file
                size = os.path.getsize(cur)
                pos = max(0, size - args.replay * 400)
                if pos:
                    with open(cur, "rb") as f:
                        f.seek(pos)
                        f.readline()
                        pos = f.tell()
        if cur:
            with open(cur, "rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.startswith(b"{"):
                    try:
                        view.event(json.loads(line))
                    except Exception:
                        pass
        if args.once:
            return
        time.sleep(0.25)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
