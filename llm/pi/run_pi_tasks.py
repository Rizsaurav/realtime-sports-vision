#!/usr/bin/env python3
"""Run the Pi coding agent on new tasks against the Qwen server, then grade the result with hidden tests.

Each task gets its own empty working directory, its own virtual environment (so package installs never touch the
system Python) and a minimal environment. The agent works in that directory with Pi's default tools. When it
finishes, any process it left running is stopped and the hidden acceptance tests are run against its work.

  python llm/pi/run_pi_tasks.py --out /path/to/results --thinking high
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "pi-qwen.sh"
TASKS = {"url_shortener": ("url_shortener.md", "test_url_shortener.py"),
         "mdlite": ("mdlite.md", "test_mdlite.py"),
         "scheduler": ("scheduler.md", "test_scheduler.py")}

COMMON = """A client has asked you to build the software described below. Work in the current directory (it is empty).
A Python virtual environment with Flask, requests and pytest is already active and `pip install` works inside it.
Build the whole thing yourself, run it and test it with your own checks before you finish, and fix whatever you
find. An automated acceptance test suite that you cannot see will check the exact contract in the specification, so
follow every detail literally. Finish with a short summary of what you built.

--- SPECIFICATION ---
{spec}
--- END OF SPECIFICATION ---
"""


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def make_venv(path):
    sh([sys.executable, "-m", "venv", str(path)], check=True)
    r = sh([str(path / "bin" / "pip"), "install", "-q", "--disable-pip-version-check", "flask", "requests", "pytest"])
    if r.returncode:
        raise RuntimeError("venv setup failed: " + r.stderr[-400:])


def usage_stats(events_path):
    msgs, tools = [], {}
    for line in events_path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "message_end" and e.get("message", {}).get("role") == "assistant":
            msgs.append(e["message"])
        if e.get("type") == "tool_execution_end":
            tools[e.get("toolName")] = tools.get(e.get("toolName"), 0) + 1
    u = [m["usage"] for m in msgs if "usage" in m]
    out = sum(x.get("output", 0) for x in u)
    dur = sum(m.get("durationMs", 0) for m in msgs) / 1000
    per_msg = [x["output"] / (m["durationMs"] / 1000) for x, m in zip(u, msgs) if m.get("durationMs", 0) > 500 and x.get("output", 0) > 50]
    return {"model_turns": len(msgs), "tool_calls": tools, "output_tokens": out,
            "reasoning_tokens": sum(x.get("reasoning", 0) for x in u),
            "peak_context_tokens": max((x.get("input", 0) + x.get("output", 0) for x in u), default=0),
            "total_input_tokens": sum(x.get("input", 0) for x in u),
            "generation_seconds": round(dur, 1),
            "peak_turn_tok_s": round(max(per_msg), 1) if per_msg else None,
            "mean_turn_tok_s": round(sum(per_msg) / len(per_msg), 1) if per_msg else None}


def run_task(name, args, out_root):
    spec_file, test_file = TASKS[name]
    out = out_root / name
    work, venv = out / "work", out / "venv"
    work.mkdir(parents=True, exist_ok=True)
    (out / "hidden").mkdir(exist_ok=True)
    make_venv(venv)
    prompt = COMMON.format(spec=(HERE / "specs" / spec_file).read_text())
    env = {"PATH": f"{venv}/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin",
           "HOME": os.environ["HOME"], "PI_AGENT_HOME": str(out_root / "pihome"), "VIRTUAL_ENV": str(venv),
           "LANG": "en_US.UTF-8", "TERM": "dumb"}
    cmd = [str(LAUNCHER), "--no-session", "-nc", "-ne", "-ns", "--no-mcp", "--thinking", args.thinking,
           "--mode", "json", "-p", prompt]
    t0 = time.perf_counter()
    status = "ok"
    with open(out / "events.jsonl", "w") as ev, open(out / "stderr.txt", "w") as er:
        p = subprocess.Popen(cmd, cwd=work, env=env, stdout=ev, stderr=er, start_new_session=True)
        try:
            p.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            status = "timeout"
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        try:
            os.killpg(p.pid, signal.SIGTERM)             # stop anything the agent left running
        except ProcessLookupError:
            pass
    agent_s = time.perf_counter() - t0
    time.sleep(1)
    hidden = out / "hidden" / test_file
    hidden.write_text((HERE / "tests" / test_file).read_text())
    genv = dict(env, APP_DIR=str(work), TEST_PORT="8123")
    r = sh([str(venv / "bin" / "python"), "-m", "pytest", str(hidden), "-q", "-rA", "-p", "no:cacheprovider",
            "--tb=short", f"--rootdir={out / 'hidden'}"], cwd=work, env=genv, timeout=900)
    (out / "acceptance.txt").write_text(r.stdout[-20000:])
    passed = re.findall(r"^PASSED\s+\S+::(\S+)", r.stdout, flags=re.M)
    failed = re.findall(r"^FAILED\s+\S+::(\S+)", r.stdout, flags=re.M)
    errors = re.findall(r"^ERROR\s+\S+::(\S+)", r.stdout, flags=re.M)
    collection_error = ("ERROR collecting" in r.stdout) or (not (passed or failed or errors))
    result = {"task": name, "status": status, "collection_error": collection_error, "agent_seconds": round(agent_s, 1),
              "acceptance": {"passed": len(passed), "failed": len(failed), "errors": len(errors),
                             "total": len(passed) + len(failed) + len(errors), "failed_tests": failed + errors},
              "usage": usage_stats(out / "events.jsonl"),
              "files": sorted(str(x.relative_to(work)) for x in work.rglob("*")
                              if x.is_file() and "__pycache__" not in x.parts and not x.name.endswith((".db", ".pyc")))[:30]}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--thinking", default="high")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    out_root = Path(args.out).resolve()
    (out_root / "pihome" / ".pi" / "agent").mkdir(parents=True, exist_ok=True)
    (out_root / "pihome" / ".pi" / "agent" / "models.json").write_text((HERE / "models.json").read_text())
    results, threads = {}, []

    def go(t):
        try:
            results[t] = run_task(t, args, out_root)
        except Exception as e:
            results[t] = {"task": t, "error": repr(e)}

    for t in args.tasks:
        th = threading.Thread(target=go, args=(t,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    (out_root / "summary.json").write_text(json.dumps(results, indent=1))
    for t, r in results.items():
        a = r.get("acceptance", {})
        flag = "  [NO TESTS RAN: nothing importable was built]" if r.get("collection_error") else ""
        print(f"{t:14s} {r.get('status', 'error'):8s} tests {a.get('passed')}/{a.get('total')}{flag}  "
              f"agent {r.get('agent_seconds')}s  usage {json.dumps(r.get('usage'))[:240]}")


if __name__ == "__main__":
    main()
