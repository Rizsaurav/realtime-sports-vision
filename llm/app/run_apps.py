#!/usr/bin/env python3
"""Run the app-building tasks in parallel against one server, with Docker kept under a small budget.

Only the python:3.12-slim image is pulled; it is removed at the end, containers are removed by exact name, and the
Docker state is compared with the state at the start.

  python llm/app/run_apps.py --port 8000 --out /data/saurav/results/app/run1
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = "python:3.12-slim"
TASKS = ["notes_api", "expense_cli", "kanban_board"]


def sh(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def df():
    out = {}
    for line in sh(["docker", "system", "df", "--format", "{{json .}}"]).stdout.splitlines():
        r = json.loads(line)
        out[r["Type"]] = r["Size"]
    return out


def containers():
    return set(sh(["docker", "ps", "-a", "--format", "{{.Names}}"]).stdout.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per task, agent plus grading")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--step-limit", type=int, default=150)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    before, c_before = df(), containers()
    had_image = sh(["docker", "image", "inspect", IMAGE]).returncode == 0
    log = open(out / "run.log", "a")

    def note(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    note(f"docker before: {before}; image already present: {had_image}")
    if not had_image:
        r = sh(["docker", "pull", "-q", IMAGE], timeout=900)
        note(f"pulled {IMAGE}: rc={r.returncode} docker after pull: {df()}")

    def one(task):
        cmd = [sys.executable, str(HERE / "run_app_task.py"), "--task", task, "--port", str(args.port),
               "--out", str(out), "--step-limit", str(args.step_limit)] + (["--no-thinking"] if args.no_thinking else [])
        t0 = time.perf_counter()
        try:
            with open(out / f"{task}.stdout.log", "w") as lf:
                p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=args.timeout)
            status = f"exit {p.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
        note(f"{task}: {status} after {time.perf_counter() - t0:.0f}s")
        return task, status

    with ThreadPoolExecutor(len(args.tasks)) as ex:
        statuses = dict(ex.map(one, args.tasks))
    mine = sorted(containers() - c_before)
    for name in mine:
        sh(["docker", "rm", "-f", name])
    if not had_image:
        sh(["docker", "rmi", "-f", IMAGE])
    after = df()
    results = {}
    for t in args.tasks:
        f = out / t / "result.json"
        results[t] = json.loads(f.read_text()) if f.exists() else {"status": statuses[t]}
    summary = {"statuses": statuses, "removed_containers": mine, "docker_before": before, "docker_after": after,
               "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    note(f"docker after cleanup: {after}")


if __name__ == "__main__":
    main()
