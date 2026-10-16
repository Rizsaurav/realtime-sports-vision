#!/usr/bin/env python3
"""Supervisor for unattended coding agents.

Runs several Pi agents (each one a Qwen model on the GPU box, reached through the SSH forwards on localhost) on a
queue of tasks. Everything runs on this machine; the box only serves the model.

For each task the supervisor
  1. clones the integration branch into a private folder and creates the branch agent/<task>,
  2. runs one Pi agent inside a macOS sandbox (it can write only to its own folder and reach only the model servers
     and package registries),
  3. restores the protected test and spec files, commits what the agent left uncommitted, runs the acceptance
     commands in the sandbox and scans the work for stray mentions of tools or assistants,
  4. on failure starts a new attempt with the failing output, and on success merges the branch into `integration`
     and re-runs every merged task's acceptance commands.
Nothing is ever pushed. Progress is written to run.jsonl, status.json and finally report.md.

  python3 llm/orchestrator/orchestrate.py plan.json
  touch <root>/STOP        # stop after the current attempts
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pi"))
from run_pi_tasks import usage_stats  # noqa: E402

PROFILE = HERE / "agent.sb"
PI_DIR = Path.home() / ".pi-install"
PI_BIN = PI_DIR / "node_modules" / ".bin" / "pi"
MODELS_JSON = HERE.parent / "pi" / "models.json"
TRACE_RE = re.compile(r"claude|anthropic|openai|chatgpt|\bgpt-?\d|\bqwen|\bllm\b|language model|\bAI\b|assistant|"
                      r"generated (by|with)|co-authored-by|copilot", re.I)
LOCK = threading.Lock()
MERGE_LOCK = threading.Lock()


def sh(cmd, cwd=None, env=None, timeout=None, check=False):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))[:120]}: {(r.stderr or r.stdout)[-400:]}")
    return r


class Night:
    def __init__(self, plan):
        self.plan = plan
        self.root = Path(plan["root"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_repo, self.base_branch = plan["base_repo"], plan["base_branch"]
        self.ports = plan.get("servers", [8000, 8001])
        self.deadline = time.time() + 3600 * plan.get("deadline_hours", 10)
        self.node_dir = sorted(Path.home().glob(".nvm/versions/node/v22.*"))[-1]
        self.log = open(self.root / "run.jsonl", "a")
        self.status = {"started": time.time(), "agents": {}, "servers": {}, "tasks": {}}
        self.tasks = {t["id"]: t for t in plan["tasks"]}
        self.state = {t: "pending" for t in self.tasks}          # pending | running | done | failed | skipped
        self.results = {}
        self.integration = self.root / "integration"
        name = sh(["git", "-C", self.base_repo, "config", "user.name"]).stdout.strip()
        email = sh(["git", "-C", self.base_repo, "config", "user.email"]).stdout.strip()
        self.ident = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_NAME": name,
                      "GIT_COMMITTER_EMAIL": email}

    # -- logging ---------------------------------------------------------------------------------------------
    def event(self, kind, **kw):
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": kind, **kw}
        with LOCK:
            self.log.write(json.dumps(row) + "\n")
            self.log.flush()
        print(f"{row['t']} {kind} " + " ".join(f"{k}={str(v)[:80]}" for k, v in kw.items()), flush=True)

    def write_status(self):
        self.status["updated"] = time.time()
        self.status["states"] = dict(self.state)
        with LOCK:
            (self.root / "status.json").write_text(json.dumps(self.status, indent=1, default=str))

    # -- model servers ---------------------------------------------------------------------------------------
    def server_ok(self, port):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=4).read()
            return True
        except Exception:
            return False

    def reforward(self, port):
        host = self.plan.get("ssh_host", "scc")
        r = sh(["ssh", "-O", "forward", "-L", f"{port}:127.0.0.1:{port}", host], timeout=30)
        if r.returncode:                                     # no shared connection: open our own tunnel
            sh(["ssh", "-f", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes",
                "-L", f"{port}:127.0.0.1:{port}", host], timeout=60)

    def wait_for_server(self, port, max_wait=1800):
        t0 = time.time()
        while not self.server_ok(port):
            if time.time() - t0 > max_wait or (self.root / "STOP").exists():
                return False
            self.event("server_down", port=port)
            self.reforward(port)
            time.sleep(30)
        return True

    # -- folders and git ---------------------------------------------------------------------------------------
    def prepare_integration(self):
        if self.integration.exists():
            return
        sh(["git", "clone", "-q", "--no-hardlinks", self.base_repo, str(self.integration)], check=True)
        sh(["git", "checkout", "-q", "-b", "integration", f"origin/{self.base_branch}"], cwd=self.integration, check=True)
        for k, v in (("user.name", self.ident["GIT_AUTHOR_NAME"]), ("user.email", self.ident["GIT_AUTHOR_EMAIL"])):
            sh(["git", "config", k, v], cwd=self.integration, check=True)

    def make_venv(self, home, packages):
        venv = home / "venv"
        if not venv.exists():
            sh([sys.executable, "-m", "venv", str(venv)], check=True)
        if packages:
            r = sh([str(venv / "bin" / "pip"), "install", "-q", "--disable-pip-version-check", *packages], timeout=900)
            if r.returncode:
                raise RuntimeError("pip install failed: " + r.stderr[-300:])
        return venv

    def sandbox_cmd(self, d, cmd):
        return ["sandbox-exec", "-f", str(PROFILE), "-D", f"WORK={d['work']}", "-D", f"AGENT_HOME={d['home']}",
                "-D", f"AGENT_TMP={d['tmp']}", "-D", f"NODE_DIR={self.node_dir}", "-D", f"PI_DIR={PI_DIR}",
                *cmd]

    def agent_env(self, d):
        return {"HOME": str(d["home"]), "TMPDIR": str(d["tmp"]), "LANG": "en_US.UTF-8", "TERM": "dumb",
                "VIRTUAL_ENV": str(d["home"] / "venv"), "PYTHONDONTWRITEBYTECODE": "1",
                "PATH": f"{d['home'] / 'venv' / 'bin'}:{self.node_dir / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin",
                **self.ident}

    def run_sandboxed(self, d, cmd, timeout, out_file=None):
        env = self.agent_env(d)
        full = self.sandbox_cmd(d, ["env", *[f"{k}={v}" for k, v in env.items()], *cmd])
        stdout = open(out_file, "w") if out_file else subprocess.PIPE
        p = subprocess.Popen(full, cwd=d["work"], stdout=stdout, stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout)
            status = "ok"
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            status = "timeout"
        try:
            os.killpg(p.pid, signal.SIGTERM)                  # anything the agent left running
        except ProcessLookupError:
            pass
        if out_file:
            stdout.close()
        return p.returncode, status, out or ""

    # -- one task --------------------------------------------------------------------------------------------
    def prompt_for(self, task, attempt, failure):
        checks = "\n".join(f"  {c}" for c in task["acceptance"])
        p = (f"You are one of several engineers working in parallel on a soccer-tracking web app. You work alone in the "
             f"current directory, which is a git repository on branch agent/{task['id']}.\n\n"
             f"Your task is described in `{task['spec']}`. Read it first, together with docs/WEBAPP_CONTRACT.md.\n\n"
             f"Rules:\n"
             f"- Make these acceptance checks pass, and run them yourself often:\n{checks}\n"
             f"- You may not change anything under tests/, web_tests/, tasks/ or docs/WEBAPP_CONTRACT.md; changes there "
             f"are discarded.\n"
             f"- Commit your work with git in small steps. Commit messages are short, imperative and describe the "
             f"change (for example \"Add heatmap binning\"). Never push.\n"
             f"- Do not write anything in code, comments, documentation or commit messages about who or what wrote "
             f"it (no mention of assistants, models, AI or tools).\n"
             f"- Work only inside this directory. Python packages install with pip (a virtual environment is active) "
             f"and JavaScript packages with npm.\n"
             f"- When every check passes, finish with a short summary of what you built.\n")
        if attempt > 1:
            p += (f"\nThis is attempt {attempt}. The previous attempt did not pass. The existing work is in the "
                  f"directory; fix it rather than starting over. The failing output was:\n\n{failure[-6000:]}\n")
        return p

    def scan_traces(self, work, base_commit):
        diff = sh(["git", "diff", base_commit, "HEAD", "--unified=0"], cwd=work).stdout
        hits = []
        file = None
        for line in diff.splitlines():
            if line.startswith("+++ "):
                file = line[4:]
            elif line.startswith("+") and not line.startswith("+++") and TRACE_RE.search(line):
                hits.append(f"{file}: {line[1:].strip()[:100]}")
        log = sh(["git", "log", f"{base_commit}..HEAD", "--format=%an <%ae>%n%B"], cwd=work).stdout
        for line in log.splitlines():
            if TRACE_RE.search(line):
                hits.append(f"commit message: {line.strip()[:100]}")
        return hits[:12]

    def run_task(self, task, slot):
        tid = task["id"]
        port = self.ports[slot % len(self.ports)]
        d = {k: self.root / "agents" / tid / k for k in ("work", "home", "tmp")}
        shutil.rmtree(self.root / "agents" / tid, ignore_errors=True)
        for p in d.values():
            p.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(d["work"])
        with MERGE_LOCK:
            sh(["git", "clone", "-q", "--no-hardlinks", str(self.integration), str(d["work"])], check=True)
        work = d["work"]
        sh(["git", "checkout", "-q", "-b", f"agent/{tid}"], cwd=work, check=True)
        for k, v in (("user.name", self.ident["GIT_AUTHOR_NAME"]), ("user.email", self.ident["GIT_AUTHOR_EMAIL"])):
            sh(["git", "config", k, v], cwd=work, check=True)
        base_commit = sh(["git", "rev-parse", "HEAD"], cwd=work).stdout.strip()
        self.make_venv(d["home"], self.plan.get("python_packages", []) + task.get("python_packages", []))
        if (work / "package.json").exists():
            sh([str(self.node_dir / "bin" / "npm"), "install", "--ignore-scripts", "--no-fund", "--no-audit", "--silent"],
               cwd=work, env={**self.agent_env(d)}, timeout=900)
        agent_dir = d["home"] / ".pi" / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(MODELS_JSON.read_text())
        prov = cfg["providers"]["qwen-box"]
        prov["baseUrl"] = f"http://localhost:{port}/v1"
        m = prov["models"][0]
        m["contextWindow"], m["maxTokens"] = self.plan.get("context_window", 262144), self.plan.get("max_tokens", 98304)
        (agent_dir / "models.json").write_text(json.dumps(cfg, indent=1))

        failure, usage_total, t_task = "", {"output_tokens": 0, "model_turns": 0}, time.time()
        levels = task.get("thinking", self.plan.get("thinking", ["medium", "high", "high", "high"]))
        for attempt in range(1, task.get("max_attempts", 4) + 1):
            if (self.root / "STOP").exists() or time.time() > self.deadline:
                self.event("stopped_before_attempt", task=tid, attempt=attempt)
                return {"status": "stopped", "attempts": attempt - 1}
            if not self.wait_for_server(port):
                return {"status": "no_server", "attempts": attempt - 1}
            thinking = levels[min(attempt - 1, len(levels) - 1)]
            self.status["agents"][tid] = {"attempt": attempt, "phase": "agent", "since": time.time(), "port": port,
                                           "thinking": thinking}
            self.write_status()
            self.event("attempt_start", task=tid, attempt=attempt, thinking=thinking, port=port)
            ev_file = self.root / "agents" / tid / f"events_{attempt}.jsonl"
            cmd = [str(PI_BIN), "--provider", "qwen-box", "--model", "qwen3.8-27b", "--no-session", "-nc", "-ne", "-ns",
                   "--no-mcp", "--thinking", thinking, "--mode", "json", "-p", self.prompt_for(task, attempt, failure)]
            t0 = time.time()
            rc, pstatus, _ = self.run_sandboxed(d, cmd, 60 * task.get("timeout_min", 90), out_file=ev_file)
            usage = usage_stats(ev_file) if ev_file.exists() else {}
            for k in ("output_tokens", "model_turns"):
                usage_total[k] += usage.get(k, 0)
            # protect the specs and tests, keep whatever else the agent did
            for path in self.plan.get("protected", []):
                sh(["git", "checkout", base_commit, "--", path], cwd=work)
            sh(["git", "add", "-A"], cwd=work)
            if sh(["git", "status", "--porcelain"], cwd=work).stdout.strip():
                sh(["git", "commit", "-q", "-m", f"Implement {task.get('title', tid)}"], cwd=work)
            self.status["agents"][tid]["phase"] = "acceptance"
            self.write_status()
            ok, out = True, ""
            for c in task["acceptance"]:
                rc2, st, o = self.run_sandboxed(d, ["/bin/sh", "-c", c], timeout=900)
                out += f"$ {c}\n{o[-4000:]}\n"
                if rc2 != 0 or st == "timeout":
                    ok = False
                    break
            traces = self.scan_traces(work, base_commit)
            if ok and traces:
                ok = False
                out += "\nThe acceptance checks pass, but the work mentions tools or assistants. Remove these:\n" + "\n".join(traces)
            self.event("attempt_end", task=tid, attempt=attempt, passed=ok, pi_status=pstatus, seconds=round(time.time() - t0),
                       output_tokens=usage.get("output_tokens"), turns=usage.get("model_turns"),
                       tail=out[-300:].replace("\n", " | "))
            if ok:
                return {"status": "passed", "attempts": attempt, "usage": usage_total, "work": str(work),
                        "seconds": round(time.time() - t_task)}
            failure = out
        return {"status": "failed", "attempts": task.get("max_attempts", 4), "usage": usage_total, "work": str(work),
                "seconds": round(time.time() - t_task), "last_failure": failure[-1500:]}

    def integrate(self, task, result):
        """Merge a passed task into the integration branch and re-run every merged task's checks."""
        tid = task["id"]
        with MERGE_LOCK:
            work = result["work"]
            sh(["git", "fetch", "-q", work, f"agent/{tid}:refs/heads/agent/{tid}"], cwd=self.integration)
            before = sh(["git", "rev-parse", "HEAD"], cwd=self.integration).stdout.strip()
            m = sh(["git", "merge", "--no-ff", "-q", "-m", f"Merge {task.get('title', tid)}", f"agent/{tid}"],
                   cwd=self.integration)
            if m.returncode:
                sh(["git", "merge", "--abort"], cwd=self.integration)
                return {"merged": False, "reason": "merge conflict: " + (m.stdout + m.stderr)[-300:]}
            d = {k: self.root / "agents" / "_integration" / k for k in ("home", "tmp")}
            d["work"] = self.integration
            for p in (d["home"], d["tmp"]):
                p.mkdir(parents=True, exist_ok=True)
            self.make_venv(d["home"], self.plan.get("python_packages", []))
            if (self.integration / "package.json").exists() and not (self.integration / "node_modules").exists():
                sh([str(self.node_dir / "bin" / "npm"), "install", "--ignore-scripts", "--no-fund", "--no-audit", "--silent"],
                   cwd=self.integration, env=self.agent_env(d), timeout=900)
            for done in [t for t, s in self.state.items() if s == "done"] + [tid]:
                for c in self.tasks[done]["acceptance"]:
                    rc, st, o = self.run_sandboxed(d, ["/bin/sh", "-c", c], timeout=900)
                    if rc != 0:
                        sh(["git", "reset", "-q", "--hard", before], cwd=self.integration)
                        return {"merged": False, "reason": f"integration check failed for {done}: {c}\n{o[-600:]}"}
            return {"merged": True}

    # -- scheduling --------------------------------------------------------------------------------------------
    def worker(self, slot):
        while True:
            if (self.root / "STOP").exists() or time.time() > self.deadline:
                return
            with LOCK:
                ready = [t for t, s in self.state.items() if s == "pending"
                         and all(self.state[x] == "done" for x in self.tasks[t].get("depends_on", []))]
                blocked = [t for t, s in self.state.items() if s == "pending"
                           and any(self.state[x] in ("failed", "skipped") for x in self.tasks[t].get("depends_on", []))]
                for t in blocked:
                    self.state[t] = "skipped"
                if not ready:
                    running = any(s == "running" for s in self.state.values())
                    waiting = any(s == "pending" for s in self.state.values())
                    if not running and not waiting:
                        return
                    task = None
                else:
                    task = self.tasks[ready[0]]
                    self.state[task["id"]] = "running"
            if task is None:
                time.sleep(10)
                continue
            self.event("task_start", task=task["id"], slot=slot)
            try:
                res = self.run_task(task, slot)
                if res["status"] == "passed":
                    mres = self.integrate(task, res)
                    res["merge"] = mres
                    if not mres["merged"]:
                        res["status"] = "failed"
                        res["last_failure"] = mres["reason"]
            except Exception as e:                                 # a crash in one task must not stop the night
                res = {"status": "failed", "error": repr(e)}
            with LOCK:
                self.state[task["id"]] = "done" if res["status"] == "passed" else "failed"
                self.results[task["id"]] = res
            self.event("task_end", task=task["id"], status=res["status"], attempts=res.get("attempts"))
            self.status["agents"].pop(task["id"], None)
            self.write_status()

    def heartbeat(self, stop):
        while not stop.is_set():
            self.status["servers"] = {str(p): self.server_ok(p) for p in self.ports}
            self.write_status()
            stop.wait(30)

    def report(self):
        lines = [f"# Night run `{self.plan.get('name', '')}`", "",
                 "| Task | Status | Attempts | Output tokens | Minutes | Merged |", "|---|---|---|---|---|---|"]
        for tid in self.tasks:
            r = self.results.get(tid, {})
            lines.append(f"| {tid} | {r.get('status', self.state[tid])} | {r.get('attempts', '-')} | "
                         f"{r.get('usage', {}).get('output_tokens', '-')} | "
                         f"{round(r.get('seconds', 0) / 60, 1) if r.get('seconds') else '-'} | "
                         f"{r.get('merge', {}).get('merged', '-')} |")
        lines += ["", "To review: `git fetch " + str(self.integration) + " integration:night-review` in your repository.", ""]
        for tid, r in self.results.items():
            if r.get("status") != "passed":
                lines += [f"## {tid}: {r.get('status')}", "```", (r.get("last_failure") or r.get("error") or "")[-1200:], "```"]
        (self.root / "report.md").write_text("\n".join(lines) + "\n")

    def run(self):
        self.prepare_integration()
        stop = threading.Event()
        threading.Thread(target=self.heartbeat, args=(stop,), daemon=True).start()
        self.event("night_start", tasks=list(self.tasks), agents=self.plan.get("agents", 3))
        threads = [threading.Thread(target=self.worker, args=(i,)) for i in range(self.plan.get("agents", 3))]
        for th in threads:
            th.start()
            time.sleep(2)
        for th in threads:
            th.join()
        stop.set()
        self.report()
        self.event("night_end", states=dict(self.state))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    args = ap.parse_args()
    Night(json.loads(Path(args.plan).read_text())).run()
