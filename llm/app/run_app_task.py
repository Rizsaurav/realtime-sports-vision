#!/usr/bin/env python3
"""Build one app with the model acting as an autonomous coding agent, then grade it with hidden tests.

The agent (mini-swe-agent) works in a throwaway Docker container (python:3.12-slim) with limits on memory, CPUs and
processes. It reads a client-style specification, builds the app in /workspace, tests it itself, and submits. After it
finishes, the hidden acceptance tests (never shown to it) are copied into the same container and run there.

  python llm/app/run_app_task.py --task notes_api --port 8000 --out /data/saurav/results/app/run1
"""
import argparse
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.utils.serialize import recursive_merge

HERE = Path(__file__).resolve().parent
TASKS = {"notes_api": "test_notes_api.py", "expense_cli": "test_expense_cli.py", "kanban_board": "test_kanban_board.py"}
IMAGE = "python:3.12-slim"

COMMON = """A client has asked you to build the application described below. Work in /workspace (Python 3.12 is installed
and `pip install` works). Build the whole application yourself. Run it and test it with your own checks (curl, a Python
script, or tests you write) before you submit; fix whatever you find. An automated acceptance test suite that you cannot
see will check the exact contract in the specification, so follow every detail literally, including field names, status
codes, output formats and file names.

--- SPECIFICATION ---
{spec}
--- END OF SPECIFICATION ---
"""

WORKFLOW = """## Recommended Workflow
1. Read the specification carefully and note every requirement and edge case
2. Plan the files you need, then write the code
3. Run it and exercise every requirement yourself, including error cases and restart persistence where relevant
4. Fix problems you find, then re-check
5. Submit your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>
"""


def build_config(args):
    cfg = yaml.safe_load((Path(builtin_config_dir) / "mini.yaml").read_text())
    agent = dict(cfg["agent"])
    agent.pop("mode", None)                                   # run unattended
    t = agent["instance_template"].replace("Please solve this issue: {{task}}", "{{task}}")
    t = t.replace("You can execute bash commands and edit files to implement the necessary changes.\n", "")
    t = re.sub(r"## Recommended Workflow.*?(?=## Command Execution Rules)", WORKFLOW + "\n", t, flags=re.S)
    agent.update(instance_template=t, step_limit=args.step_limit, cost_limit=0)
    kwargs = {"api_base": f"http://127.0.0.1:{args.port}/v1", "api_key": "EMPTY", "temperature": args.temperature,
              "top_p": 0.95, "max_tokens": args.max_tokens, "timeout": 1800,
              "extra_body": {"top_k": 20, "min_p": 0.0,
                             "chat_template_kwargs": {"enable_thinking": not args.no_thinking}}}
    model = recursive_merge(cfg["model"], {"model_name": "hosted_vllm/qwen3.8-27b", "cost_tracking": "ignore_errors",
                                           "model_kwargs": kwargs})
    env = recursive_merge(cfg.get("environment", {}), {
        "environment_class": "docker", "image": IMAGE, "cwd": "/workspace", "timeout": 300,
        "container_timeout": "1h", "pull_timeout": 600,
        "run_args": ["--rm", "--memory", "8g", "--memory-swap", "8g", "--cpus", "4", "--pids-limit", "1024",
                     "--log-driver", "none"],
        "env": {"PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONUNBUFFERED": "1"}})
    return agent, model, env


def kill_leftover_python(env):
    env.execute({"command": (
        "python - <<'PY'\nimport os, signal\nme = os.getpid()\n"
        "for p in os.listdir('/proc'):\n    if p.isdigit() and int(p) != me:\n        try:\n"
        "            c = open(f'/proc/{p}/cmdline','rb').read().decode(errors='ignore')\n"
        "            if 'python' in c or 'flask' in c or 'uvicorn' in c:\n"
        "                os.kill(int(p), signal.SIGKILL)\n        except Exception:\n            pass\nPY")}, cwd="/")


def grade(env, task, out):
    tests = (HERE / "tests" / TASKS[task]).read_text()
    kill_leftover_python(env)
    env.execute({"command": "mkdir -p /tests && cat > /tests/test_hidden.py <<'HIDDEN_TESTS_END_7f3a'\n"
                            + tests + "\nHIDDEN_TESTS_END_7f3a"}, cwd="/")
    env.execute({"command": "pip install -q pytest requests 2>&1 | tail -2"}, cwd="/", timeout=600)
    r = env.execute({"command": "cd /workspace && timeout 900 python -m pytest /tests/test_hidden.py -q -rA "
                                "-p no:cacheprovider --tb=short 2>&1 | tail -250"}, cwd="/workspace", timeout=1000)
    text = r["output"]
    (out / "acceptance.txt").write_text(text)
    passed = re.findall(r"^PASSED\s+\S+::(\S+)", text, flags=re.M)
    failed = re.findall(r"^FAILED\s+\S+::(\S+)", text, flags=re.M)
    errors = re.findall(r"^ERROR\s+\S+::(\S+)", text, flags=re.M)
    return {"passed": len(passed), "failed": len(failed), "errors": len(errors), "total": len(passed) + len(failed) + len(errors),
            "failed_tests": failed + errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-limit", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) / args.task
    out.mkdir(parents=True, exist_ok=True)
    spec = (HERE / "specs" / {"notes_api": "notes_api.md", "expense_cli": "expense_cli.md",
                              "kanban_board": "kanban_board.md"}[args.task]).read_text()
    agent_cfg, model_cfg, env_cfg = build_config(args)
    model = get_model(config=model_cfg)
    env = get_environment(env_cfg)
    env.execute({"command": "mkdir -p /workspace"}, cwd="/")
    agent = DefaultAgent(model, env, **agent_cfg)
    t0 = time.perf_counter()
    exit_status, submission, err = None, None, None
    try:
        info = agent.run(COMMON.format(spec=spec))
        exit_status, submission = info.get("exit_status"), info.get("submission")
    except Exception as e:
        exit_status, err = type(e).__name__, traceback.format_exc()
    agent_s = time.perf_counter() - t0
    result = {"task": args.task, "exit_status": exit_status, "agent_seconds": round(agent_s, 1),
              "messages": len(agent.messages), "settings": {"temperature": args.temperature,
                                                            "thinking": not args.no_thinking, "step_limit": args.step_limit}}
    if err:
        result["error"] = err[-1500:]
    try:
        result["acceptance"] = grade(env, args.task, out)
        subprocess.run(f"docker exec {env.container_id} tar czf - -C /workspace --exclude=.venv --exclude=venv "
                       f"--exclude=node_modules --exclude=__pycache__ . > {out}/workspace.tgz", shell=True, timeout=120)
    except Exception:
        result["grading_error"] = traceback.format_exc()[-1500:]
    try:
        agent.save(out / "trajectory.json", {"info": {"exit_status": exit_status, "submission": submission}})
    except Exception:
        pass
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "error"}, indent=1))
    env.cleanup()


if __name__ == "__main__":
    main()
