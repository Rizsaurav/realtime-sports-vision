"""Self-improvement loop for the supervisor: automated checks, a review by a model, adversarial tests, fixes.

All reading and writing of code is done by the coding agents. This module only runs cheap static checks, asks a
reviewer agent for a structured verdict on the written expectations, runs what the agents produce, and condenses
everything into digest.md, a short page meant for a human who does not want to read code.
"""
import ast
import json
import re
import time
from pathlib import Path

SEV = {"high": 3, "medium": 2, "low": 1}


# ---------------------------------------------------------------------------------------------- static checks
def py_metrics(path):
    src = path.read_text(errors="ignore")
    out = {"file": str(path), "loc": sum(1 for l in src.splitlines() if l.strip() and not l.strip().startswith("#"))}
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {**out, "syntax_error": str(e)}
    lens, cc, depth = [], [], 0

    def nest(n, d=0):
        nonlocal depth
        for c in ast.iter_child_nodes(n):
            deeper = isinstance(c, (ast.If, ast.For, ast.While, ast.Try, ast.With))
            depth = max(depth, d + (1 if deeper else 0))
            nest(c, d + (1 if deeper else 0))

    nest(tree)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lens.append((n.end_lineno - n.lineno + 1, n.name))
            cc.append((1 + sum(isinstance(x, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.IfExp, ast.BoolOp, ast.comprehension))
                               for x in ast.walk(n)), n.name))
    out.update(functions=len(lens), longest_function=max(lens, default=(0, ""))[0],
               longest_function_name=max(lens, default=(0, ""))[1], max_complexity=max(cc, default=(0, ""))[0],
               max_complexity_name=max(cc, default=(0, ""))[1], max_nesting=depth,
               bare_except=len(re.findall(r"except\s*:", src)) + len(re.findall(r"except[^:\n]*:\s*\n\s*pass", src)),
               prints=len(re.findall(r"^\s*print\(", src, re.M)), todos=len(re.findall(r"TODO|FIXME|XXX", src)),
               commented_code=len(re.findall(r"^\s*#\s*(def |class |import |return |if .*:|for .*:)", src, re.M)),
               long_lines=sum(1 for l in src.splitlines() if len(l) > 130))
    return out


def js_metrics(path):
    src = path.read_text(errors="ignore")
    return {"file": str(path), "loc": sum(1 for l in src.splitlines() if l.strip() and not l.strip().startswith("//")),
            "functions": len(re.findall(r"\bfunction\b|=>", src)), "console_logs": len(re.findall(r"console\.(log|debug)", src)),
            "todos": len(re.findall(r"TODO|FIXME|XXX", src)), "empty_catch": len(re.findall(r"catch\s*(\([^)]*\))?\s*\{\s*\}", src)),
            "long_lines": sum(1 for l in src.splitlines() if len(l) > 130)}


def gates(night, task, d):
    """Cheap, model-free measurements of the files a task owns."""
    work = d["work"]
    files = [work / f for f in task.get("files", []) if (work / f).is_file()]
    py = [f for f in files if f.suffix == ".py"]
    js = [f for f in files if f.suffix in (".js", ".mjs")]
    m = {"files": [str(f.relative_to(work)) for f in files], "py": [py_metrics(f) for f in py], "js": [js_metrics(f) for f in js]}
    flakes = 0
    if py and night.pyflakes.exists():
        r = night_sh(night, [str(night.pyflakes), "-m", "pyflakes", *[str(f) for f in py]])
        flakes = len([l for l in r.stdout.splitlines() if l.strip()])
    m["lint_warnings"] = flakes
    # statement coverage of the task's python files while its acceptance tests run (inside the sandbox)
    cov = None
    pyfiles = ",".join(str(f.relative_to(work)) for f in py)
    tests = [c for c in task.get("acceptance", []) if "pytest" in c]
    if py and tests and pyfiles:
        cmd = f"python -m coverage run --include='{pyfiles}' -m {tests[0].split('python -m ', 1)[-1]} >/dev/null 2>&1; python -m coverage report --include='{pyfiles}' | tail -1"
        rc, st, out = night.run_sandboxed(d, ["/bin/sh", "-c", cmd], timeout=600)
        mm = re.search(r"(\d+)%", out)
        cov = int(mm.group(1)) if mm else None
    m["coverage_pct"] = cov
    bad = []
    for x in m["py"]:
        if x.get("syntax_error"):
            bad.append(f"{x['file']}: syntax error")
        if x.get("longest_function", 0) > 80:
            bad.append(f"{x['file']}: function {x['longest_function_name']} is {x['longest_function']} lines")
        if x.get("max_complexity", 0) > 25:
            bad.append(f"{x['file']}: function {x['max_complexity_name']} has complexity {x['max_complexity']}")
        if x.get("bare_except"):
            bad.append(f"{x['file']}: {x['bare_except']} bare or silent except")
        if x.get("prints"):
            bad.append(f"{x['file']}: {x['prints']} print() calls")
    if flakes:
        bad.append(f"{flakes} lint warnings (pyflakes)")
    for x in m["js"]:
        if x.get("console_logs"):
            bad.append(f"{x['file']}: {x['console_logs']} console.log calls")
        if x.get("empty_catch"):
            bad.append(f"{x['file']}: {x['empty_catch']} empty catch blocks")
    if cov is not None and cov < task.get("min_coverage", 85):
        bad.append(f"test coverage {cov}% is below {task.get('min_coverage', 85)}%")
    m["problems"] = bad
    return m


def night_sh(night, cmd):
    import subprocess
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


# ------------------------------------------------------------------------------------------------- prompts
def reviewer_prompt(task, m):
    exp = "\n".join(f"  {i + 1}. {e}" for i, e in enumerate(task.get("expectations", [])))
    probs = "\n".join(f"  - {p}" for p in m["problems"]) or "  (none)"
    return (f"You are a strict code reviewer. Work in the current directory (a git repository). Do NOT change any tracked file.\n\n"
            f"The task being reviewed is described in `{task['spec']}` (read it fully). The files to review: "
            f"{', '.join(task.get('files', []))}. Run the acceptance checks if you want to see them pass: "
            f"{'; '.join(task['acceptance'])}.\n\n"
            f"The owner's written expectations (judge each one separately, with evidence):\n{exp}\n\n"
            f"Automated measurements found these problems:\n{probs}\n\n"
            f"Review the code the way a senior engineer would: read every file listed in full, then look for real bugs, "
            f"behaviour the specification demands that the code does not implement, unhandled errors and edge cases, "
            f"duplicated logic, dead code, unclear naming, functions that do several jobs, and anything that could not "
            f"possibly work. Do not praise. Do not invent problems: every issue must cite a file and line you have read.\n\n"
            f"Write your result as JSON to `.context/review.json` (create the folder if needed), with exactly this shape:\n"
            f'{{"expectations": [{{"id": 1, "verdict": "pass|partial|fail", "evidence": "file:line and a short reason"}}], '
            f'"issues": [{{"severity": "high|medium|low", "file": "path", "line": 0, "problem": "what is wrong", "fix": "what to change"}}], '
            f'"spec_gaps": ["a behaviour in the specification the code does not implement"], '
            f'"quality": {{"readability": 1, "duplication": 1, "error_handling": 1, "test_strength": 1}}, '
            f'"lessons": ["a general rule that would have avoided the biggest problems, written for the next engineer"]}}\n'
            f"The quality scores are 1 (poor) to 5 (excellent). Use severity high only for bugs or missing required behaviour, "
            f"medium for real weaknesses, low for style. When the JSON file is written, reply with one line: done.\n")


def hardener_prompt(task):
    return (f"You are a test engineer. Work in the current directory (a git repository). Do NOT change any source file or any "
            f"existing test.\n\nThe task is described in `{task['spec']}`; the implementation is in "
            f"{', '.join(task.get('files', []))}. Read the specification and the implementation, then write NEW tests that "
            f"try to break the implementation: boundary values, empty and malformed input, large input, repeated calls, "
            f"concurrency or ordering effects where relevant, and every rule of the specification the existing tests do not "
            f"check. Each test must assert behaviour the SPECIFICATION requires (not behaviour you merely guess). Put them in "
            f"new files only: Python tests as `tests_extra/test_{task['id']}_extra.py`, JavaScript tests as "
            f"`web_tests_extra/{task['id']}_extra.test.mjs`. Keep the whole extra suite fast (under 60 seconds) and "
            f"deterministic. Run them. Leave failing tests in place when they fail because the implementation is wrong; "
            f"delete a test if you find it contradicts the specification. Commit the new files. When finished, reply with one "
            f"line: the number of tests and how many fail.\n")


def fixer_prompt(task, issues, extra_failures):
    lines = [f"- [{i['severity']}] {i.get('file', '?')}:{i.get('line', '?')}: {i['problem']} Fix: {i.get('fix', '')}" for i in issues]
    ex = f"\nThese additional tests fail and need the implementation fixed:\n{extra_failures[-3500:]}\n" if extra_failures else ""
    return (f"You are improving existing work in the current directory (a git repository on its own branch). The task is "
            f"`{task['spec']}`. A reviewer found the problems below. Fix them in the source files. Rules: the acceptance checks "
            f"({'; '.join(task['acceptance'])}) must still pass; do not change anything under tests/, web_tests/, tasks/ or "
            f"docs/WEBAPP_CONTRACT.md. If you are certain an extra test under tests_extra/ or web_tests_extra/ contradicts the "
            f"specification, delete that test and write one line saying why in `.context/notes.md`. Do not rewrite working "
            f"code for style. Commit with a short imperative message. Never write anything about who or what wrote the code.\n\n"
            f"Problems:\n" + "\n".join(lines) + ex + "\nWhen done, reply with one line summarising what you changed.\n")


# ------------------------------------------------------------------------------------------------ the loop
def load_review(work):
    try:
        r = json.loads((work / ".context" / "review.json").read_text())
        return r if isinstance(r, dict) else None
    except Exception:
        return None


def open_issues(review, task):
    issues = [i for i in review.get("issues", []) if SEV.get(i.get("severity"), 0) >= 2]
    exp = task.get("expectations", [])
    for e in review.get("expectations", []):
        if e.get("verdict") in ("fail", "partial") and 1 <= e.get("id", 0) <= len(exp):
            issues.append({"severity": "high" if e["verdict"] == "fail" else "medium", "file": "", "line": "",
                           "problem": f"expectation not met ({e['verdict']}): {exp[e['id'] - 1]}. {e.get('evidence', '')}",
                           "fix": "make the code meet this expectation"})
    for g in review.get("spec_gaps", []):
        issues.append({"severity": "high", "file": "", "line": "", "problem": f"specification gap: {g}", "fix": "implement it"})
    return issues


def run_extra_tests(night, d, task):
    work = d["work"]
    cmds = []
    if (work / "tests_extra").is_dir():
        cmds.append("python -m pytest tests_extra -q -x --maxfail=15 2>&1 | tail -60")
    if (work / "web_tests_extra").is_dir():
        cmds.append("node --test web_tests_extra/ 2>&1 | tail -60")
    out, ok = "", True
    for c in cmds:
        rc, st, o = night.run_sandboxed(d, ["/bin/sh", "-c", c.replace(" 2>&1 | tail -60", "")], timeout=600)
        out += f"$ {c}\n{o[-3000:]}\n"
        ok = ok and rc == 0 and st != "timeout"
    return ok, out


def improve_loop(night, task, d, port):
    """Review, harden and fix until clean. Returns a summary dict that goes into the digest."""
    work, rounds = d["work"], task.get("review_rounds", 3)
    summary = {"rounds": [], "final": "not_run"}
    last_ok_commit = night_sh(night, ["git", "-C", str(work), "rev-parse", "HEAD"]).stdout.strip()
    for rnd in range(1, rounds + 1):
        if (night.root / "STOP").exists():
            break
        night.status["agents"][task["id"]] = {"attempt": f"review {rnd}", "phase": "gates", "since": time.time(), "port": port}
        m = gates(night, task, d)
        rec = {"round": rnd, "problems": len(m["problems"]), "coverage": m["coverage_pct"], "lint": m["lint_warnings"]}
        # reviewer (read-only by instruction; verified afterwards)
        night.status["agents"][task["id"]]["phase"] = "review"
        (work / ".context").mkdir(exist_ok=True)
        (work / ".context" / "review.json").unlink(missing_ok=True)
        review = None
        for _ in range(2):
            night.run_pi_agent(d, port, reviewer_prompt(task, m), "high", task.get("timeout_min", 90), f"review{rnd}")
            night_sh(night, ["git", "-C", str(work), "checkout", "--", "."])          # undo any edit by the reviewer
            review = load_review(work)
            if review:
                break
        if not review:
            rec["review"] = "failed to produce a valid review"
            summary["rounds"].append(rec)
            break
        quality = review.get("quality", {})
        rec.update(review_issues=len(review.get("issues", [])), expectations=[e.get("verdict") for e in review.get("expectations", [])],
                   quality=quality, spec_gaps=len(review.get("spec_gaps", [])))
        for lesson in review.get("lessons", [])[:3]:
            night.add_lesson(lesson)
        # adversarial tests, first round only
        extra_fail = ""
        if rnd == 1 and task.get("harden", True):
            night.status["agents"][task["id"]]["phase"] = "harden"
            night.run_pi_agent(d, port, hardener_prompt(task), "medium", task.get("timeout_min", 90), "harden")
            night_sh(night, ["git", "-C", str(work), "add", "-A"])
            night_sh(night, ["git", "-C", str(work), "commit", "-q", "-m", f"Add extra tests for {task.get('title', task['id'])}"])
        if (work / "tests_extra").is_dir() or (work / "web_tests_extra").is_dir():
            ok_x, extra_fail = run_extra_tests(night, d, task)
            rec["extra_tests_pass"] = ok_x
            extra_fail = "" if ok_x else extra_fail
        issues = open_issues(review, task) + [{"severity": "medium", "file": "", "line": "", "problem": p, "fix": "fix it"}
                                              for p in m["problems"]]
        rec["open_issues"] = len(issues) + (1 if extra_fail else 0)
        summary["rounds"].append(rec)
        night.event("review_round", task=task["id"], **{k: str(v)[:80] for k, v in rec.items()})
        if not issues and not extra_fail:
            summary["final"] = "clean"
            break
        if rnd == rounds:
            summary["final"] = "open_issues"
            break
        # fix
        night.status["agents"][task["id"]]["phase"] = "fix"
        night.run_pi_agent(d, port, fixer_prompt(task, issues[:25], extra_fail), "high", task.get("timeout_min", 90), f"fix{rnd}")
        night_sh(night, ["git", "-C", str(work), "add", "-A"])
        night_sh(night, ["git", "-C", str(work), "commit", "-q", "-m", f"Address review findings for {task.get('title', task['id'])}"])
        ok = True
        for c in task["acceptance"]:
            rc, st, o = night.run_sandboxed(d, ["/bin/sh", "-c", c], timeout=900)
            if rc != 0 or st == "timeout":
                ok = False
                break
        traces = night.scan_traces(work, night.base_commit_of(d))
        if not ok or traces:                                  # the fix broke the required behaviour: go back
            night_sh(night, ["git", "-C", str(work), "reset", "-q", "--hard", last_ok_commit])
            summary["rounds"][-1]["fix_reverted"] = True
            night.event("fix_reverted", task=task["id"], round=rnd, traces=len(traces))
        else:
            last_ok_commit = night_sh(night, ["git", "-C", str(work), "rev-parse", "HEAD"]).stdout.strip()
    if summary["final"] == "not_run":
        summary["final"] = "stopped"
    return summary


# ----------------------------------------------------------------------------------------------- the digest
def write_digest(night):
    L = [f"# Night digest  (updated {time.strftime('%H:%M:%S')})", ""]
    st = night.state
    L.append(f"Tasks: {sum(1 for s in st.values() if s == 'done')} done, {sum(1 for s in st.values() if s == 'running')} running, "
             f"{sum(1 for s in st.values() if s == 'pending')} waiting, {sum(1 for s in st.values() if s in ('failed', 'skipped'))} failed or skipped. "
             f"Servers: {night.status.get('servers')}")
    L += ["", "| Task | Status | Build tries | Tokens | Reviews | Expectations met | Open issues | Coverage | Lint | Review quality |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    attention = []
    for tid, t in night.tasks.items():
        r = night.results.get(tid, {})
        q = r.get("quality", {})
        rounds = q.get("rounds", [])
        last = rounds[-1] if rounds else {}
        ex = last.get("expectations") or []
        met = f"{sum(1 for v in ex if v == 'pass')}/{len(ex)}" if ex else "-"
        qual = last.get("quality") or {}
        qs = "/".join(str(qual.get(k, "-")) for k in ("readability", "duplication", "error_handling", "test_strength")) if qual else "-"
        L.append(f"| {tid} | {r.get('status', st[tid])}{' (' + q['final'] + ')' if q.get('final') else ''} | {r.get('attempts', '-')} | "
                 f"{r.get('usage', {}).get('output_tokens', '-')} | {len(rounds) or '-'} | {met} | {last.get('open_issues', '-')} | "
                 f"{str(last.get('coverage', '-')) + '%' if last.get('coverage') is not None else '-'} | {last.get('lint', '-')} | {qs} |")
        if r.get("status") == "failed":
            attention.append(f"ATTENTION: task {tid} failed after {r.get('attempts')} attempts: {(r.get('last_failure') or r.get('error') or '')[-160:]!r}")
        if q.get("final") == "open_issues":
            attention.append(f"ATTENTION: task {tid} finished review rounds with open issues ({last.get('open_issues')})")
    if night.lessons:
        L += ["", "Lessons the agents are now applying:"] + [f"- {x}" for x in night.lessons[-8:]]
    if any(not ok for ok in (night.status.get("servers") or {}).values()):
        attention.append("ATTENTION: a model server is not answering")
    if all(s in ("done", "failed", "skipped") for s in st.values()):
        attention.append("NIGHT END: every task has finished")
    L += [""] + attention
    text = "\n".join(L) + "\n"
    (night.root / "digest.md").write_text(text)
    return text
