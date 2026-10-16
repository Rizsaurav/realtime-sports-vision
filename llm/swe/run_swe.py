#!/usr/bin/env python3
"""Controlled SWE-bench Verified experiment: the same agent and instances against two servers.

For each sampled instance: pull its image, run mini-swe-agent once per arm (each arm is one
vLLM server), evaluate each patch with the official swebench harness, then remove the image.
Disk use is bounded: at most `--workers` images exist at a time, a hard cap is checked before
every pull, images are removed by exact name (never `docker system prune`), and the Docker
state is compared with the state at the start. Every step is written to a JSONL log.

  python llm/swe/run_swe.py --arms mtp3=8000 nospec=8001 --n 30 --workers 2 --cap-gb 20
  python llm/swe/run_swe.py --arms mtp3=8000 --n 1            # smoke test, measures image size
"""
import argparse
import json
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VENV_BIN = Path(sys.executable).parent
UNITS = {"B": 1, "kB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}
LOG_LOCK = threading.Lock()
DATASET = "SWE-bench/SWE-bench_Verified"     # same 500 instances as princeton-nlp/SWE-bench_Verified, with image names
IMAGES = {}


def sh(cmd, timeout=None, check=False):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])}: {r.stderr.strip()[-300:]}")
    return r


def to_bytes(s):
    m = re.match(r"([0-9.]+)\s*([kMGT]?B)", s.strip())
    return float(m.group(1)) * UNITS[m.group(2)] if m else 0.0


def docker_df():
    out = {}
    for line in sh(["docker", "system", "df", "--format", "{{json .}}"]).stdout.splitlines():
        row = json.loads(line)
        out[row["Type"]] = {"size": to_bytes(row["Size"]), "count": int(row["TotalCount"])}
    return out


def fs_used(path="/var/lib"):
    return int(sh(["df", "--output=used", "-B1", path]).stdout.split()[1])


def image_name(iid):
    return IMAGES.get(iid) or f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_').lower()}:latest"


def vllm_counters(port):
    try:
        text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    except Exception:
        return {}
    keys = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total")
    out = {}
    for line in text.splitlines():
        for k in keys:
            if line.startswith(k + "{"):
                out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def log(fp, **row):
    row["t"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with LOG_LOCK:
        fp.write(json.dumps(row) + "\n")
        fp.flush()


def arm_config(path, port, args):
    path.write_text(f"""model:
  model_name: hosted_vllm/qwen3.8-27b
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: http://127.0.0.1:{port}/v1
    api_key: EMPTY
    temperature: {args.temperature}
    top_p: 0.95
    max_tokens: {args.max_tokens}
    timeout: 900
    extra_body:
      top_k: 20
      chat_template_kwargs:
        enable_thinking: {str(args.thinking).lower()}
agent:
  step_limit: {args.step_limit}
  cost_limit: 0
environment:
  run_args: ["--rm", "--memory", "{args.mem_gb}g", "--memory-swap", "{args.mem_gb}g", "--cpus", "{args.cpus}",
             "--pids-limit", "1024", "--network", "none", "--log-driver", "none"]
""")


def run_arm(iid, arm, port, out, cfg, args):
    """-> dict with the patch and agent statistics for one instance and one arm."""
    adir = out / arm
    adir.mkdir(parents=True, exist_ok=True)
    c0 = vllm_counters(port)
    t0 = time.perf_counter()
    cmd = [str(VENV_BIN / "mini-extra"), "swebench", "--subset", "verified", "--split", "test",
           "--filter", f"^{re.escape(iid)}$", "-w", "1", "-o", str(adir / "agent"), "-c", "swebench.yaml",
           "-c", str(cfg)]
    with open(adir / f"{iid}.agent.log", "w") as lf:
        try:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=args.instance_timeout)
            status = f"exit {r.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
    wall = time.perf_counter() - t0
    c1 = vllm_counters(port)
    patch, calls = "", 0
    preds = adir / "agent" / "preds.json"
    if preds.exists():
        p = json.loads(preds.read_text()).get(iid, {})
        patch = p.get("model_patch", "") or ""
    traj = adir / "agent" / iid / f"{iid}.traj.json"
    exit_status = None
    if traj.exists():
        t = json.loads(traj.read_text())
        info = t.get("info", {})
        exit_status = info.get("exit_status")
        calls = (info.get("model_stats") or {}).get("api_calls", 0)
    d = {k.split(":")[1]: c1.get(k, 0) - c0.get(k, 0) for k in c1}
    return {"arm": arm, "status": status, "exit_status": exit_status, "wall_s": round(wall, 1),
            "api_calls": calls, "patch_chars": len(patch), "patch": patch, "vllm_delta": d}


def evaluate(iid, arm, patch, out, args):
    """Official swebench harness on one instance; -> True/False/None (None = error)."""
    edir = out / arm / "eval"
    edir.mkdir(parents=True, exist_ok=True)
    pred = edir / f"{iid}.preds.jsonl"
    pred.write_text(json.dumps({"instance_id": iid, "model_name_or_path": arm, "model_patch": patch}) + "\n")
    run_id = f"{arm}_{iid}"
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name",
           DATASET, "--split", "test", "--predictions_path", str(pred),
           "--instance_ids", iid, "--max_workers", "1", "--run_id", run_id,
           "--timeout", str(args.eval_timeout), "--report_dir", str(edir)]
    with open(edir / f"{iid}.eval.log", "w") as lf:
        sh_r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=edir, timeout=args.eval_timeout + 900)
    rep = edir / f"{arm}.{run_id}.json"
    if sh_r.returncode or not rep.exists():
        return None
    return iid in json.loads(rep.read_text()).get("resolved_ids", [])


def cleanup(iid, image, started_after):
    """Remove only this instance's image and containers whose names mark them as ours."""
    names = sh(["docker", "ps", "-a", "--format", "{{.Names}}"]).stdout.split()
    mine = [n for n in names if n.startswith("minisweagent-") or n.startswith(f"sweb.eval.{iid}")]
    for n in mine:
        sh(["docker", "rm", "-f", n])
    sh(["docker", "rmi", "-f", image])
    # also any image whose name carries this instance id (guards against a naming difference)
    key = iid.replace("__", "_1776_").lower()
    for line in sh(["docker", "images", "--format", "{{.Repository}}:{{.Tag}} {{.ID}}"]).stdout.splitlines():
        name, img_id = line.split()
        if name.startswith("swebench/sweb.eval") and key in name.lower() and img_id not in started_after:
            sh(["docker", "rmi", "-f", name])
    return mine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=vllm_port, for example mtp3=8000")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None, help="explicit instance ids instead of a sample")
    ap.add_argument("--limit", type=int, default=0, help="run only the first LIMIT of the sampled ids (pilot)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cap-gb", type=float, default=20.0, help="max Docker image growth over the baseline")
    ap.add_argument("--fs-cap-gb", type=float, default=30.0, help="max growth of the Docker filesystem")
    ap.add_argument("--step-limit", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--mem-gb", type=int, default=12)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--instance-timeout", type=int, default=2400)
    ap.add_argument("--eval-timeout", type=int, default=1800)
    ap.add_argument("--out", default="/data/saurav/results/swe")
    ap.add_argument("--name", default=time.strftime("run_%Y%m%d_%H%M"))
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(DATASET, split="test")
    IMAGES.update({r["instance_id"]: r["image"] for r in ds})
    all_ids = sorted(ds["instance_id"])
    ids = args.ids or random.Random(args.seed).sample(all_ids, args.n)
    ids = ids[: args.limit or None]
    arms = [a.split("=") for a in args.arms]
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    for arm, port in arms:
        arm_config(out / f"{arm}.yaml", port, args)

    base_df, base_fs = docker_df(), fs_used()
    base_images = set(sh(["docker", "images", "-q"]).stdout.split())
    fp = open(out / "run.jsonl", "a")
    log(fp, event="start", ids=ids, arms=args.arms, baseline_df=base_df, baseline_fs_bytes=base_fs,
        caps={"images_gb": args.cap_gb, "fs_gb": args.fs_cap_gb}, args=vars(args))
    print(f"[swe] {len(ids)} instances, arms {args.arms}, baseline images {base_df['Images']['size'] / 1e9:.2f} GB",
          flush=True)
    stop = threading.Event()
    results = {}

    def one(iid):
        if stop.is_set():
            return
        df, fs = docker_df(), fs_used()
        grown, fs_grown = df["Images"]["size"] - base_df["Images"]["size"], fs - base_fs
        if grown > args.cap_gb * 1e9 or fs_grown > args.fs_cap_gb * 1e9:
            log(fp, event="budget_stop", iid=iid, image_growth_gb=grown / 1e9, fs_growth_gb=fs_grown / 1e9)
            print(f"[swe] BUDGET STOP at {iid}: images +{grown / 1e9:.1f} GB, fs +{fs_grown / 1e9:.1f} GB", flush=True)
            stop.set()
            return
        image = image_name(iid)
        t0 = time.perf_counter()
        pr = sh(["docker", "pull", "-q", f"docker.io/{image}"], timeout=1800)
        if pr.returncode:
            log(fp, event="pull_failed", iid=iid, err=pr.stderr.strip()[-300:])
            return
        size = float(sh(["docker", "image", "inspect", image, "--format", "{{.Size}}"]).stdout.strip() or 0)
        log(fp, event="pulled", iid=iid, image=image, size_gb=round(size / 1e9, 2), secs=round(time.perf_counter() - t0, 1))
        try:
            with ThreadPoolExecutor(len(arms)) as ex:
                futs = [ex.submit(run_arm, iid, a, p, out, out / f"{a}.yaml", args) for a, p in arms]
                runs = [f.result() for f in futs]
            for r in runs:
                r["resolved"] = evaluate(iid, r["arm"], r["patch"], out, args) if r["patch"] else False
                (out / r["arm"] / f"{iid}.patch").write_text(r.pop("patch"))
                log(fp, event="instance", iid=iid, **r)
                results.setdefault(r["arm"], {})[iid] = r
                print(f"[swe] {iid:36s} {r['arm']:8s} resolved={r['resolved']} wall={r['wall_s']:.0f}s "
                      f"calls={r['api_calls']} exit={r['exit_status']}", flush=True)
        finally:
            removed = cleanup(iid, image, base_images)
            after = docker_df()
            log(fp, event="cleaned", iid=iid, removed_containers=removed,
                image_growth_gb=round((after["Images"]["size"] - base_df["Images"]["size"]) / 1e9, 2),
                fs_growth_gb=round((fs_used() - base_fs) / 1e9, 2))

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, ids))

    final = docker_df()
    leftover = set(sh(["docker", "images", "-q"]).stdout.split()) - base_images
    summary = {"ids_requested": len(ids), "arms": {}}
    for arm, rs in results.items():
        done = [r for r in rs.values() if r["resolved"] is not None]
        summary["arms"][arm] = {"instances": len(rs), "resolved": sum(1 for r in done if r["resolved"]),
                                "mean_wall_s": round(sum(r["wall_s"] for r in rs.values()) / max(len(rs), 1), 1),
                                "mean_api_calls": round(sum(r["api_calls"] for r in rs.values()) / max(len(rs), 1), 1)}
    summary["docker_after"] = final
    summary["docker_baseline"] = base_df
    summary["leftover_images"] = sorted(leftover)
    summary["fs_growth_gb"] = round((fs_used() - base_fs) / 1e9, 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(fp, event="done", **summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
