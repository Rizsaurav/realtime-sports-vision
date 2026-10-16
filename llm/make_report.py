#!/usr/bin/env python3
"""Turn the raw experiment logs in llm/results/ into the tables used in llm/REPORT.md.

  llm/collect_results.sh        # copies the small raw logs from the GPU box into llm/results/
  python llm/make_report.py     # writes llm/results/tables.md
"""
import json
import math
import re
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
LABELS = {
    "1_eager_nospec": "Eager, no speculation (the slow setup)",
    "2_graphs_nospec": "CUDA graphs, no speculation",
    "3_graphs_mtp1": "CUDA graphs + MTP, 1 draft token",
    "4_graphs_mtp2": "CUDA graphs + MTP, 2 draft tokens",
    "5_graphs_mtp3": "CUDA graphs + MTP, 3 draft tokens",
    "6_graphs_mtp4": "CUDA graphs + MTP, 4 draft tokens",
    "7_graphs_mtp5": "CUDA graphs + MTP, 5 draft tokens",
}


def parse_bench(path):
    if not path.exists():
        return None
    t = path.read_text()
    m = re.search(r"decode tok/s: mean ([0-9.]+)\s+median ([0-9.]+)\s+min ([0-9.]+)\s+max ([0-9.]+)\s+ttft median (\d+)", t)
    a = re.search(r"MTP acceptance: \d+/\d+ draft tokens = ([0-9.]+)%", t)
    if not m:
        return None
    return {"mean": float(m[1]), "median": float(m[2]), "min": float(m[3]), "max": float(m[4]),
            "ttft": int(m[5]), "accept": float(a[1]) if a else None}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def speed_table():
    rows = ["| Configuration | Instruct tok/s (mean) | median | Thinking tok/s (mean) | MTP accept. (instruct / thinking) | TTFT ms | Coding check |",
            "|---|---|---|---|---|---|---|"]
    base = None
    for name, label in LABELS.items():
        d = R / "sweep" / name
        i, t = parse_bench(d / "bench_instruct.txt"), parse_bench(d / "bench_thinking.txt")
        if not i:
            continue
        q = ""
        qf = d / "quality.json"
        if qf.exists():
            j = json.loads(qf.read_text())
            q = f"{j['pass']}/{j['n']}"
        if name == "2_graphs_nospec":
            base = i["mean"]
        acc = " / ".join(f"{x['accept']:.0f}%" if x and x["accept"] is not None else "-" for x in (i, t))
        speed = f"{i['mean']:.1f}" + (f" ({i['mean'] / base:.2f}x)" if base and name > "2" else "")
        rows.append(f"| {label} | {speed} | {i['median']:.1f} | {t['mean']:.1f} | {acc} | {i['ttft']} | {q} |"
                    if t else f"| {label} | {speed} | {i['median']:.1f} | - | {acc} | {i['ttft']} | {q} |")
    return "\n".join(rows)


def swe_table(run):
    f = R / "swe" / run / "run.jsonl"
    if not f.exists():
        return ""
    inst, pulled, start = {}, {}, None
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r["event"] == "instance":
            inst.setdefault(r["arm"], {})[r["iid"]] = r
        elif r["event"] == "pulled":
            pulled[r["iid"]] = r
        elif r["event"] == "start":
            start = r
    arms = sorted(inst)
    rows = ["| Arm | Instances | Resolved | Rate (95% CI) | Mean agent time (s) | Mean model calls | Hit step limit | Draft acceptance |",
            "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        rs = list(inst[a].values())
        k = sum(1 for r in rs if r["resolved"])
        lo, hi = wilson(k, len(rs))
        lim = sum(1 for r in rs if r["exit_status"] == "LimitsExceeded")
        acc = sum(r["vllm_delta"].get("spec_decode_num_accepted_tokens_total", 0) for r in rs)
        drf = sum(r["vllm_delta"].get("spec_decode_num_draft_tokens_total", 0) for r in rs)
        rows.append(f"| {a} | {len(rs)} | {k} | {100 * k / len(rs):.1f}% ({100 * lo:.0f}-{100 * hi:.0f}%) | "
                    f"{sum(r['wall_s'] for r in rs) / len(rs):.0f} | {sum(r['api_calls'] for r in rs) / len(rs):.1f} | "
                    f"{lim} | {f'{100 * acc / drf:.0f}%' if drf else '-'} |")
    out = "\n".join(rows)
    if len(arms) == 2:
        a, b = arms
        common = sorted(set(inst[a]) & set(inst[b]))
        both = sum(1 for i in common if inst[a][i]["resolved"] and inst[b][i]["resolved"])
        only_a = sum(1 for i in common if inst[a][i]["resolved"] and not inst[b][i]["resolved"])
        only_b = sum(1 for i in common if inst[b][i]["resolved"] and not inst[a][i]["resolved"])
        neither = len(common) - both - only_a - only_b
        out += (f"\n\nPaired on {len(common)} common instances: both {both}, only {a} {only_a}, only {b} {only_b}, "
                f"neither {neither}.")
    sizes = [p["size_gb"] for p in pulled.values()]
    if sizes:
        out += (f"\n\nImages: {len(sizes)} pulled, mean {sum(sizes) / len(sizes):.2f} GB (max {max(sizes):.2f} GB) "
                f"as reported by Docker; every one was removed after its instance.")
    return out


def main():
    parts = ["## Speed and coding check by configuration\n", speed_table() or "(no sweep results yet)"]
    for run in sorted(p.name for p in (R / "swe").glob("*") if p.is_dir()) if (R / "swe").exists() else []:
        t = swe_table(run)
        if t:
            parts += [f"\n\n## SWE-bench Verified: {run}\n", t]
    (R / "tables.md").write_text("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
