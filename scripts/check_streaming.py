#!/usr/bin/env python3
"""Regression check: the streaming decision engine (src/sideline/decisions.py)
against the batch decision code (gsr_assemble.py) on a cached perception run.

Replays every clip frame by frame through DecisionEngine, writes SoccerNet-GSR
predictions, scores them with the official evaluator, and reports per-box
agreement of role / team / jersey / identity with the batch predictions.

Usage:
  python scripts/check_streaming.py --perception results/gsr_perception/s1280_kf10_mcbyte_shcmc2 \
      --jersey results/jersey_scores_valid_mcbyte_pose.npz --trackeval /data/saurav/repos/sn-trackeval
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gsr_assemble import CAT, assemble_seq, link_by_number, run_eval
from gsr_jersey_eval import gate
from sideline.decisions import DecisionEngine

STITCH = dict(max_gap=150, v_max=0.4, d0=2.0, feat_thr=3.0)


def stream_seq(pr, jz, L, tau, ratio):
    jer = {}
    if jz is not None:
        m = np.flatnonzero(jz["seq"] == pr["seq"])
        jer = {int(jz["ann_id"][i]): (jz["P"][i], jz["Pnone"][i]) for i in m}
    by_frame = {}
    for i, r in enumerate(pr["rows"]):
        by_frame.setdefault(r["frame"], []).append({**r, "jersey": jer.get(i)})
    eng = DecisionEngine(delay=L, tau=tau, ratio=ratio, role_window=25, stitch=STITCH)
    out = []
    for f in sorted(by_frame):
        out += eng.push({"frame": f, "image_id": by_frame[f][0]["image_id"], "rows": by_frame[f]})
    out += eng.flush()
    preds = []
    for fr in out:
        for j, p in enumerate(fr["preds"]):
            x, y, w, h = p["bbox"]
            q = p["pitch"]
            preds.append({
                "id": f"{fr['image_id']}_{j}", "image_id": fr["image_id"], "track_id": p["track_id"],
                "supercategory": "object", "category_id": CAT[p["attributes"]["role"]],
                "attributes": p["attributes"],
                "bbox_image": {"x": x, "y": y, "w": w, "h": h, "x_center": x + w / 2, "y_center": y + h / 2},
                "bbox_pitch": {"x_bottom_left": q[0], "y_bottom_left": q[1], "x_bottom_middle": q[2],
                               "y_bottom_middle": q[3], "x_bottom_right": q[4], "y_bottom_right": q[5]}})
    return preds


def agreement(a, b):
    key = lambda p: (p["image_id"], round(p["bbox_image"]["x"], 2), round(p["bbox_image"]["y"], 2))
    B = {key(p): p for p in b}
    n, same = 0, {"role": 0, "team": 0, "jersey": 0, "track": 0}
    for p in a:
        q = B.get(key(p))
        if q is None:
            continue
        n += 1
        for k in ("role", "team", "jersey"):
            same[k] += p["attributes"][k] == q["attributes"][k]
        same["track"] += p["track_id"] == q["track_id"]
    return n, same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", required=True)
    ap.add_argument("--jersey", required=True)
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--trackeval", required=True)
    ap.add_argument("--lookahead", type=int, nargs="+", default=[0, 25, 125])
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--name", default="stream")
    args = ap.parse_args()

    pdir = Path(args.perception)
    seqs = sorted(p.stem for p in pdir.glob("SNGS-*.json"))
    prs = {s: json.loads((pdir / f"{s}.json").read_text()) for s in seqs}
    jz = dict(np.load(args.jersey))
    jz["P"], jz["Pnone"] = gate(jz["P"], 1.0 - jz["leg"], 0.97, 0.3)
    root = Path("results/_work/streaming_check").resolve() / args.name
    shutil.rmtree(root, ignore_errors=True)
    tags, agree = [], {}
    for L in args.lookahead:
        tag = f"{args.name}_L{L}"
        out = root / f"SoccerNetGS-{args.split}" / tag / "data"
        out.mkdir(parents=True)
        n, same = 0, {"role": 0, "team": 0, "jersey": 0, "track": 0}
        for s in seqs:
            sp = stream_seq(prs[s], jz, L, args.tau, args.ratio)
            bp = link_by_number(assemble_seq(prs[s], jz, L, args.tau, args.ratio, 25, STITCH))
            (out / f"{s}.json").write_text(json.dumps({"predictions": sp}))
            k, sm = agreement(sp, bp)
            n += k
            for key in same:
                same[key] += sm[key]
        agree[L] = {k: round(100 * v / n, 3) for k, v in same.items()} | {"boxes": n}
        print(f"L={L}: agreement with batch {agree[L]}", flush=True)
        tags.append(tag)
    with ThreadPoolExecutor(len(tags)) as ex:
        res = list(ex.map(lambda t: run_eval(args.trackeval, Path(args.gt).resolve(), root, args.split, t, seqs),
                          tags))
    summary = {}
    for L, r in zip(args.lookahead, res):
        summary[L] = {"gs_hota": r["HOTA"], "deta": r["DetA"], "assa": r["AssA"], "agreement": agree[L]}
        print(f"{args.name} L={L:4d} GS-HOTA={r['HOTA']:6.2f} DetA={r['DetA']:6.2f} AssA={r['AssA']:6.2f}")
    Path(f"results/streaming_check_{args.name}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
