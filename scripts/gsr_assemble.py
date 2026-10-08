#!/usr/bin/env python3
"""Stage C of the streaming GSR pipeline: attribute decisions under an output
delay L, then official GS-HOTA.

From the perception cache (gsr_perceive.py) and per-box jersey scores
(score_jersey_parseq.py --perception), for each frame t the system may use
evidence up to t + L frames:
  role   = majority detector class of the track (player/goalkeeper/referee)
  team   = online re-fitted 2-way colour clusters, side by mean pitch x;
           goalkeepers by the half they stand in; referees none
  jersey = legibility-weighted PARSeq evidence per track, committed at tau
L = -1 is the offline upper bound (whole clip).

Usage:
  python scripts/gsr_assemble.py --perception results/gsr_perception/s1280_kf10 \
      --jersey results/jersey_scores_valid_pred.npz --gt data/raw/soccernet/gsr --split valid \
      --trackeval /data/saurav/repos/sn-trackeval --lookahead 0 25 125 -1
"""
import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsr_jersey_eval import predict as jersey_commits
from gsr_team_eval import predict_teams_refit

CAT = {"player": 1, "goalkeeper": 2, "referee": 3}


def roles_online(rows, L):
    """-> {row_index: role} using the track's majority class up to t+L."""
    by_track = {}
    for i, r in enumerate(rows):
        by_track.setdefault(r["track"], []).append(i)
    out = {}
    for tr, idx in by_track.items():
        idx.sort(key=lambda i: rows[i]["frame"])
        frames = [rows[i]["frame"] for i in idx]
        cnt, ptr = Counter(), 0
        for j, i in enumerate(idx):
            horizon = frames[-1] if L < 0 else frames[j] + L
            while ptr < len(idx) and frames[ptr] <= horizon:
                cnt[rows[idx[ptr]]["cls"]] += 1
                ptr += 1
            out[i] = cnt.most_common(1)[0][0]
    return out


def assemble_seq(pr, jz, L, tau, ratio):
    rows = [r for r in pr["rows"] if r["cls"] != "ball"]
    role = roles_online(rows, L)
    team_rows = [(i, r["frame"], r["track"], role[i], r["pitch"][3] if r["pitch"] else 0.0,
                  np.asarray(r["feat"]))
                 for i, r in enumerate(rows)
                 if role[i] in ("player", "goalkeeper") and r["feat"] is not None]
    team = predict_teams_refit(team_rows, L, warmup=25)
    jc = {}
    if jz is not None:
        m = jz["seq"] == pr["seq"]
        if m.any():
            meta = {k: jz[k][m] for k in ("seq", "frame", "track", "ann_id")}
            jc = jersey_commits(meta, jz["P"][m], 1.0 - jz["leg"][m], L, tau, ratio).get(pr["seq"], {})
    preds = []
    for i, r in enumerate(rows):
        if r["pitch"] is None:
            continue
        rl = role[i]
        jersey = None
        if rl in ("player", "goalkeeper"):
            c = jc.get(r["track"])
            jersey = c[1] if c and r["frame"] >= c[0] else None
        x, y, w, h = r["bbox"]
        p = r["pitch"]
        preds.append({
            "id": f"{r['image_id']}_{i}", "image_id": r["image_id"], "track_id": r["track"],
            "supercategory": "object", "category_id": CAT[rl],
            "attributes": {"role": rl, "jersey": jersey,
                           "team": team.get(i) if rl in ("player", "goalkeeper") else None},
            "bbox_image": {"x": x, "y": y, "w": w, "h": h, "x_center": x + w / 2,
                           "y_center": y + h / 2},
            "bbox_pitch": {"x_bottom_left": p[0], "y_bottom_left": p[1],
                           "x_bottom_middle": p[2], "y_bottom_middle": p[3],
                           "x_bottom_right": p[4], "y_bottom_right": p[5]}})
    return preds


def run_eval(trackeval_dir, gt_root, trk_root, split, tracker, seqs, use_attrs=True):
    cmd = [sys.executable, str(Path(trackeval_dir) / "scripts" / "run_soccernet_gs.py"),
           "--GT_FOLDER", str(gt_root), "--TRACKERS_FOLDER", str(trk_root),
           "--SPLIT_TO_EVAL", split, "--TRACKERS_TO_EVAL", tracker,
           "--METRICS", "HOTA", "--PRINT_CONFIG", "False", "--USE_PARALLEL", "False",
           "--USE_ROLES", str(use_attrs), "--USE_TEAMS", str(use_attrs),
           "--USE_JERSEY_NUMBERS", str(use_attrs), "--SEQ_INFO", *seqs]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    summ = trk_root / f"SoccerNetGS-{split}" / tracker / "cls_comb_det_av_summary.txt"
    head, vals = summ.read_text().splitlines()[:2]
    return dict(zip(head.split(), map(float, vals.split())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", required=True)
    ap.add_argument("--jersey", default="")
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--trackeval", required=True)
    ap.add_argument("--lookahead", type=int, nargs="+", default=[0, 25, 125, -1])
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    pdir = Path(args.perception)
    name = args.name or pdir.name
    gt_root = Path(args.gt).resolve()
    seqs = sorted(p.stem for p in pdir.glob("SNGS-*.json"))
    prs = {s: json.loads((pdir / f"{s}.json").read_text()) for s in seqs}
    jz = dict(np.load(args.jersey)) if args.jersey else None
    trk_root = Path("results/_work/gsr_e2e").resolve() / name
    shutil.rmtree(trk_root, ignore_errors=True)
    tags = []
    for L in args.lookahead:
        tag = f"{name}_L{'inf' if L < 0 else L}"
        out = trk_root / f"SoccerNetGS-{args.split}" / tag / "data"
        out.mkdir(parents=True)
        for s in seqs:
            (out / f"{s}.json").write_text(json.dumps(
                {"predictions": assemble_seq(prs[s], jz, L, args.tau, args.ratio)}))
        tags.append(tag)
    noattr = f"{tags[0]}__noattr"
    (trk_root / f"SoccerNetGS-{args.split}" / noattr).mkdir()
    (trk_root / f"SoccerNetGS-{args.split}" / noattr / "data").symlink_to(
        trk_root / f"SoccerNetGS-{args.split}" / tags[0] / "data")
    jobs = [(t, True) for t in tags] + [(noattr, False)]
    with ThreadPoolExecutor(len(jobs)) as ex:
        res = dict(zip([j[0] for j in jobs], ex.map(
            lambda j: run_eval(args.trackeval, gt_root, trk_root, args.split, j[0], seqs, j[1]),
            jobs)))
    ms = {k: float(np.median([prs[s]["ms"][k] for s in seqs])) for k in prs[seqs[0]]["ms"]}
    rows = []
    for t, L in zip(tags, args.lookahead):
        r = res[t]
        rows.append({"variant": t, "lookahead_frames": L, "gs_hota": r["HOTA"], "deta": r["DetA"],
                     "assa": r["AssA"], "tau": args.tau, "n_seqs": len(seqs), **ms})
        print(f"{t:30} GS-HOTA={r['HOTA']:6.2f} DetA={r['DetA']:6.2f} AssA={r['AssA']:6.2f}", flush=True)
    print(f"{'(no attributes, pitch HOTA)':30} HOTA={res[noattr]['HOTA']:6.2f}   stage ms: {ms}")
    dest = Path(f"results/gsr_e2e_{args.split}.json")
    prev = json.loads(dest.read_text()) if dest.exists() else []
    merged = {r["variant"]: r for r in prev}
    merged.update({r["variant"]: r for r in rows})
    merged[f"{name}__pitch_hota_no_attrs"] = {"variant": f"{name}__pitch_hota_no_attrs",
                                              "hota": res[noattr]["HOTA"]}
    dest.write_text(json.dumps(list(merged.values()), indent=1))
    print(f"[e2e] wrote {dest}")


if __name__ == "__main__":
    main()
