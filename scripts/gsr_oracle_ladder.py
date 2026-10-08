#!/usr/bin/env python3
"""GS-HOTA error budget: degrade ground truth one factor at a time.

Each variant starts from the SoccerNet-GSR labels and corrupts exactly one
component (jersey, team, role, pitch position, identity), then is scored with
the official GS-HOTA (SoccerNet sn-trackeval). The drop from 100 shows how many
points each component of a real pipeline can cost, which tells us where
engineering effort pays off.

Usage:
  python scripts/gsr_oracle_ladder.py --gt data/raw/soccernet/gsr --split valid \
      --trackeval /data/saurav/repos/sn-trackeval [--only gt,no_jersey]
Writes results/gsr_oracle_ladder_<split>.json
"""
import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def _objects(anns):
    return [a for a in anns if a.get("supercategory") == "object"]


def v_gt(anns, rng):
    return anns


def v_no_jersey(anns, rng):
    for a in _objects(anns):
        a["attributes"]["jersey"] = None
    return anns


def v_no_team(anns, rng):
    for a in _objects(anns):
        a["attributes"]["team"] = None
    return anns


def v_team_side_swapped(anns, rng):
    """Clustering found the right two teams but named the sides the wrong way round."""
    sw = {"left": "right", "right": "left"}
    for a in _objects(anns):
        t = a["attributes"]["team"]
        a["attributes"]["team"] = sw.get(t, t)
    return anns


def v_roles_all_player(anns, rng):
    for a in _objects(anns):
        if a["attributes"]["role"] in ("goalkeeper", "referee"):
            a["attributes"]["role"] = "player"
    return anns


def _shift_pitch(a, dx, dy):
    bp = a.get("bbox_pitch")
    if not bp:
        return
    for k in list(bp):
        if k.startswith("x_"):
            bp[k] += dx
        elif k.startswith("y_"):
            bp[k] += dy


def make_pitch_noise(sigma):
    def v(anns, rng):
        for a in _objects(anns):
            _shift_pitch(a, rng.gauss(0, sigma), rng.gauss(0, sigma))
        return anns
    v.__name__ = f"v_pitch_noise_{sigma}m"
    return v


def make_frame_bias(sigma):
    """Calibration error: one shared offset per frame (all players shift together)."""
    def v(anns, rng):
        off = {}
        for a in _objects(anns):
            iid = a["image_id"]
            if iid not in off:
                off[iid] = (rng.gauss(0, sigma), rng.gauss(0, sigma))
            _shift_pitch(a, *off[iid])
        return anns
    v.__name__ = f"v_frame_bias_{sigma}m"
    return v


def make_id_fragment(every):
    """Tracker breaks every track into new IDs every `every` frames."""
    def v(anns, rng):
        for a in _objects(anns):
            seg = (int(a["image_id"][-6:]) - 1) // every
            a["track_id"] = a["track_id"] * 1000 + seg
        return anns
    v.__name__ = f"v_id_fragment_{every}f"
    return v


def make_drop(p):
    """Detector misses a fraction p of boxes (uniformly at random)."""
    def v(anns, rng):
        return [a for a in anns if a.get("supercategory") != "object" or rng.random() >= p]
    v.__name__ = f"v_drop_{int(p * 100)}pct"
    return v


VARIANTS = {
    "gt": v_gt,
    "no_jersey": v_no_jersey,
    "no_team": v_no_team,
    "team_side_swapped": v_team_side_swapped,
    "roles_all_player": v_roles_all_player,
    "pitch_noise_0.5m": make_pitch_noise(0.5),
    "pitch_noise_1m": make_pitch_noise(1.0),
    "pitch_noise_2m": make_pitch_noise(2.0),
    "frame_bias_1m": make_frame_bias(1.0),
    "frame_bias_2m": make_frame_bias(2.0),
    "frame_bias_4m": make_frame_bias(4.0),
    "id_fragment_250f": make_id_fragment(250),
    "id_fragment_50f": make_id_fragment(50),
    "drop_10pct": make_drop(0.10),
}


def run_eval(trackeval_dir, gt_root, trk_root, split, tracker, use_attrs):
    # sn-trackeval's own USE_PARALLEL crashes (pickles dict_keys); parallelise across runs instead
    cmd = [sys.executable, str(Path(trackeval_dir) / "scripts" / "run_soccernet_gs.py"),
           "--GT_FOLDER", str(gt_root), "--TRACKERS_FOLDER", str(trk_root),
           "--SPLIT_TO_EVAL", split, "--TRACKERS_TO_EVAL", tracker,
           "--METRICS", "HOTA", "Identity", "--PRINT_CONFIG", "False",
           "--USE_PARALLEL", "False",
           "--USE_ROLES", str(use_attrs), "--USE_TEAMS", str(use_attrs),
           "--USE_JERSEY_NUMBERS", str(use_attrs),
           "--OUTPUT_SUMMARY", "True", "--OUTPUT_DETAILED", "False",
           "--PLOT_CURVES", "False"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # TrackLab reports cls_comb_det_av as the official GS-HOTA
    summ = trk_root / f"SoccerNetGS-{split}" / tracker / "cls_comb_det_av_summary.txt"
    head, vals = summ.read_text().splitlines()[:2]
    row = dict(zip(head.split(), map(float, vals.split())))
    return {k: row[k] for k in ("HOTA", "DetA", "AssA", "IDF1") if k in row}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--trackeval", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--cores", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gt_root = Path(args.gt).resolve()
    seqs = sorted(p.name for p in (gt_root / args.split).iterdir()
                  if (p / "Labels-GameState.json").exists())
    names = [n for n in VARIANTS if not args.only or n in args.only.split(",")]
    trk_root = Path("results/_work/gsr_ladder").resolve()
    shutil.rmtree(trk_root, ignore_errors=True)

    gts = {s: json.loads((gt_root / args.split / s / "Labels-GameState.json").read_text())
           for s in seqs}
    for n in names:
        rng = random.Random(args.seed)
        base = trk_root / f"SoccerNetGS-{args.split}"
        out = base / n / "data"
        out.mkdir(parents=True)
        for s in seqs:
            preds = VARIANTS[n](copy.deepcopy(gts[s]["annotations"]), rng)
            (out / f"{s}.json").write_text(json.dumps({"predictions": preds}))
        (base / f"{n}__noattr").mkdir()
        (base / f"{n}__noattr" / "data").symlink_to(out)

    jobs = [(n, attrs) for n in names for attrs in (True, False)]
    with ThreadPoolExecutor(args.cores) as ex:
        res = dict(zip(jobs, ex.map(
            lambda j: run_eval(args.trackeval, gt_root, trk_root, args.split,
                               j[0] if j[1] else f"{j[0]}__noattr", j[1]), jobs)))
    rows = []
    for n in names:
        gs, loc = res[(n, True)], res[(n, False)]
        rows.append({"variant": n, "gs_hota": gs["HOTA"], "gs_deta": gs["DetA"],
                     "gs_assa": gs["AssA"], "hota_pitch_no_attrs": loc["HOTA"]})
        print(f"{n:20} GS-HOTA={gs['HOTA']:6.2f} DetA={gs['DetA']:6.2f} "
              f"AssA={gs['AssA']:6.2f} | pitch-HOTA (no attrs)={loc['HOTA']:6.2f}", flush=True)

    dest = Path(f"results/gsr_oracle_ladder_{args.split}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=1))
    shutil.rmtree(trk_root, ignore_errors=True)
    print(f"[ladder] wrote {dest}")


if __name__ == "__main__":
    main()
