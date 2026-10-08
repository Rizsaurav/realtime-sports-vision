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


def roles_online(rows, L, window=0):
    """-> {row_index: role}: majority detector class of the track over frames
    (t - window, t + L]; window=0 = the whole track history (up to t + L).
    A finite window stops an identity hand-over (e.g. referee -> player)
    from dragging the old role along."""
    by_track = {}
    for i, r in enumerate(rows):
        by_track.setdefault(r["track"], []).append(i)
    out = {}
    for tr, idx in by_track.items():
        idx.sort(key=lambda i: rows[i]["frame"])
        frames = [rows[i]["frame"] for i in idx]
        cnt, ptr, lo = Counter(), 0, 0
        for j, i in enumerate(idx):
            horizon = frames[-1] if L < 0 else frames[j] + L
            while ptr < len(idx) and frames[ptr] <= horizon:
                cnt[rows[idx[ptr]]["cls"]] += 1
                ptr += 1
            if window:
                while lo < ptr and frames[lo] <= frames[j] - window:
                    cnt[rows[idx[lo]]["cls"]] -= 1
                    lo += 1
            out[i] = (cnt.most_common(1)[0][0] if sum(cnt.values()) > 0
                      else rows[i]["cls"])
    return out


def stitch_tracks(rows, max_gap=75, v_max=0.4, d0=2.0, feat_thr=3.0):
    """Online re-linking of broken tracks in pitch space (causal).

    When a track id first appears, link it to the best recently lost track
    (not visible in that frame) that is reachable at running speed
    (dist <= d0 + v_max * gap metres, gap <= max_gap frames), in the same role
    group, with a similar kit colour. Returns {row_index: canonical_track_id}.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i]["frame"])
    canon, last, first_seen = {}, {}, set()
    group = lambda c: "ref" if c == "referee" else "pl"
    by_frame = {}
    for i in order:
        by_frame.setdefault(rows[i]["frame"], []).append(i)
    out = {}
    for f in sorted(by_frame):
        visible = {canon.get(rows[i]["track"], rows[i]["track"]) for i in by_frame[f]
                   if rows[i]["track"] in first_seen}
        for i in by_frame[f]:
            r = rows[i]
            t = r["track"]
            if t not in first_seen:
                first_seen.add(t)
                best, bc = None, 1e9
                if r["pitch"] is not None:
                    x, y = r["pitch"][2], r["pitch"][3]
                    for c, (lf, lx, ly, lg, lfeat) in last.items():
                        gap = f - lf
                        if c in visible or gap <= 0 or gap > max_gap or lg != group(r["cls"]):
                            continue
                        dist = ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
                        if dist > d0 + v_max * gap:
                            continue
                        fd = (np.linalg.norm((np.asarray(r["feat"]) - lfeat) / (np.abs(lfeat) + 10))
                              if r["feat"] is not None and lfeat is not None else 0.0)
                        if fd > feat_thr:
                            continue
                        cost = dist / (d0 + v_max * gap) + fd / feat_thr + gap / max_gap
                        if cost < bc:
                            best, bc = c, cost
                if best is not None:
                    canon[t] = best
                    visible.add(best)
            out[i] = canon.get(t, t)
        # the tracker can revive a lost id that was already handed to a new
        # fragment: never emit one id twice in a frame - the original keeps it
        used = {}
        for i in sorted(by_frame[f], key=lambda i: rows[i]["track"] != out[i]):
            c = out[i]
            if c in used:
                t = rows[i]["track"]
                canon[t] = t
                c = t if t not in used else -(10 ** 6 + i)
                out[i] = c
            used[c] = i
        for i in by_frame[f]:
            r, c = rows[i], out[i]
            if r["pitch"] is not None:
                prev = last.get(c)
                feat = np.asarray(r["feat"]) if r["feat"] is not None else None
                if prev is not None and prev[4] is not None and feat is not None:
                    feat = 0.9 * prev[4] + 0.1 * feat
                elif prev is not None and feat is None:
                    feat = prev[4]
                last[c] = (f, r["pitch"][2], r["pitch"][3], group(r["cls"]), feat)
    return out


def assemble_seq(pr, jz, L, tau, ratio, role_window=0, stitch=None):
    rows = [dict(r) for r in pr["rows"] if r["cls"] != "ball"]
    if stitch is not None:
        cid = stitch_tracks(rows, **stitch)
        for i, r in enumerate(rows):
            r["orig_track"] = r["track"]
            r["track"] = cid[i]
    role = roles_online(rows, L, role_window)
    team_rows = [(i, r["frame"], r["track"], role[i], r["pitch"][2] if r["pitch"] else 0.0,
                  np.asarray(r["feat"]))
                 for i, r in enumerate(rows)
                 if role[i] in ("player", "goalkeeper") and r["feat"] is not None]
    team = predict_teams_refit(team_rows, L, warmup=25)
    jc = {}
    if jz is not None:
        m = jz["seq"] == pr["seq"]
        if m.any():
            meta = {k: jz[k][m] for k in ("seq", "frame", "track", "ann_id")}
            if stitch is not None:                         # jersey evidence follows the stitched id
                remap = {r["orig_track"]: r["track"] for r in rows}
                meta["track"] = np.asarray([remap.get(int(t), int(t)) for t in meta["track"]])
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
    ap.add_argument("--stitch", action="store_true", help="online pitch-space track stitching")
    ap.add_argument("--stitch-gap", type=int, default=75)
    ap.add_argument("--stitch-vmax", type=float, default=0.4, help="metres per frame")
    ap.add_argument("--stitch-d0", type=float, default=2.0)
    ap.add_argument("--stitch-feat", type=float, default=3.0)
    ap.add_argument("--role-window", type=int, default=0,
                    help="frames of history for the role vote (0 = whole track)")
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
                {"predictions": assemble_seq(
                    prs[s], jz, L, args.tau, args.ratio, args.role_window,
                    dict(max_gap=args.stitch_gap, v_max=args.stitch_vmax, d0=args.stitch_d0,
                         feat_thr=args.stitch_feat) if args.stitch else None)}))
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
