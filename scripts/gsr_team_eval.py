#!/usr/bin/env python3
"""Team-only GS-HOTA: online team assignment under a bounded output delay.

Everything except the team attribute is ground truth (boxes, IDs, pitch
positions, roles, jerseys). Team is predicted causally:
  1. torso colour feature per player box (Lab mean/std of non-grass pixels)
  2. 2-way k-means fitted on the first `--warmup` frames' outfield players
  3. each box -> nearest cluster; each track's label = majority vote over its
     boxes seen up to frame t + L (L = allowed lookahead / output delay)
  4. cluster -> "left"/"right" side by the running mean pitch x of its players
  5. goalkeepers take the side of the half they stand in
L = -1 means unlimited lookahead (offline upper bound).

Usage:
  python scripts/gsr_team_eval.py --gt data/raw/soccernet/gsr --split valid \
      --trackeval /data/saurav/repos/sn-trackeval --lookahead 0 25 125 -1
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def torso_feature(frame, b):
    x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
    x1, x2 = max(0, x + w // 4), min(frame.shape[1], x + 3 * w // 4)
    y1, y2 = max(0, y + int(h * 0.15)), min(frame.shape[0], y + int(h * 0.55))
    crop = frame[y1:y2, x1:x2]
    if crop.size < 12:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 1] > 60)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    keep = lab[~grass] if (~grass).sum() >= 6 else lab
    return np.concatenate([keep.mean(0), keep.std(0)])


def extract(job):
    """Per sequence: torso features for every player/goalkeeper annotation."""
    seq_dir = Path(job)
    cv2.setNumThreads(1)
    d = json.loads((seq_dir / "Labels-GameState.json").read_text())
    by_img = {}
    for a in d["annotations"]:
        if a.get("supercategory") == "object" and a["attributes"]["role"] in ("player", "goalkeeper"):
            by_img.setdefault(a["image_id"], []).append(a)
    rows = []
    for fi, im in enumerate(d["images"]):
        anns = by_img.get(im["image_id"], [])
        if not anns:
            continue
        frame = cv2.imread(str(seq_dir / "img1" / im["file_name"]))
        for a in anns:
            f = torso_feature(frame, a["bbox_image"])
            if f is not None:
                rows.append((a["id"], fi, a["track_id"], a["attributes"]["role"],
                             a["bbox_pitch"]["x_bottom_middle"] if a.get("bbox_pitch") else 0.0, f))
    return seq_dir.name, rows


def kmeans2(X, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), 2, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(-1), 1)
        new = np.stack([X[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(2)])
        if np.allclose(new, c):
            break
        c = new
    return c


def predict_teams(rows, L, warmup):
    """Return {annotation_id: 'left'|'right'|None} under lookahead L (frames, -1 = all)."""
    out = {}
    fields = [r for r in rows if r[3] == "player"]
    if not fields:
        return out
    t_fit = max(r[1] for r in fields) if L < 0 else min(r[1] for r in fields) + warmup
    X = np.array([r[5] for r in fields if r[1] <= t_fit])
    if len(X) < 4:
        return out
    mu, sd = X.mean(0), X.std(0) + 1e-6
    C = kmeans2((X - mu) / sd)
    assign = {}
    for r in rows:
        if r[3] == "player":
            assign[r[0]] = int(np.argmin((((r[5] - mu) / sd - C) ** 2).sum(-1)))
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r[1], []).append(r)
    frames = sorted(by_frame)
    last = frames[-1]
    votes, xs = {}, {0: [], 1: []}
    ptr = 0
    for t in frames:
        horizon = last if L < 0 else t + L
        while ptr < len(frames) and frames[ptr] <= horizon:
            for r in by_frame[frames[ptr]]:
                if r[3] == "player":
                    v = votes.setdefault(r[2], [0, 0])
                    v[assign[r[0]]] += 1
                    xs[assign[r[0]]].append(r[4])
            ptr += 1
        if t < (min(frames) + warmup if L >= 0 else 0) and L >= 0 and t_fit > t + max(L, 0):
            continue                                   # clusters not fitted yet at output time
        mx = [np.mean(xs[k]) if xs[k] else 0.0 for k in (0, 1)]
        left_cluster = int(np.argmin(mx))
        for r in by_frame[t]:
            if r[3] == "player":
                v = votes.get(r[2])
                k = int(np.argmax(v)) if v else assign[r[0]]
                out[r[0]] = "left" if k == left_cluster else "right"
            else:                                     # goalkeeper: side of the half
                out[r[0]] = "left" if r[4] < 0 else "right"
    return out


def kmeans2_init(X, c, iters=10):
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(-1), 1)
        new = np.stack([X[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(2)])
        if np.allclose(new, c):
            break
        c = new
    return c


def predict_teams_refit(rows, L, warmup, refit_every=25, max_fit=4000, seed=0):
    """Online clusters re-fitted every `refit_every` frames on everything seen so far
    (warm-started from the previous centroids so labels stay stable); each track is
    classified from its mean colour feature up to the output horizon."""
    out = {}
    if not any(r[3] == "player" for r in rows):
        return out
    rng = np.random.default_rng(seed)
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r[1], []).append(r)
    frames = sorted(by_frame)
    first, last = frames[0], frames[-1]
    seen, tsum, tcnt, tx = [], {}, {}, {}
    C = mu = sd = None
    ptr = 0
    last_fit = None
    for t in frames:
        horizon = last if L < 0 else t + L
        while ptr < len(frames) and frames[ptr] <= horizon:
            for r in by_frame[frames[ptr]]:
                if r[3] == "player":
                    seen.append(r[5])
                    tsum[r[2]] = tsum.get(r[2], 0) + r[5]
                    tcnt[r[2]] = tcnt.get(r[2], 0) + 1
                    tx.setdefault(r[2], []).append(r[4])
            ptr += 1
        if horizon < first + warmup or len(seen) < 4:
            continue
        if C is None or t - last_fit >= refit_every:
            X = np.asarray(seen)
            if len(X) > max_fit:
                X = X[rng.choice(len(X), max_fit, replace=False)]
            mu, sd = X.mean(0), X.std(0) + 1e-6
            Z = (X - mu) / sd
            C = kmeans2(Z, seed=seed) if C is None else kmeans2_init(Z, C)
            last_fit = t
        lab = {tid: int(np.argmin((((tsum[tid] / tcnt[tid] - mu) / sd - C) ** 2).sum(-1)))
               for tid in tsum}
        mx = [np.mean([x for tid, xs in tx.items() if lab[tid] == k for x in xs] or [0.0])
              for k in (0, 1)]
        left_cluster = int(np.argmin(mx))
        for r in by_frame[t]:
            if r[3] == "player":
                out[r[0]] = "left" if lab[r[2]] == left_cluster else "right"
            else:
                out[r[0]] = "left" if r[4] < 0 else "right"
    return out


def run_eval(trackeval_dir, gt_root, trk_root, split, tracker, seqs):
    cmd = [sys.executable, str(Path(trackeval_dir) / "scripts" / "run_soccernet_gs.py"),
           "--GT_FOLDER", str(gt_root), "--TRACKERS_FOLDER", str(trk_root),
           "--SPLIT_TO_EVAL", split, "--TRACKERS_TO_EVAL", tracker,
           "--METRICS", "HOTA", "--PRINT_CONFIG", "False", "--USE_PARALLEL", "False",
           "--SEQ_INFO", *seqs]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    summ = trk_root / f"SoccerNetGS-{split}" / tracker / "cls_comb_det_av_summary.txt"
    head, vals = summ.read_text().splitlines()[:2]
    return dict(zip(head.split(), map(float, vals.split())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--trackeval", required=True)
    ap.add_argument("--lookahead", type=int, nargs="+", default=[0, 25, 125, -1])
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--method", default="refit", choices=["warmup", "refit"])
    ap.add_argument("--feat-cache", default="")
    args = ap.parse_args()

    gt_root = Path(args.gt).resolve()
    seq_dirs = sorted(p for p in (gt_root / args.split).iterdir()
                      if (p / "Labels-GameState.json").exists())[: args.max_seqs or None]
    seqs = [p.name for p in seq_dirs]
    import pickle
    fc = Path(args.feat_cache) if args.feat_cache else None
    if fc and fc.exists():
        feats = pickle.loads(fc.read_bytes())
    else:
        with Pool(args.workers) as pool:
            feats = dict(pool.map(extract, [str(p) for p in seq_dirs]))
        if fc:
            fc.write_bytes(pickle.dumps(feats))
    print(f"[team] features for {len(feats)} seqs", flush=True)

    trk_root = Path("results/_work/gsr_team").resolve()
    shutil.rmtree(trk_root, ignore_errors=True)
    tags, acc = [], {}
    for L in args.lookahead:
        tag = f"team_{args.method}_L{'inf' if L < 0 else L}_w{args.warmup}"
        out = trk_root / f"SoccerNetGS-{args.split}" / tag / "data"
        out.mkdir(parents=True)
        right = total = 0
        for p in seq_dirs:
            d = json.loads((p / "Labels-GameState.json").read_text())
            fn = predict_teams_refit if args.method == "refit" else predict_teams
            pred = fn(feats[p.name], L, args.warmup)
            anns = []
            for a in d["annotations"]:
                a = dict(a, attributes=dict(a.get("attributes", {})))
                if a.get("supercategory") == "object" and a["attributes"]["role"] in ("player", "goalkeeper"):
                    t = pred.get(a["id"])
                    total += 1
                    right += t == a["attributes"]["team"]
                    a["attributes"]["team"] = t
                anns.append(a)
            (out / f"{p.name}.json").write_text(json.dumps({"predictions": anns}))
        acc[tag] = (round(100 * right / max(total, 1), 2), L)
        tags.append(tag)

    with ThreadPoolExecutor(len(tags)) as ex:
        res = dict(zip(tags, ex.map(lambda t: run_eval(args.trackeval, gt_root, trk_root,
                                                       args.split, t, seqs), tags)))
    rows = []
    for t in tags:
        rows.append({"variant": t, "method": args.method, "lookahead_frames": acc[t][1],
                     "warmup": args.warmup,
                     "box_team_accuracy_pct": acc[t][0], "gs_hota_team_only": res[t]["HOTA"]})
        print(f"{t:24} box team acc={acc[t][0]:6.2f}%  GS-HOTA(team only)={res[t]['HOTA']:6.2f}",
              flush=True)
    dest = Path(f"results/gsr_team_{args.split}.json")
    prev = json.loads(dest.read_text()) if dest.exists() else []
    merged = {r["variant"]: r for r in prev}
    merged.update({r["variant"]: r for r in rows})
    dest.write_text(json.dumps(list(merged.values()), indent=1))
    shutil.rmtree(trk_root, ignore_errors=True)
    print(f"[team] wrote {dest}")


if __name__ == "__main__":
    main()
