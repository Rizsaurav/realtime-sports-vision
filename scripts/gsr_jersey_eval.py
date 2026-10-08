#!/usr/bin/env python3
"""Jersey-only GS-HOTA under a bounded output delay.

Everything except jersey is ground truth. For every player/goalkeeper crop the
classifier gives P(number) for 0-99 and P(none). Each track accumulates
legibility-weighted evidence  s_n += (1 - P(none)) * P(n)  over crops seen up to
frame t + L, and commits a number once  max_n s_n >= tau  and the leader holds
>= `ratio` of the evidence mass. Until it commits the track reports null.

Usage:
  python scripts/gsr_jersey_eval.py --crops data/processed/jersey/valid.npz \
      --model runs/jersey/resnet18_64.pt --gt data/raw/soccernet/gsr --split valid \
      --trackeval /data/saurav/repos/sn-trackeval --lookahead 0 25 125 250 -1
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_jersey import JerseyNet, to_tensor


def number_probs(p1, p2):
    """(N,11),(N,11) -> (N,100) P(number), (N,) P(none)."""
    n = np.zeros((len(p1), 100), np.float32)
    n[:, :10] = p1[:, 10:11] * p2[:, :10]
    for t in range(1, 10):
        n[:, t * 10:(t + 1) * 10] = p1[:, t:t + 1] * p2[:, :10]
    return n, p1[:, 10] * p2[:, 10]


def predict(meta, P, Pnone, L, tau, ratio):
    """-> {seq: {track_id: (commit_frame, number)}}: the frame from which the
    online system outputs the number for that track (null before it)."""
    out = {}
    order = np.lexsort((meta["frame"], meta["seq"]))
    by_seq = {}
    for i in order:
        by_seq.setdefault(meta["seq"][i], []).append(i)
    for seq, idx in by_seq.items():
        idx = np.asarray(idx)
        frames = meta["frame"][idx]
        last = frames.max()
        ev, committed = {}, {}
        ptr = 0
        for t in np.unique(frames):
            horizon = last if L < 0 else t + L
            while ptr < len(idx) and frames[ptr] <= horizon:
                i = idx[ptr]
                tr = int(meta["track"][i])
                ev[tr] = ev.get(tr, 0) + (1 - Pnone[i]) * P[i]
                ptr += 1
            for tr, s in ev.items():
                if tr not in committed:
                    k = int(s.argmax())
                    if s[k] >= tau and s[k] >= ratio * s.sum():
                        committed[tr] = (0 if L < 0 else int(t), str(k))
        out[seq] = committed
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
    ap.add_argument("--crops", default="data/processed/jersey/valid.npz")
    ap.add_argument("--model", default="runs/jersey/resnet18_64.pt")
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--trackeval", required=True)
    ap.add_argument("--lookahead", type=int, nargs="+", default=[0, 25, 125, 250, -1])
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probs", default="",
                    help="precomputed per-crop scores (score_jersey_parseq.py): P[N,100], leg[N]")
    ap.add_argument("--name", default="r18")
    args = ap.parse_args()

    if args.probs:
        z = np.load(args.probs)
        meta = {k: z[k] for k in ("seq", "frame", "track", "ann_id")}
        P, Pnone = z["P"], 1.0 - z["leg"]
        print(f"[jersey] loaded {len(P)} scored crops", flush=True)
    else:
        P, Pnone, meta = score_with_classifier(args)
    run_curve(args, P, Pnone, meta)


def score_with_classifier(args):
    z = np.load(args.crops)
    meta = {k: z[k] for k in ("seq", "frame", "track", "ann_id")}
    net = JerseyNet(pretrained=False).to(args.device)
    net.load_state_dict(torch.load(args.model, map_location=args.device))
    net.eval()
    p1s, p2s = [], []
    with torch.no_grad():
        for i in range(0, len(z["X"]), 4096):
            o1, o2 = net(to_tensor(z["X"][i:i + 4096], args.device))
            p1s.append(o1.softmax(1).cpu().numpy())
            p2s.append(o2.softmax(1).cpu().numpy())
    P, Pnone = number_probs(np.concatenate(p1s), np.concatenate(p2s))
    print(f"[jersey] scored {len(P)} crops", flush=True)
    return P, Pnone, meta


def run_curve(args, P, Pnone, meta):
    gt_root = Path(args.gt).resolve()
    seq_dirs = sorted(p for p in (gt_root / args.split).iterdir()
                      if (p / "Labels-GameState.json").exists())
    seqs = [p.name for p in seq_dirs]
    gts = {p.name: json.loads((p / "Labels-GameState.json").read_text()) for p in seq_dirs}
    trk_root = Path("results/_work/gsr_jersey").resolve()
    shutil.rmtree(trk_root, ignore_errors=True)
    tags, stats = [], {}
    for L in args.lookahead:
        tag = f"jersey_{args.name}_L{'inf' if L < 0 else L}_tau{args.tau}_r{args.ratio}"
        pred = predict(meta, P, Pnone, L, args.tau, args.ratio)
        out = trk_root / f"SoccerNetGS-{args.split}" / tag / "data"
        out.mkdir(parents=True)
        right = total = 0
        for s in seqs:
            tracks = pred.get(s, {})
            fidx = {im["image_id"]: fi for fi, im in enumerate(gts[s]["images"])}
            anns = []
            for a in gts[s]["annotations"]:
                a = dict(a, attributes=dict(a.get("attributes", {})))
                if a.get("supercategory") == "object" and a["attributes"]["role"] in ("player", "goalkeeper"):
                    c = tracks.get(int(a["track_id"]))
                    j = c[1] if c and fidx[a["image_id"]] >= c[0] else None
                    total += 1
                    right += j == a["attributes"]["jersey"]
                    a["attributes"]["jersey"] = j
                anns.append(a)
            (out / f"{s}.json").write_text(json.dumps({"predictions": anns}))
        stats[tag] = (round(100 * right / max(total, 1), 2), L)
        tags.append(tag)
    with ThreadPoolExecutor(len(tags)) as ex:
        res = dict(zip(tags, ex.map(lambda t: run_eval(args.trackeval, gt_root, trk_root,
                                                       args.split, t, seqs), tags)))
    rows = []
    for t in tags:
        rows.append({"variant": t, "lookahead_frames": stats[t][1], "tau": args.tau,
                     "ratio": args.ratio, "box_jersey_accuracy_pct": stats[t][0],
                     "gs_hota_jersey_only": res[t]["HOTA"]})
        print(f"{t:34} box jersey acc={stats[t][0]:6.2f}%  GS-HOTA(jersey only)={res[t]['HOTA']:6.2f}",
              flush=True)
    dest = Path(f"results/gsr_jersey_{args.split}.json")
    prev = json.loads(dest.read_text()) if dest.exists() else []
    merged = {r["variant"]: r for r in prev}
    merged.update({r["variant"]: r for r in rows})
    dest.write_text(json.dumps(list(merged.values()), indent=1))
    shutil.rmtree(trk_root, ignore_errors=True)
    print(f"[jersey] wrote {dest}")


if __name__ == "__main__":
    main()
