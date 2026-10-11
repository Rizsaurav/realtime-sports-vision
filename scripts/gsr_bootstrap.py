#!/usr/bin/env python3
"""Clip-level bootstrap confidence intervals for GS-HOTA.

Runs the official SoccerNet GS-HOTA evaluation once per prediction set to get
per-clip HOTA statistics, then resamples clips with replacement and recombines
them with TrackEval's own combine_sequences (exact pooled GS-HOTA per sample).
Paired comparisons resample the same clips for both systems.

Usage:
  python scripts/gsr_bootstrap.py --split test --trackeval /data/saurav/repos/sn-trackeval \
      --set final_L0=results/_work/gsr_e2e/test_final:test_final_L0 ... \
      --pair final_L0:final_Linf --n 2000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def per_clip(trackeval_dir, gt_root, trk_root, split, tracker):
    sys.path.insert(0, trackeval_dir)
    import trackeval
    ev_cfg = trackeval.Evaluator.get_default_eval_config()
    ev_cfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
                   "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
                   "PRINT_ONLY_COMBINED": True, "TIME_PROGRESS": False, "OUTPUT_EMPTY_CLASSES": False})
    ds_cfg = trackeval.datasets.SoccerNetGS.get_default_dataset_config()
    ds_cfg.update({"GT_FOLDER": gt_root, "TRACKERS_FOLDER": trk_root, "SPLIT_TO_EVAL": split,
                   "TRACKERS_TO_EVAL": [tracker], "PRINT_CONFIG": False})
    hota = trackeval.metrics.HOTA({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    res, _ = trackeval.Evaluator(ev_cfg).evaluate([trackeval.datasets.SoccerNetGS(ds_cfg)], [hota])
    seqs = res["SoccerNetGS"][tracker]
    clips = {}
    for seq, r in seqs.items():
        if seq in ("COMBINED_SEQ", "SUMMARIES"):
            continue
        cls = next(iter(r))                        # single evaluated class
        clips[seq] = r[cls]["HOTA"]
    full = 100 * float(np.mean(hota.combine_sequences(clips)["HOTA"]))
    return clips, hota, full


def score(hota, clips, keys):
    sub = {f"{i}_{k}": clips[k] for i, k in enumerate(keys)}
    return 100 * float(np.mean(hota.combine_sequences(sub)["HOTA"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--split", default="test")
    ap.add_argument("--trackeval", required=True)
    ap.add_argument("--set", nargs="+", required=True, help="name=<trackers_root>:<tracker>")
    ap.add_argument("--pair", nargs="*", default=[], help="nameA:nameB paired differences")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = str(Path(args.gt).resolve())
    data, hota = {}, None
    for s in args.set:
        name, spec = s.split("=", 1)
        root, tracker = spec.rsplit(":", 1)
        clips, hota, full = per_clip(args.trackeval, gt, str(Path(root).resolve()), args.split, tracker)
        data[name] = (clips, full)
        print(f"[boot] {name}: {len(clips)} clips, GS-HOTA {full:.2f}", flush=True)
    seqs = sorted(next(iter(data.values()))[0])
    rng = np.random.default_rng(0)
    samples = [list(rng.choice(seqs, len(seqs), replace=True)) for _ in range(args.n)]
    out = {"split": args.split, "n_boot": args.n, "n_clips": len(seqs), "sets": {}, "pairs": {}}
    for name, (clips, full) in data.items():
        b = np.array([score(hota, clips, k) for k in samples])
        lo, hi = np.percentile(b, [2.5, 97.5])
        out["sets"][name] = {"gs_hota": round(full, 2), "ci95": [round(lo, 2), round(hi, 2)]}
        print(f"{name:24} {full:6.2f}  95% CI [{lo:.2f}, {hi:.2f}]", flush=True)
    for p in args.pair:
        a, b_ = p.split(":")
        d = np.array([score(hota, data[b_][0], k) - score(hota, data[a][0], k) for k in samples])
        lo, hi = np.percentile(d, [2.5, 97.5])
        diff = data[b_][1] - data[a][1]
        out["pairs"][f"{b_} - {a}"] = {"diff": round(diff, 2), "ci95": [round(lo, 2), round(hi, 2)],
                                       "p_le_0": round(float((d <= 0).mean()), 4)}
        print(f"{b_} - {a}: {diff:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  P(diff<=0)={float((d <= 0).mean()):.4f}",
              flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[boot] wrote {args.out}")


if __name__ == "__main__":
    main()
