#!/usr/bin/env python3
"""Supporting statistics for the paper, saved to results/paper_stats.json.

- identity fragmentation: predicted tracks per ground-truth identity and the
  share of identity breaks with no frame gap (validation, live predictions)
- jersey coverage on ground-truth boxes: share of numbered identities with any
  gated read, and with a gated read of the correct number
- decision-stage cost: CPU time of role/team/jersey decisions per frame
- provenance of training numbers (detector, jersey fine-tune)

Usage:
  python scripts/paper_stats.py
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsr_assemble import assemble_seq
from gsr_jersey_eval import gate
from gsr_oracle_swap import match

G = Path("data/raw/soccernet/gsr/valid")
E = Path("results/_work/gsr_e2e")


def fragmentation(pred_dir):
    tracks_per_id, breaks = [], defaultdict(int)
    for f in sorted(pred_dir.glob("*.json")):
        P = json.loads(f.read_text())["predictions"]
        g = json.loads((G / f.stem / "Labels-GameState.json").read_text())
        fidx = {im["image_id"]: i for i, im in enumerate(g["images"])}
        gb, pb, gt_frames = {}, {}, defaultdict(set)
        for a in g["annotations"]:
            if a.get("supercategory") == "object" and a["attributes"]["role"] != "ball":
                gb.setdefault(a["image_id"], []).append(a)
                gt_frames[a["track_id"]].add(fidx[a["image_id"]])
        for i, p in enumerate(P):
            pb.setdefault(p["image_id"], []).append((i, p))
        seq = defaultdict(list)
        for iid, ps in pb.items():
            for i, ga in match(ps, gb.get(iid, [])).items():
                seq[ga["track_id"]].append((fidx[iid], P[i]["track_id"]))
        for gid, lst in seq.items():
            lst.sort()
            tracks_per_id.append(len({t for _, t in lst}))
            for (f0, t0), (f1, t1) in zip(lst, lst[1:]):
                if t1 != t0:
                    gap = f1 - f0 - 1
                    if gap == 0:
                        breaks["no_gap"] += 1
                    else:
                        vis = sum(1 for x in range(f0 + 1, f1) if x in gt_frames[gid]) / gap
                        breaks["gap_visible" if vis > 0.5 else "gap_offscreen"] += 1
    tot = sum(breaks.values())
    return {"tracks_per_identity_mean": round(float(np.mean(tracks_per_id)), 2),
            "breaks_total": tot, "breaks_no_gap_pct": round(100 * breaks["no_gap"] / tot, 1),
            "breaks_gap_visible_pct": round(100 * breaks["gap_visible"] / tot, 1),
            "breaks_gap_offscreen_pct": round(100 * breaks["gap_offscreen"] / tot, 1)}


def jersey_coverage(npz, read_conf=0.97, leg_thr=0.3):
    z = np.load(npz)
    P, Pn = gate(z["P"], 1.0 - z["leg"], read_conf, leg_thr)
    used = (1 - Pn) > 0
    truth = {}
    for f in sorted(G.iterdir()):
        lab = f / "Labels-GameState.json"
        if lab.exists():
            for a in json.loads(lab.read_text())["annotations"]:
                if a.get("supercategory") == "object":
                    truth[(f.name, a["track_id"])] = a["attributes"]["jersey"]
    ids = defaultdict(list)
    for i, key in enumerate(zip(z["seq"].tolist(), z["track"].tolist())):
        ids[key].append(i)
    n = anyread = right = 0
    for key, idx in ids.items():
        tj = truth.get(key)
        if tj is None:
            continue
        n += 1
        g = [i for i in idx if used[i]]
        anyread += bool(g)
        right += any(str(int(P[i].argmax())) == tj for i in g)
    return {"numbered_identities": n, "with_any_gated_read_pct": round(100 * anyread / n, 1),
            "with_correct_gated_read_pct": round(100 * right / n, 1),
            "never_legible_pct": round(100 * (n - anyread) / n, 1)}


def decision_cost(perc_dir, jersey_npz, clips=5):
    jz = dict(np.load(jersey_npz))
    jz["P"], jz["Pnone"] = gate(jz["P"], 1.0 - jz["leg"], 0.97, 0.3)
    ms = []
    for f in sorted(Path(perc_dir).glob("SNGS-*.json"))[:clips]:
        pr = json.loads(f.read_text())
        frames = len({r["frame"] for r in pr["rows"]})
        t = time.perf_counter()
        assemble_seq(pr, jz, 0, 4.0, 0.6, 25, dict(max_gap=150, v_max=0.4, d0=2.0, feat_thr=3.0))
        ms.append((time.perf_counter() - t) * 1000 / frames)
    return {"clips": clips, "decision_ms_per_frame_mean": round(float(np.mean(ms)), 2),
            "note": "batch replay of the causal decision code over whole clips, single CPU thread, L=0"}


def test_without_mislabeled(clips_out=("SNGS-126", "SNGS-131", "SNGS-197")):
    """Test GS-HOTA excluding clips reported to have wrong team labels (Broadcast2Pitch, Table 7)."""
    from gsr_bootstrap import per_clip
    gt = str(Path("data/raw/soccernet/gsr").resolve())
    root = str(Path("results/_work/gsr_e2e/test_final").resolve())
    res = {"excluded": list(clips_out)}
    for L in ("L0", "L25", "L125", "L250", "Linf"):
        clips, h, full = per_clip("/data/saurav/repos/sn-trackeval", gt, root, "test", f"test_final_{L}")
        keep = {k: v for k, v in clips.items() if k not in clips_out}
        r = h.combine_sequences(keep)
        res[L] = {"all_clips": round(full, 2), "n_kept": len(keep),
                  "gs_hota": round(100 * float(np.mean(r["HOTA"])), 2)}
    return res


def main():
    out = {
        "fragmentation_baseline_bytetrack_valid_live": fragmentation(E / "st_off/SoccerNetGS-valid/st_off_L0/data"),
        "fragmentation_final_valid_live": fragmentation(E / "e2e_v5_pose/SoccerNetGS-valid/e2e_v5_pose_L0/data"),
        "jersey_coverage_valid_gt_boxes_pretrained": jersey_coverage("results/jersey_scores_valid_gt_pose_pre.npz"),
        "jersey_coverage_valid_gt_boxes_finetuned": jersey_coverage("results/jersey_scores_valid_gt_pose_ft.npz"),
        "jersey_coverage_valid_gt_boxes_finetuned_gate0.9": jersey_coverage("results/jersey_scores_valid_gt_pose_ft.npz", read_conf=0.9),
        "jersey_coverage_valid_gt_boxes_pretrained_gate0.9": jersey_coverage("results/jersey_scores_valid_gt_pose_pre.npz", read_conf=0.9),
        "test_without_mislabeled_clips": test_without_mislabeled(),
        "decision_stage": decision_cost("results/gsr_perception/s1280_kf10_mcbyte_shcmc2",
                                        "results/jersey_scores_valid_mcbyte_pose.npz"),
    }
    rc = Path("runs/finetune/yolo26s_gsr_1280/results.csv").read_text().strip().splitlines()
    head, last = rc[0].split(","), rc[-1].split(",")
    col = {h.strip(): v for h, v in zip(head, last)}
    out["detector_training"] = {"epochs_run": len(rc) - 1,
                                "holdout_mAP50": round(float(col["metrics/mAP50(B)"]), 3),
                                "holdout_mAP50_95": round(float(col["metrics/mAP50-95(B)"]), 3)}
    ft = Path("logs/jersey_ft3.log").read_text()
    out["jersey_finetune_log_excerpt"] = [l for l in ft.splitlines() if l.startswith("[ft]")]
    Path("results/paper_stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
