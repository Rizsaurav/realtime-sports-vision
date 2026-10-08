#!/usr/bin/env python3
"""Oracle swaps on end-to-end predictions: how many GS-HOTA points would a
perfect version of one attribute add, given our actual detections/tracks?

Each predicted box is matched to a ground-truth box in the same frame by image
IoU >= 0.5; the chosen attribute(s) are replaced by the matched GT value
(unmatched boxes keep the prediction). Scored with the official GS-HOTA.

Usage:
  python scripts/gsr_oracle_swap.py --pred results/_work/gsr_e2e/e2e_rw25/SoccerNetGS-valid/e2e_rw25_L0/data \
      --gt data/raw/soccernet/gsr --split valid --trackeval /data/saurav/repos/sn-trackeval
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SWAPS = {"pred": (), "oracle_role": ("role",), "oracle_team": ("team",),
         "oracle_jersey": ("jersey",), "oracle_team+jersey": ("team", "jersey"),
         "oracle_all_attrs": ("role", "team", "jersey"),
         "oracle_pitch": ("pitch",), "oracle_pitch+attrs": ("pitch", "role", "team", "jersey")}


def match(preds, gts):
    """Greedy IoU matching per frame -> {pred_index: gt_annotation}."""
    out = {}
    if not preds or not gts:
        return out
    P = np.array([[p["bbox_image"]["x"], p["bbox_image"]["y"], p["bbox_image"]["w"], p["bbox_image"]["h"]]
                  for _, p in preds])
    G = np.array([[g["bbox_image"]["x"], g["bbox_image"]["y"], g["bbox_image"]["w"], g["bbox_image"]["h"]]
                  for g in gts])
    x1 = np.maximum(P[:, None, 0], G[None, :, 0]); y1 = np.maximum(P[:, None, 1], G[None, :, 1])
    x2 = np.minimum(P[:, None, 0] + P[:, None, 2], G[None, :, 0] + G[None, :, 2])
    y2 = np.minimum(P[:, None, 1] + P[:, None, 3], G[None, :, 1] + G[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    iou = inter / (P[:, None, 2] * P[:, None, 3] + G[None, :, 2] * G[None, :, 3] - inter + 1e-9)
    used = set()
    for pi, gi in sorted(zip(*np.where(iou >= 0.5)), key=lambda t: -iou[t[0], t[1]]):
        if pi in out or gi in used:
            continue
        out[pi] = gts[gi]
        used.add(gi)
    return {preds[pi][0]: g for pi, g in out.items()}


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
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--trackeval", required=True)
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    pred_dir, gt_root = Path(args.pred), Path(args.gt).resolve()
    seqs = sorted(p.stem for p in pred_dir.glob("*.json"))
    trk_root = Path("results/_work/gsr_oracle_swap").resolve()
    shutil.rmtree(trk_root, ignore_errors=True)
    outs = {k: trk_root / f"SoccerNetGS-{args.split}" / k / "data" for k in SWAPS}
    for d in outs.values():
        d.mkdir(parents=True)
    for s in seqs:
        preds = json.loads((pred_dir / f"{s}.json").read_text())["predictions"]
        g = json.loads((gt_root / args.split / s / "Labels-GameState.json").read_text())
        gb, pb = {}, {}
        for a in g["annotations"]:
            if a.get("supercategory") == "object" and a["attributes"]["role"] != "ball":
                gb.setdefault(a["image_id"], []).append(a)
        for i, p in enumerate(preds):
            pb.setdefault(p["image_id"], []).append((i, p))
        m = {}
        for iid, ps in pb.items():
            m.update(match(ps, gb.get(iid, [])))
        for k, fields in SWAPS.items():
            new = []
            for i, p in enumerate(preds):
                q = dict(p, attributes=dict(p["attributes"]))
                ga = m.get(i)
                if ga is not None:
                    for f in fields:
                        if f == "pitch":
                            if ga.get("bbox_pitch"):
                                q["bbox_pitch"] = dict(ga["bbox_pitch"])
                        else:
                            q["attributes"][f] = ga["attributes"][f]
                new.append(q)
            (outs[k] / f"{s}.json").write_text(json.dumps({"predictions": new}))
    with ThreadPoolExecutor(len(SWAPS)) as ex:
        res = dict(zip(SWAPS, ex.map(lambda k: run_eval(args.trackeval, gt_root, trk_root,
                                                        args.split, k, seqs), SWAPS)))
    rows = []
    for k in SWAPS:
        rows.append({"variant": k, "gs_hota": res[k]["HOTA"], "deta": res[k]["DetA"], "assa": res[k]["AssA"]})
        print(f"{k:20} GS-HOTA={res[k]['HOTA']:6.2f} DetA={res[k]['DetA']:6.2f} AssA={res[k]['AssA']:6.2f}",
              flush=True)
    dest = Path(f"results/gsr_oracle_swap_{args.name or pred_dir.parent.name}_{args.split}.json")
    dest.write_text(json.dumps(rows, indent=1))
    shutil.rmtree(trk_root, ignore_errors=True)
    print(f"[oracle-swap] wrote {dest}")


if __name__ == "__main__":
    main()
