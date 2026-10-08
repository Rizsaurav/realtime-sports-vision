#!/usr/bin/env python3
"""Score every player crop of a SoccerNet-GSR split with a legibility classifier
and a jersey-number PARSeq reader; save per-crop number distributions.

Runs in the separate jersey env (PARSeq's strhub pins pytorch-lightning<2).
Weights: Koshkina & Elder, "A General Framework for Jersey Number Recognition
in Sports Video" (CVPRW 2024), SoccerNet legibility ResNet34 + SoccerNet
fine-tuned PARSeq. Their license is non-commercial: research measurement only.

Per crop: leg = P(number legible) on the whole-player crop; P[n] (n=0..99) from
PARSeq's first three token distributions over {end, 0..9} on the torso crop:
  P(d) = p0[d] * p1[end],   P(ab) = p0[a] * p1[b] * p2[end]   (renormalised)

Usage (jersey env):
  python scripts/score_jersey_parseq.py --gt data/raw/soccernet/gsr --split valid \
      --repo /data/saurav/repos/jersey-number-pipeline --out results/jersey_scores_valid.npz
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_models(repo, device):
    sys.path.insert(0, repo)
    sys.path.insert(0, f"{repo}/str/parseq")
    from networks import LegibilityClassifier34
    from strhub.models.utils import load_from_checkpoint
    leg = LegibilityClassifier34()
    sd = torch.load(next(Path(repo, "experiments").glob("legibility_resnet34_soccer*.pth")),
                    map_location=device)
    leg.load_state_dict(sd)
    leg.to(device).eval()
    ckpt = next(Path(repo, "models").glob("parseq_*.ckpt"))
    stm = load_from_checkpoint(str(ckpt)).eval().to(device)
    return leg, stm


def crops_for_frame(frame, boxes):
    """-> list of (whole_player_rgb, torso_rgb) per box (full resolution)."""
    H, W = frame.shape[:2]
    out = []
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        x1, x2 = int(max(0, x)), int(min(W, x + w))
        y1, y2 = int(max(0, y)), int(min(H, y + h))
        ty1, ty2 = int(max(0, y + 0.10 * h)), int(min(H, y + 0.55 * h))
        whole, torso = frame[y1:y2, x1:x2], frame[ty1:ty2, x1:x2]
        if whole.size == 0 or torso.size == 0:
            out.append(None)
            continue
        out.append((cv2.cvtColor(whole, cv2.COLOR_BGR2RGB), cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)))
    return out


def batch_tensor(imgs, hw, device, norm):
    t = torch.stack([torch.from_numpy(cv2.resize(i, (hw[1], hw[0]), interpolation=cv2.INTER_CUBIC))
                     for i in imgs]).permute(0, 3, 1, 2).float().div(255).to(device)
    return (t - MEAN.to(device)) / STD.to(device) if norm == "imagenet" else (t - 0.5) / 0.5


def number_dist(logits):
    p = logits[:, :3, :11].float().softmax(-1).cpu().numpy()     # (N,3,{end,0..9})
    end, dig = 0, slice(1, 11)
    out = np.zeros((len(p), 100), np.float32)
    out[:, :10] = p[:, 0, dig] * p[:, 1, end:end + 1]
    two = p[:, 0, dig][:, :, None] * p[:, 1, dig][:, None, :] * p[:, 2, end][:, None, None]
    out[:, 10:] = two.reshape(len(p), 100)[:, 10:]               # leading zero not a number
    s = out.sum(1, keepdims=True)
    return out / np.maximum(s, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--repo", default="/data/saurav/repos/jersey-number-pipeline")
    ap.add_argument("--out", default="results/jersey_scores_valid.npz")
    ap.add_argument("--min-h", type=int, default=40)
    ap.add_argument("--leg-thr", type=float, default=0.3, help="run PARSeq only above this")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--perception", default="",
                    help="score boxes from a gsr_perceive.py output dir instead of ground truth")
    args = ap.parse_args()

    leg_model, stm = load_models(args.repo, args.device)
    img_hw = tuple(stm.hparams.img_size)
    seqs = sorted(p for p in (Path(args.gt) / args.split).iterdir()
                  if (p / "Labels-GameState.json").exists())[: args.max_seqs or None]
    rows = {k: [] for k in ("seq", "frame", "track", "ann_id", "leg")}
    Ps = []
    pool = ThreadPoolExecutor(16)
    for s in seqs:
        d = json.loads((s / "Labels-GameState.json").read_text())
        by_img = {}
        if args.perception:
            pr = json.loads((Path(args.perception) / f"{s.name}.json").read_text())
            for i, r in enumerate(pr["rows"]):
                x, y, w, h = r["bbox"]
                if r["cls"] in ("player", "goalkeeper") and h >= args.min_h:
                    by_img.setdefault(r["image_id"], []).append(
                        {"id": i, "track_id": r["track"],
                         "bbox_image": {"x": x, "y": y, "w": w, "h": h}})
        else:
            for a in d["annotations"]:
                if (a.get("supercategory") == "object" and a["attributes"]["role"] in ("player", "goalkeeper")
                        and a["bbox_image"]["h"] >= args.min_h):
                    by_img.setdefault(a["image_id"], []).append(a)
        ims = [(fi, im) for fi, im in enumerate(d["images"]) if im["image_id"] in by_img]
        frames = pool.map(lambda x: cv2.imread(str(s / "img1" / x[1]["file_name"])), ims)
        meta, whole, torso = [], [], []
        for (fi, im), fr in zip(ims, frames):
            anns = by_img[im["image_id"]]
            for a, c in zip(anns, crops_for_frame(fr, [a["bbox_image"] for a in anns])):
                if c is None:
                    continue
                meta.append((s.name, fi, a["track_id"], a["id"]))
                whole.append(c[0])
                torso.append(c[1])
        legs, P = np.zeros(len(meta), np.float32), np.zeros((len(meta), 100), np.float32)
        with torch.no_grad():
            for i in range(0, len(meta), 512):
                legs[i:i + 512] = leg_model(batch_tensor(whole[i:i + 512], (256, 256), args.device,
                                                         "imagenet")).view(-1).cpu().numpy()
            run = np.where(legs >= args.leg_thr)[0]
            for i in range(0, len(run), 512):
                idx = run[i:i + 512]
                x = batch_tensor([torso[j] for j in idx], img_hw, args.device, "half")
                P[idx] = number_dist(stm(x))
        for (sq, fi, tr, aid), l in zip(meta, legs):
            rows["seq"].append(sq); rows["frame"].append(fi); rows["track"].append(tr)
            rows["ann_id"].append(aid); rows["leg"].append(l)
        Ps.append(P)
        print(f"  {s.name}: {len(meta)} crops, {len(run)} legible>= {args.leg_thr}", flush=True)
    np.savez(args.out, P=np.concatenate(Ps), **{k: np.asarray(v) for k, v in rows.items()})
    print(f"[jersey] wrote {args.out}")


if __name__ == "__main__":
    main()
