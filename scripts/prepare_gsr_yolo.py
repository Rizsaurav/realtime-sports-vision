#!/usr/bin/env python3
"""Convert SoccerNet-GSR labels to an Ultralytics detection dataset.

Classes: 0 player, 1 goalkeeper, 2 referee, 3 ball (role comes straight from
the detector, so the pipeline needs no separate role model). Frames are
symlinked. The last `--holdout` training clips become the early-stopping
split, so the official valid split stays untouched for evaluation.

Usage:
  python scripts/prepare_gsr_yolo.py --gt data/raw/soccernet/gsr \
      --out data/processed/gsr_yolo --stride 5 --holdout 5
"""
import argparse
import json
from pathlib import Path

import yaml

CLASSES = {"player": 0, "goalkeeper": 1, "referee": 2, "ball": 3}


def convert(seq_dir, img_out, lbl_out, stride):
    d = json.loads((seq_dir / "Labels-GameState.json").read_text())
    by_img = {}
    for a in d["annotations"]:
        if a.get("supercategory") == "object" and a["attributes"]["role"] in CLASSES:
            by_img.setdefault(a["image_id"], []).append(a)
    n_img = n_box = 0
    for fi, im in enumerate(d["images"]):
        if fi % stride or not im.get("has_labeled_person", True):
            continue
        W, H = im["width"], im["height"]
        lines = []
        for a in by_img.get(im["image_id"], []):
            b = a["bbox_image"]
            x1, y1 = max(0.0, b["x"]), max(0.0, b["y"])
            x2, y2 = min(W, b["x"] + b["w"]), min(H, b["y"] + b["h"])
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{CLASSES[a['attributes']['role']]} {(x1 + x2) / 2 / W:.6f} "
                         f"{(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        name = f"{seq_dir.name}_{Path(im['file_name']).stem}"
        dst = img_out / f"{name}.jpg"
        if not dst.exists():
            dst.symlink_to((seq_dir / "img1" / im["file_name"]).resolve())
        (lbl_out / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        n_box += len(lines)
    return n_img, n_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--out", default="data/processed/gsr_yolo")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--holdout", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    seqs = sorted(p for p in (Path(args.gt) / "train").iterdir()
                  if (p / "Labels-GameState.json").exists())
    splits = {"train": seqs[: -args.holdout], "val": seqs[-args.holdout:]}
    for split, lst in splits.items():
        img_out, lbl_out = out / "images" / split, out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        ni = nb = 0
        for s in lst:
            a, b = convert(s, img_out, lbl_out, args.stride if split == "train" else args.stride * 2)
            ni, nb = ni + a, nb + b
        print(f"{split}: {len(lst)} clips, {ni} images, {nb} boxes")
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out), "train": "images/train", "val": "images/val",
        "names": {v: k for k, v in CLASSES.items()}}))
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
