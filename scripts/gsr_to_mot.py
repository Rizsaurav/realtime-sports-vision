#!/usr/bin/env python3
"""Mirror a SoccerNet-GSR split in MOT layout (people only, no ball) so the
image-space tracker tooling (dump_detections.py, sweep_trackers.py) runs on it.

  <out>/<split>/<seq>/{img1 -> original frames, gt/gt.txt, seqinfo.ini}

Usage:
  python scripts/gsr_to_mot.py --gt data/raw/soccernet/gsr --split valid --out data/processed/gsr_mot
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", default="data/processed/gsr_mot")
    args = ap.parse_args()
    n = 0
    for s in sorted(Path(args.gt, args.split).iterdir()):
        lab = s / "Labels-GameState.json"
        if not lab.exists():
            continue
        d = json.loads(lab.read_text())
        dst = Path(args.out, args.split, s.name)
        (dst / "gt").mkdir(parents=True, exist_ok=True)
        if not (dst / "img1").exists():
            (dst / "img1").symlink_to((s / "img1").resolve())
        fidx = {im["image_id"]: fi + 1 for fi, im in enumerate(d["images"])}
        lines = []
        for a in d["annotations"]:
            if a.get("supercategory") != "object" or a["attributes"]["role"] == "ball":
                continue
            b = a["bbox_image"]
            lines.append(f"{fidx[a['image_id']]},{a['track_id']},{b['x']:.2f},{b['y']:.2f},"
                         f"{b['w']:.2f},{b['h']:.2f},1,1,1")
        (dst / "gt" / "gt.txt").write_text("\n".join(sorted(lines, key=lambda l: int(l.split(",")[0]))) + "\n")
        im = d["images"][0]
        (dst / "seqinfo.ini").write_text(
            f"[Sequence]\nname={s.name}\nimDir=img1\nframeRate=25\nseqLength={len(d['images'])}\n"
            f"imWidth={im['width']}\nimHeight={im['height']}\nimExt=.jpg\n")
        n += 1
    print(f"wrote {n} sequences to {Path(args.out, args.split)}")


if __name__ == "__main__":
    main()
