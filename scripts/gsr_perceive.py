#!/usr/bin/env python3
"""Stage A of the streaming GSR pipeline: per-frame perception, cached to disk.

detector (player/goalkeeper/referee/ball) -> roboflow ByteTrack on people ->
image->pitch homography (accurate calibrator on keyframes every k frames,
optical-flow propagation in between; keyframe homographies come from the
calibration cache) -> per tracked box: pitch foot points + torso colour feature.

Attribute decisions (role/team/jersey under an output delay) are made later in
gsr_assemble.py from this cache, so each delay setting costs no GPU time.

Usage:
  python scripts/gsr_perceive.py --weights runs/finetune/yolo26s_gsr_1280/weights/best.pt \
      --imgsz 1280 --calib-cache results/calib_cache/pnlcalib_valid --keyframe-k 10 \
      --gt data/raw/soccernet/gsr --split valid --out results/gsr_perception/s1280_kf10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration.propagation import HomographyPropagator
from gsr_team_eval import torso_feature

NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}


def project(H, pts):
    p = cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), H)
    return p.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--calib-cache", required=True)
    ap.add_argument("--keyframe-k", type=int, default=10)
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import supervision as sv
    import trackers
    from ultralytics import YOLO

    model = YOLO(args.weights)
    seqs = sorted(p for p in (Path(args.gt) / args.split).iterdir()
                  if (p / "Labels-GameState.json").exists())
    si, sn = map(int, args.shard.split("/"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in seqs[si::sn]:
        if (out / f"{s.name}.json").exists():
            continue
        d = json.loads((s / "Labels-GameState.json").read_text())
        cache = np.load(Path(args.calib_cache) / f"{s.name}.npz")["H"]
        trk = trackers.ByteTrackTracker(frame_rate=25)
        prop = HomographyPropagator()
        last_key, H_last = None, None
        rows, ball, t_det, t_trk, t_cal = [], [], [], [], []
        for fi, im in enumerate(d["images"]):
            frame = cv2.imread(str(s / "img1" / im["file_name"]))
            t0 = time.perf_counter()
            r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device,
                              verbose=False)[0]
            t1 = time.perf_counter()
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            people = cls != 3
            for b, c in zip(xyxy[~people], conf[~people]):
                ball.append((fi, *map(float, b), float(c)))
            dets = sv.Detections(xyxy=xyxy[people].astype(np.float32),
                                 confidence=conf[people].astype(np.float32),
                                 class_id=cls[people])
            tracked = trk.update(dets, frame)
            t2 = time.perf_counter()
            boxes = tracked.xyxy
            # calibration: keyframe from the accurate calibrator, else propagate
            due = last_key is None or fi - last_key >= args.keyframe_k
            H = None if due else prop.step(frame, boxes)
            if H is None and not np.isnan(cache[fi]).any():
                H = cache[fi]
                prop.keyframe(frame, H, boxes)
                last_key = fi
            if H is None:
                H = H_last
            H_last = H
            t3 = time.perf_counter()
            t_det.append((t1 - t0) * 1000); t_trk.append((t2 - t1) * 1000); t_cal.append((t3 - t2) * 1000)
            if tracked.tracker_id is None or len(tracked) == 0:
                continue
            for b, tid, c, sc in zip(tracked.xyxy, tracked.tracker_id, tracked.class_id,
                                     tracked.confidence):
                if tid is None or tid < 0:
                    continue
                x1, y1, x2, y2 = map(float, b)
                pitch = None
                if H is not None:
                    pl, pm, pr = project(H, [(x1, y2), ((x1 + x2) / 2, y2), (x2, y2)])
                    pitch = [float(v) for v in (*pl, *pm, *pr)]
                f = torso_feature(frame, {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
                rows.append({"frame": fi, "image_id": im["image_id"], "track": int(tid),
                             "cls": NAMES[int(c)], "score": float(sc),
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "pitch": pitch,
                             "feat": None if f is None else [float(v) for v in f]})
        (out / f"{s.name}.json").write_text(json.dumps({
            "seq": s.name, "rows": rows, "ball": ball,
            "ms": {"det_p50": float(np.median(t_det)), "trk_p50": float(np.median(t_trk)),
                   "calib_prop_p50": float(np.median(t_cal))}}))
        print(f"  {s.name}: {len(rows)} tracked boxes, det {np.median(t_det):.1f} ms, "
              f"trk {np.median(t_trk):.2f} ms, calib {np.median(t_cal):.1f} ms", flush=True)
    (out / "meta.json").write_text(json.dumps(vars(args), indent=1))


if __name__ == "__main__":
    main()
