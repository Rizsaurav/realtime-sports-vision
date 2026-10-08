#!/usr/bin/env python3
"""Offline tracker sweep on cached detections (CPU, parallel over sequences).

Every tracker config is run over every sequence from the same cached
detections, scored with TrackEval-compatible `trackers eval`, and timed
(tracker update time only). Results: results/tracker_sweep/<name>.json

Usage:
  python scripts/sweep_trackers.py --dets results/dets/yolo26s_ft \
      [--only rf_botsort_cmc,bm_bytetrack_low] [--workers 40]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark.mot_eval import run_trackeval
from calibration.propagation import HomographyPropagator

FR = 25  # SportsMOT frame rate

# name -> (library, class, min detection score fed to the tracker, kwargs, needs_frames)
CONFIGS = {
    # reproduces the current baseline: boxmot defaults, only boxes >= 0.25
    "bm_bytetrack_conf25": ("boxmot", "ByteTrack", 0.25, {}, False),
    # same tracker, fed low-score boxes as ByteTrack intends, correct frame rate
    "bm_bytetrack_low": ("boxmot", "ByteTrack", 0.1,
                         dict(min_conf=0.1, track_thresh=0.45, match_thresh=0.8,
                              track_buffer=FR, frame_rate=FR), False),
    "bm_bytetrack_low_buf50": ("boxmot", "ByteTrack", 0.1,
                               dict(min_conf=0.1, track_thresh=0.45, match_thresh=0.8,
                                    track_buffer=50, frame_rate=FR), False),
    "bm_ocsort_byte": ("boxmot", "OcSort", 0.1,
                       dict(min_conf=0.1, use_byte=True, delta_t=3, inertia=0.2), False),
    "rf_sort": ("rf", "SORTTracker", 0.25, dict(frame_rate=FR), False),
    "rf_bytetrack": ("rf", "ByteTrackTracker", 0.1, dict(frame_rate=FR), False),
    "rf_ocsort": ("rf", "OCSORTTracker", 0.1, dict(frame_rate=FR), False),
    "rf_cbiou": ("rf", "CBIoUTracker", 0.1, dict(frame_rate=FR), False),
    "rf_botsort_nocmc": ("rf", "BoTSORTTracker", 0.1,
                         dict(frame_rate=FR, enable_cmc=False), False),
    "rf_botsort_cmc": ("rf", "BoTSORTTracker", 0.1,
                       dict(frame_rate=FR, enable_cmc=True), True),
    "rf_mcbyte_cmc": ("rf", "McByteTracker", 0.1,
                      dict(frame_rate=FR, enable_cmc=True), True),
    "rf_mcbyte_nocmc": ("rf", "McByteTracker", 0.1,
                        dict(frame_rate=FR, enable_cmc=False), False),
    # roboflow's tuned SportsMOT settings for BoT-SORT
    "rf_botsort_cmc_tuned": ("rf", "BoTSORTTracker", 0.1,
                             dict(frame_rate=FR, enable_cmc=True, lost_track_buffer=30,
                                  track_activation_threshold=0.8,
                                  high_conf_det_threshold=0.7,
                                  minimum_consecutive_frames=2,
                                  minimum_iou_threshold_first_assoc=0.1,
                                  minimum_iou_threshold_second_assoc=0.5,
                                  minimum_iou_threshold_unconfirmed_assoc=0.3), True),
    "rf_botsort_cmc_tuned_buf60": ("rf", "BoTSORTTracker", 0.1,
                                   dict(frame_rate=FR, enable_cmc=True, lost_track_buffer=60,
                                        track_activation_threshold=0.8,
                                        high_conf_det_threshold=0.7,
                                        minimum_consecutive_frames=2,
                                        minimum_iou_threshold_first_assoc=0.1,
                                        minimum_iou_threshold_second_assoc=0.5,
                                        minimum_iou_threshold_unconfirmed_assoc=0.3), True),
    # camera motion from the calibration propagator's optical flow instead of the
    # tracker's own estimator ("inv" flips the direction, to check the convention)
    "rf_mcbyte_sharedcmc": ("rf", "McByteTracker", 0.1,
                            dict(frame_rate=FR, enable_cmc=True), "shared"),
    "rf_mcbyte_sharedcmc_inv": ("rf", "McByteTracker", 0.1,
                                dict(frame_rate=FR, enable_cmc=True), "shared_inv"),
    "rf_mcbyte_sharedcmc_w960": ("rf", "McByteTracker", 0.1,
                                 dict(frame_rate=FR, enable_cmc=True), "shared_w960"),
}


class _FixedCMC:
    H = np.eye(2, 3, dtype=np.float32)

    def estimate(self, frame, dets_xyxy=None):
        return self.H

    def reset(self):
        self.H = np.eye(2, 3, dtype=np.float32)


def interpolate(txt, max_gap):
    """Linearly fill per-track gaps of <= max_gap frames (offline post-process)."""
    by_id = {}
    for line in txt.strip().splitlines():
        p = line.split(",")
        by_id.setdefault(int(p[1]), []).append((int(p[0]), *map(float, p[2:6])))
    out = []
    for tid, rows in by_id.items():
        rows.sort()
        for (f0, *b0), (f1, *b1) in zip(rows, rows[1:]):
            out.append((f0, tid, *b0))
            if 1 < f1 - f0 <= max_gap:
                for f in range(f0 + 1, f1):
                    a = (f - f0) / (f1 - f0)
                    out.append((f, tid, *[x + (y - x) * a for x, y in zip(b0, b1)]))
        out.append((rows[-1][0], tid, *rows[-1][1:]))
    out.sort()
    return "\n".join(f"{f},{t},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
                     for f, t, x, y, w, h in out) + "\n"


def make_tracker(lib, cls, kw):
    if lib == "boxmot":
        import boxmot
        return getattr(boxmot, cls)(**kw)
    import trackers
    return getattr(trackers, cls)(**kw)


def run_seq(job):
    name, seq, det_path, img_dir, wh = job
    import cv2
    cv2.setNumThreads(1)
    lib, cls, min_s, kw, needs_frames = CONFIGS[name]
    z = np.load(det_path)
    fids, dets = z["fid"], z["det"]
    keep = dets[:, 4] >= min_s
    fids, dets = fids[keep], dets[keep]
    frames = sorted(int(p.stem) for p in Path(img_dir).glob("*.jpg"))
    trk = make_tracker(lib, cls, kw)
    shared = needs_frames in ("shared", "shared_inv", "shared_w960")
    if shared:
        prop = HomographyPropagator(width=960 if needs_frames == "shared_w960" else 640)
        fixed, prev_boxes = _FixedCMC(), None
        trk.cmc = fixed
    blank = np.zeros((wh[1], wh[0], 3), np.uint8)
    lines, t_track = [], 0.0
    if lib == "rf":
        import supervision as sv
    for fid in frames:
        d = dets[fids == fid]
        img = cv2.imread(str(Path(img_dir) / f"{fid:06d}.jpg")) if needs_frames else blank
        t = time.perf_counter()
        if shared:
            M = prop.motion(img, prev_boxes)
            if needs_frames == "shared_inv":
                fixed.H = (np.eye(2, 3, dtype=np.float32) if M is None
                           else (M[:2] / M[2, 2]).astype(np.float32))
            else:
                fixed.H = prop.cmc_affine(M)
        if lib == "boxmot":
            arr = np.concatenate([d, np.zeros((len(d), 1), np.float32)], 1) if len(d) \
                else np.empty((0, 6), np.float32)
            out = trk.update(arr, img)
            res = [(r[0], r[1], r[2], r[3], int(r[4])) for r in out]
        else:
            sd = sv.Detections(xyxy=d[:, :4].astype(np.float32),
                               confidence=d[:, 4].astype(np.float32),
                               class_id=np.zeros(len(d), int))
            out = trk.update(sd, img)
            ids = out.tracker_id if out.tracker_id is not None else []
            res = [(b[0], b[1], b[2], b[3], int(i))
                   for b, i in zip(out.xyxy, ids) if i is not None and i >= 0]
            if shared:
                prev_boxes = out.xyxy
        t_track += time.perf_counter() - t
        for x1, y1, x2, y2, tid in res:
            lines.append(f"{fid},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1")
    return name, seq, "\n".join(lines) + "\n", len(frames), t_track


def seq_wh(seq_dir):
    info = dict(l.split("=", 1) for l in (seq_dir / "seqinfo.ini").read_text().splitlines()
                if "=" in l)
    return int(info["imWidth"].strip()), int(info["imHeight"].strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--interp", type=int, default=20,
                    help="also score each config with gap interpolation (0 = off)")
    args = ap.parse_args()

    det_dir = Path(args.dets)
    meta = json.loads((det_dir / "meta.json").read_text())
    gt_root = Path(meta.get("root", "data/raw/sportsmot")) / meta["split"]
    seqs = sorted(p.stem for p in det_dir.glob("*.npz"))
    names = [n for n in CONFIGS if not args.only or n in args.only.split(",")]
    jobs = [(n, s, str(det_dir / f"{s}.npz"), str(gt_root / s / "img1"),
             seq_wh(gt_root / s)) for n in names for s in seqs]

    with Pool(args.workers) as pool:
        outs = pool.map(run_seq, jobs, chunksize=1)

    out_dir = Path("results/tracker_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    variants = [0] + ([args.interp] if args.interp else [])
    for n in names:
        for gap in variants:
            label = n if not gap else f"{n}+interp{gap}"
            work = Path("results/_work/sweep") / det_dir.name / label
            shutil.rmtree(work, ignore_errors=True)
            (work / "tracker").mkdir(parents=True)
            frames = ttot = 0
            for name, seq, txt, nf, tt in outs:
                if name == n:
                    (work / "tracker" / f"{seq}.txt").write_text(
                        interpolate(txt, gap) if gap and txt.strip() else txt)
                    frames += nf
                    ttot += tt
            m = run_trackeval(gt_root, work / "tracker", seqs, work)
            lib, cls, min_s, kw, needs_frames = CONFIGS[n]
            rows.append({"name": label, "lib": lib, "class": cls, "min_score": min_s,
                         "kwargs": kw, "cmc_frames": needs_frames, "interp_gap": gap,
                         "online": gap == 0,
                         "track_ms_per_frame": round(ttot * 1000 / frames, 3), **m})
            print(f"{label:34} HOTA={m['hota']:.3f} DetA={m['deta']:.3f} AssA={m['assa']:.3f} "
                  f"IDF1={m['idf1']:.3f} MOTA={m['mota']:.3f} IDSW={m['id_switches']} "
                  f"track={rows[-1]['track_ms_per_frame']:.2f}ms", flush=True)
            shutil.rmtree(work, ignore_errors=True)

    rows.sort(key=lambda r: -r["hota"])
    dest = out_dir / f"{det_dir.name}.json"
    prev = json.loads(dest.read_text()) if dest.exists() else []
    merged = {r["name"]: r for r in prev}
    merged.update({r["name"]: r for r in rows})
    dest.write_text(json.dumps(sorted(merged.values(), key=lambda r: -r["hota"]), indent=1))
    print(f"[sweep] wrote {dest}")


if __name__ == "__main__":
    main()
