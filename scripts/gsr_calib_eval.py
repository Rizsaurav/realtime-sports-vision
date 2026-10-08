#!/usr/bin/env python3
"""Calibration-only GS-HOTA: how good is a pitch calibrator, in isolation?

Ground-truth boxes, track IDs, roles, teams and jerseys are kept; only the
pitch position of each box is recomputed from the calibrator's homography
(image -> pitch, metres, SoccerNet convention: origin at centre, x along
the 105 m length, y across the 68 m width, negative y = far touchline).
The GS-HOTA drop from 100 is the calibrator's cost to a full pipeline.

Methods:
  rf_kp   roboflow/sports football-pitch-detection (YOLOv8-pose, 32 keypoints),
          keypoints re-pinned to the standard 105 x 68 m pitch, RANSAC homography
  Each method also has a "hold" policy: when a frame has < 4 good keypoints the
  last valid homography is reused (causal, no lookahead).

Usage:
  python scripts/gsr_calib_eval.py --method rf_kp --gt data/raw/soccernet/gsr \
      --split valid --trackeval /data/saurav/repos/sn-trackeval [--max-seqs 5]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from calibration.propagation import HomographyPropagator

L, W = 105.0, 68.0                                   # SoccerNet pitch, metres
PB_D, PB_W, GB_D, GB_W, CIRC, SPOT = 16.5, 40.32, 5.5, 18.32, 9.15, 11.0


def sn_template():
    """roboflow/sports 32-keypoint layout, in SoccerNet metric coordinates."""
    x0, y0 = -L / 2, -W / 2
    yp1, yp2 = -PB_W / 2, PB_W / 2
    yg1, yg2 = -GB_W / 2, GB_W / 2
    pts = [
        (x0, y0), (x0, yp1), (x0, yg1), (x0, yg2), (x0, yp2), (x0, W / 2),     # 1-6
        (x0 + GB_D, yg1), (x0 + GB_D, yg2),                                     # 7-8
        (x0 + SPOT, 0.0),                                                       # 9
        (x0 + PB_D, yp1), (x0 + PB_D, yg1), (x0 + PB_D, yg2), (x0 + PB_D, yp2),  # 10-13
        (0.0, y0), (0.0, -CIRC), (0.0, CIRC), (0.0, W / 2),                     # 14-17
        (-x0 - PB_D, yp1), (-x0 - PB_D, yg1), (-x0 - PB_D, yg2), (-x0 - PB_D, yp2),  # 18-21
        (-x0 - SPOT, 0.0),                                                      # 22
        (-x0 - GB_D, yg1), (-x0 - GB_D, yg2),                                   # 23-24
        (-x0, y0), (-x0, yp1), (-x0, yg1), (-x0, yg2), (-x0, yp2), (-x0, W / 2),  # 25-30
        (-CIRC, 0.0), (CIRC, 0.0),                                              # 31-32
    ]
    return np.asarray(pts, np.float32)


class RFKeypointCalibrator:
    def __init__(self, weights, device="cuda", kp_conf=0.5, imgsz=None):
        from ultralytics import YOLO
        self.m = YOLO(weights)
        self.device, self.kp_conf = device, kp_conf
        self.imgsz = imgsz or self.m.overrides.get("imgsz", 640)
        self.tpl = sn_template()

    def homography(self, frame):
        r = self.m.predict(frame, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None
        k = r.keypoints.data[0].cpu().numpy()                 # (32, 3): x, y, conf
        ok = k[:, 2] > self.kp_conf
        if ok.sum() < 4:
            return None
        H, inl = cv2.findHomography(k[ok, :2], self.tpl[ok], cv2.RANSAC, 1.0)
        return H if H is not None and inl is not None and inl.sum() >= 4 else None


class PnLCalibrator:
    """PnLCalib (points + lines HRNet, SoccerNet-trained; GPL-2.0, research use)."""

    def __init__(self, repo, w_kp, w_line, device="cuda", kp_thr=0.3434, line_thr=0.7867,
                 refine=False, width=1920, height=1080):
        sys.path.insert(0, repo)
        import torch, yaml
        import torchvision.transforms as T
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from utils.utils_calib import FramebyFrameCalib
        from utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool,
                                         get_keypoints_from_heatmap_batch_maxpool_l,
                                         complete_keypoints, coords_to_dict)
        self.torch, self.dev = torch, device
        self.f = (get_keypoints_from_heatmap_batch_maxpool,
                  get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict)
        cfg = yaml.safe_load(open(f"{repo}/config/hrnetv2_w48.yaml"))
        cfg_l = yaml.safe_load(open(f"{repo}/config/hrnetv2_w48_l.yaml"))
        self.m = get_cls_net(cfg)
        self.m.load_state_dict(torch.load(w_kp, map_location=device))
        self.ml = get_cls_net_l(cfg_l)
        self.ml.load_state_dict(torch.load(w_line, map_location=device))
        self.m.to(device).eval()
        self.ml.to(device).eval()
        self.resize = T.Resize((540, 960))
        self.cam = FramebyFrameCalib(iwidth=width, iheight=height, denormalize=True)
        self.kp_thr, self.line_thr, self.refine = kp_thr, line_thr, refine

    def homography(self, frame):
        torch = self.torch
        kp_fn, line_fn, complete, to_dict = self.f
        x = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        x = self.resize(x.unsqueeze(0)).to(self.dev)
        with torch.no_grad():
            hm, hl = self.m(x), self.ml(x)
        kp = to_dict(kp_fn(hm[:, :-1]), threshold=self.kp_thr)
        ln = to_dict(line_fn(hl[:, :-1]), threshold=self.line_thr)
        kp, ln = complete(kp[0], ln[0], w=960, h=540, normalize=True)
        self.cam.update(kp, ln)
        r = self.cam.heuristic_voting(refine_lines=self.refine)
        if not r:
            return None
        c = r["cam_params"]
        K = np.array([[c["x_focal_length"], 0, c["principal_point"][0]],
                      [0, c["y_focal_length"], c["principal_point"][1]], [0, 0, 1]])
        It = np.eye(4)[:-1]
        It[:, -1] = -np.array(c["position_meters"])
        P = K @ (np.array(c["rotation_matrix"]) @ It)
        Hw2i = P[:, [0, 1, 3]]                    # world z=0 plane -> image
        if abs(np.linalg.det(Hw2i)) < 1e-12:
            return None
        return np.linalg.inv(Hw2i)


def project(H, pts):
    p = cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), H)
    return p.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="rf_kp")
    ap.add_argument("--weights", default="/data/saurav/repos/calib_weights/football-pitch-detection.pt")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--trackeval", required=True)
    ap.add_argument("--kp-conf", type=float, default=0.5)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pnl-repo", default="/data/saurav/repos/PnLCalib")
    ap.add_argument("--pnl-refine", action="store_true")
    ap.add_argument("--cache-dir", default="",
                    help="save per-frame homographies here (<seq>.npz: H[N,3,3] NaN=fail, ms[N])")
    ap.add_argument("--from-cache", default="",
                    help="read homographies from a cache dir instead of running the model")
    ap.add_argument("--keyframe-k", type=int, default=0,
                    help="with --from-cache: use the calibrator only every k frames (and on "
                         "propagation failure) and propagate with optical flow in between; "
                         "-1 = calibrate once, then propagate only")
    ap.add_argument("--prop-width", type=int, default=640)
    ap.add_argument("--mask-players", action="store_true",
                    help="exclude GT person boxes from flow features (upper bound on masking)")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th sequence (cache only)")
    ap.add_argument("--flip-y", action="store_true",
                    help="negate pitch y (if a calibrator's touchline convention is mirrored)")
    args = ap.parse_args()

    if args.from_cache:
        calib = None
    elif args.method == "pnlcalib":
        wdir = Path(args.weights).parent
        calib = PnLCalibrator(args.pnl_repo, str(wdir / "SV_kp"), str(wdir / "SV_lines"),
                              args.device, refine=args.pnl_refine)
    else:
        calib = RFKeypointCalibrator(args.weights, args.device, args.kp_conf)
    gt_root = Path(args.gt).resolve()
    seqs = sorted(p.name for p in (gt_root / args.split).iterdir()
                  if (p / "Labels-GameState.json").exists())[: args.max_seqs or None]
    si, sn = map(int, args.shard.split("/"))
    seqs = seqs[si::sn]
    k = args.keyframe_k
    tag = (f"{args.method}{'_refine' if args.pnl_refine else ''}" if args.method == "pnlcalib"
           else f"{args.method}_kp{args.kp_conf}") + ("_flipy" if args.flip_y else "") \
        + (f"_kf{k}_w{args.prop_width}{'_mask' if args.mask_players else ''}" if k else "")
    trk_root = Path("results/_work/gsr_calib").resolve()
    out = trk_root / f"SoccerNetGS-{args.split}" / tag / "data"
    if sn == 1:
        shutil.rmtree(out.parent, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    errs, ms, n_frames, n_fail = [], [], 0, 0
    prop_ms, n_key = [], 0
    for s in seqs:
        d = json.loads((gt_root / args.split / s / "Labels-GameState.json").read_text())
        img_dir = gt_root / args.split / s / "img1"
        by_img = {}
        for a in d["annotations"]:
            if a.get("supercategory") == "object":
                by_img.setdefault(a["image_id"], []).append(a)
        preds, H_last = [], None
        if args.from_cache:
            z = np.load(Path(args.from_cache) / f"{s}.npz")
            cached_H, cached_ms = z["H"], z["ms"]
        Hs, mss = [], []
        prop = HomographyPropagator(width=args.prop_width) if k else None
        last_key = None
        for fi, im in enumerate(d["images"]):
            if args.from_cache and k:
                frame = cv2.imread(str(img_dir / im["file_name"]))
                boxes = ([(a["bbox_image"]["x"], a["bbox_image"]["y"],
                           a["bbox_image"]["x"] + a["bbox_image"]["w"],
                           a["bbox_image"]["y"] + a["bbox_image"]["h"])
                          for a in by_img.get(im["image_id"], [])]
                         if args.mask_players else None)
                due = last_key is None or (k > 0 and fi - last_key >= k)
                H = None
                if not due:
                    t = time.perf_counter()
                    H = prop.step(frame, boxes)
                    prop_ms.append((time.perf_counter() - t) * 1000)
                if H is None:
                    Hc = cached_H[fi]
                    if not np.isnan(Hc).any():
                        H = Hc
                        prop.keyframe(frame, H, boxes)
                        last_key = fi
                        n_key += 1
                        ms.append(float(cached_ms[fi]))
            elif args.from_cache:
                H = None if np.isnan(cached_H[fi]).any() else cached_H[fi]
                ms.append(float(cached_ms[fi]))
            else:
                frame = cv2.imread(str(img_dir / im["file_name"]))
                t = time.perf_counter()
                H = calib.homography(frame)
                ms.append((time.perf_counter() - t) * 1000)
            Hs.append(np.full((3, 3), np.nan) if H is None else H)
            mss.append(ms[-1] if ms else 0.0)
            n_frames += 1
            if H is None:
                n_fail += 1
                H = H_last
            else:
                H_last = H
            if H is None:
                continue
            for a in by_img.get(im["image_id"], []):
                b = a["bbox_image"]
                yb = b["y"] + b["h"]
                (xl, yl), (xm, ym), (xr, yr) = project(
                    H, [(b["x"], yb), (b["x_center"], yb), (b["x"] + b["w"], yb)])
                if args.flip_y:
                    yl, ym, yr = -yl, -ym, -yr
                p = dict(a)
                p["bbox_pitch"] = {"x_bottom_left": float(xl), "y_bottom_left": float(yl),
                                   "x_bottom_middle": float(xm), "y_bottom_middle": float(ym),
                                   "x_bottom_right": float(xr), "y_bottom_right": float(yr)}
                preds.append(p)
                g = a.get("bbox_pitch")
                if g and a["attributes"]["role"] != "ball":
                    errs.append(np.hypot(xm - g["x_bottom_middle"], ym - g["y_bottom_middle"]))
        (out / f"{s}.json").write_text(json.dumps({"predictions": preds}))
        if args.cache_dir:
            Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(args.cache_dir) / f"{s}.npz",
                                H=np.asarray(Hs, np.float64), ms=np.asarray(mss, np.float32))
        print(f"  {s}: done", flush=True)
    if sn > 1:
        print(f"[calib] shard {args.shard} cached {len(seqs)} seqs")
        return

    seq_info = [str(x) for x in seqs]
    cmd = [sys.executable, str(Path(args.trackeval) / "scripts" / "run_soccernet_gs.py"),
           "--GT_FOLDER", str(gt_root), "--TRACKERS_FOLDER", str(trk_root),
           "--SPLIT_TO_EVAL", args.split, "--TRACKERS_TO_EVAL", tag,
           "--METRICS", "HOTA", "--PRINT_CONFIG", "False", "--USE_PARALLEL", "False",
           "--SEQ_INFO", *seq_info]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    summ = trk_root / f"SoccerNetGS-{args.split}" / tag / "cls_comb_det_av_summary.txt"
    head, vals = summ.read_text().splitlines()[:2]
    gs = dict(zip(head.split(), map(float, vals.split())))
    e = np.asarray(errs)
    res = {"method": tag, "split": args.split, "n_seqs": len(seqs), "frames": n_frames,
           "frames_no_homography_pct": round(100 * n_fail / max(n_frames, 1), 2),
           "gs_hota_calib_only": gs["HOTA"], "gs_deta": gs["DetA"], "gs_assa": gs["AssA"],
           "err_m_median": round(float(np.median(e)), 3), "err_m_p90": round(float(np.percentile(e, 90)), 3),
           "err_m_mean": round(float(e.mean()), 3), "pct_within_1m": round(float((e < 1).mean() * 100), 1),
           "calib_ms_p50": round(float(np.median(ms)), 2), "calib_ms_p95": round(float(np.percentile(ms, 95)), 2),
           "keyframe_k": k, "keyframe_pct": round(100 * n_key / max(n_frames, 1), 2) if k else 100.0,
           "prop_ms_p50": round(float(np.median(prop_ms)), 2) if prop_ms else None,
           "prop_ms_p95": round(float(np.percentile(prop_ms, 95)), 2) if prop_ms else None}
    dest = Path(f"results/gsr_calib_{tag}_{args.split}.json")
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
