"""Keyframe pitch calibration with PnLCalib (Gutierrez-Perez and Agudo; GPL-2.0, research use).

The calibrator returns the image->pitch homography (metres, origin at the
centre spot) for one 1920x1080 frame, or None when it finds too few pitch
features. Between keyframes the live pipeline carries the homography forward
with optical flow (src/calibration/propagation.py).
"""
import sys

import cv2
import numpy as np


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


def _worker(repo, w_kp, w_line, device, inq, outq):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    cal = PnLCalibrator(repo, w_kp, w_line, device=device)
    cal.homography(np.zeros((540, 960, 3), np.uint8))          # warm-up
    outq.put(("ready", None))
    while True:
        item = inq.get()
        if item is None:
            return
        k, small = item
        try:
            H = cal.homography(small)
        except Exception:
            H = None
        outq.put((k, H))


class CalibWorker:
    """PnLCalib in its own process, so its Python-heavy camera search does not hold
    the pipeline's GIL. request() never blocks; results() drains finished keyframes."""

    def __init__(self, repo, w_kp, w_line, device="cuda"):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.inq, self.outq = ctx.Queue(1), ctx.Queue()
        self.proc = ctx.Process(target=_worker, args=(repo, w_kp, w_line, device, self.inq, self.outq),
                                daemon=True)
        self.proc.start()
        kind, _ = self.outq.get(timeout=600)
        assert kind == "ready"

    def request(self, k, small):
        import queue
        try:
            self.inq.put_nowait((k, small))
            return True
        except queue.Full:
            return False

    def homography(self, small, k=-1):
        """Blocking call (evaluation mode)."""
        self.inq.put((k, small))
        while True:
            kk, H = self.outq.get()
            if kk == k:
                return H

    def results(self, timeout=None):
        import queue
        try:
            return self.outq.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        try:
            self.inq.put_nowait(None)
        except Exception:
            pass
