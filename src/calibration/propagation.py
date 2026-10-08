"""Keyframe calibration with cheap homography propagation, plus shared camera motion.

An accurate pitch calibrator (e.g. PnLCalib, ~30 ms optimised) runs only on
keyframes. Every frame, the frame-to-frame image homography is estimated once
from sparse optical flow on a downscaled grey image (a few ms on CPU). That one
estimate serves two consumers:
  - calibration: the image->pitch homography is carried forward between keyframes
  - tracking: its affine part is the tracker's camera motion compensation (CMC)
When flow tracking fails (camera cut, heavy blur) motion is None so the caller
can force a keyframe and the tracker gets an identity transform.
"""

import cv2
import numpy as np


class HomographyPropagator:
    def __init__(self, width=640, max_corners=400, quality=0.01, min_dist=8,
                 ransac_px=1.5, min_inliers=40, mask_top_frac=0.0):
        self.width = width
        self.feat = dict(maxCorners=max_corners, qualityLevel=quality, minDistance=min_dist,
                         blockSize=7)
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.ransac_px, self.min_inliers = ransac_px, min_inliers
        self.mask_top_frac = mask_top_frac
        self.prev, self.prev_mask, self.H = None, None, None
        self.last_inliers = 0

    def _grey(self, frame):
        h, w = frame.shape[:2]
        s = self.width / w
        g = cv2.cvtColor(cv2.resize(frame, (self.width, int(round(h * s))),
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return g, s

    def _mask(self, g, boxes, s):
        m = np.full(g.shape, 255, np.uint8)
        if self.mask_top_frac:
            m[: int(g.shape[0] * self.mask_top_frac)] = 0
        for x1, y1, x2, y2 in (boxes if boxes is not None else []):
            m[max(0, int(y1 * s)): int(y2 * s) + 1, max(0, int(x1 * s)): int(x2 * s) + 1] = 0
        return m

    def motion(self, frame, boxes=None):
        """Frame-to-frame homography current->previous (full-res pixels), or None.
        Always advances the reference frame to `frame`."""
        g, s = self._grey(frame)
        M = None
        if self.prev is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev, mask=self.prev_mask, **self.feat)
            if p0 is not None and len(p0) >= self.min_inliers:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None, **self.lk)
                ok = st.reshape(-1) == 1
                if ok.sum() >= self.min_inliers:
                    Ms, inl = cv2.findHomography(p1[ok], p0[ok], cv2.RANSAC, self.ransac_px)
                    self.last_inliers = int(inl.sum()) if inl is not None else 0
                    if Ms is not None and self.last_inliers >= self.min_inliers:
                        S = np.diag([s, s, 1.0])
                        M = np.linalg.inv(S) @ Ms @ S
        self.prev, self.prev_mask = g, self._mask(g, boxes, s)
        return M

    @staticmethod
    def cmc_affine(M):
        """Tracker CMC transform (previous->current, 2x3) from motion() output."""
        if M is None:
            return np.eye(2, 3, dtype=np.float32)
        A = np.linalg.inv(M)
        return (A[:2] / A[2, 2]).astype(np.float32)

    def keyframe(self, frame, H_img2pitch, boxes=None):
        """Reset the calibration chain with an accurate homography for this frame."""
        self.prev, s = self._grey(frame)
        self.prev_mask = self._mask(self.prev, boxes, s)
        self.H = H_img2pitch

    def advance(self, M):
        """Carry the calibration forward with a motion() estimate; None on failure."""
        if M is None or self.H is None:
            self.H = None
            return None
        self.H = self.H @ M
        return self.H

    def step(self, frame, boxes=None):
        """Estimate motion and propagate the calibration to `frame` in one call."""
        return self.advance(self.motion(frame, boxes))
