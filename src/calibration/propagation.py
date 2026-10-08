"""Keyframe calibration with cheap homography propagation.

An accurate pitch calibrator (e.g. PnLCalib, ~30 ms optimised) runs only on
keyframes. Between keyframes the image->pitch homography is carried forward
by estimating the frame-to-frame image homography from sparse optical flow
on a downscaled grey image (a few ms on CPU) and composing it with the last
calibration. When flow tracking fails (camera cut, heavy blur) the
propagator reports failure so the caller can force a keyframe.
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
        self.prev, self.scale, self.H = None, None, None
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

    def keyframe(self, frame, H_img2pitch, boxes=None):
        """Reset the chain with an accurate calibration for this frame."""
        self.prev, self.scale = self._grey(frame)
        self.prev_mask = self._mask(self.prev, boxes, self.scale)
        self.H = H_img2pitch

    def step(self, frame, boxes=None):
        """Propagate to `frame`. Returns the homography, or None if tracking failed."""
        if self.prev is None or self.H is None:
            return None
        g, s = self._grey(frame)
        p0 = cv2.goodFeaturesToTrack(self.prev, mask=self.prev_mask, **self.feat)
        if p0 is None or len(p0) < self.min_inliers:
            return self._fail(g, boxes, s)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None, **self.lk)
        ok = st.reshape(-1) == 1
        if ok.sum() < self.min_inliers:
            return self._fail(g, boxes, s)
        M, inl = cv2.findHomography(p1[ok], p0[ok], cv2.RANSAC, self.ransac_px)  # cur -> prev
        self.last_inliers = int(inl.sum()) if inl is not None else 0
        if M is None or self.last_inliers < self.min_inliers:
            return self._fail(g, boxes, s)
        S = np.diag([s, s, 1.0])
        M_full = np.linalg.inv(S) @ M @ S
        self.H = self.H @ M_full
        self.prev, self.prev_mask = g, self._mask(g, boxes, s)
        return self.H

    def _fail(self, g, boxes, s):
        self.prev, self.prev_mask = g, self._mask(g, boxes, s)
        self.H = None
        return None
