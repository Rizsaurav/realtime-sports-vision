"""Unified tracker interface.

Primary: boxmot (pip install boxmot) — bundles ByteTrack, OC-SORT, etc.
Fallback: a small built-in greedy IoU tracker (no deps) so smoke tests and
CPU dev never block on tracker installation.

ByteTrack: https://github.com/FoundationVision/ByteTrack (motion-only)
OC-SORT:   https://github.com/noahcao/OC_SORT (motion-only)
Neither has learned weights -> nothing to train.
"""

import numpy as np


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class _GreedyIoUTracker:
    """Minimal fallback tracker: greedy IoU matching + track aging.

    Good enough for smoke tests and UI dev. NOT the benchmark tracker —
    Session A numbers use ByteTrack/OC-SORT.
    """

    def __init__(self, iou_thr=0.3, max_age=30):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks = {}  # tid -> dict(box, age, cls)
        self._next_id = 1

    def update(self, detections):
        dets = sorted(detections, key=lambda d: -d[4])
        assigned = {}
        used_tids = set()
        for d in dets:
            best, best_iou = None, self.iou_thr
            for tid, t in self.tracks.items():
                if tid in used_tids:
                    continue
                iou = _iou(d[:4], t["box"])
                if iou > best_iou:
                    best, best_iou = tid, iou
            if best is None:
                tid = self._next_id
                self._next_id += 1
            else:
                tid = best
                used_tids.add(tid)
            assigned[tid] = {"box": d[:4], "age": 0, "cls": d[5]}
        for tid in list(self.tracks):
            if tid not in assigned:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] <= self.max_age:
                    assigned[tid] = self.tracks[tid]
        self.tracks = assigned
        return [(t["box"][0], t["box"][1], t["box"][2], t["box"][3], tid, t["cls"])
                for tid, t in self.tracks.items() if t["age"] == 0]

    def reset(self):
        self.tracks, self._next_id = {}, 1


class Tracker:
    def __init__(self, name="bytetrack"):
        self.name = name
        self._impl = None
        self._kind = None
        try:
            if name == "bytetrack":
                from boxmot import ByteTrack
                self._impl = ByteTrack()
            elif name == "ocsort":
                from boxmot import OcSort
                self._impl = OcSort()
            else:
                raise ValueError(f"unknown tracker {name}")
            self._kind = "boxmot"
        except ImportError:
            print(f"[tracker] boxmot not installed; using built-in IoU fallback "
                  f"(pip install boxmot for the real {name})")
            self._impl = _GreedyIoUTracker()
            self._kind = "fallback"

    def update(self, detections, frame):
        """detections: list of (x1,y1,x2,y2,score,cls); frame: HxWx3 BGR.
        Returns list of (x1,y1,x2,y2,track_id,cls)."""
        if self._kind == "boxmot":
            import numpy as np
            dets = np.array([[x1, y1, x2, y2, s, c]
                             for x1, y1, x2, y2, s, c in detections],
                            dtype=np.float32).reshape(-1, 6)
            out = self._impl.update(dets, frame)
            # boxmot -> (x1,y1,x2,y2,track_id,conf,cls,...)
            return [(float(r[0]), float(r[1]), float(r[2]), float(r[3]),
                     int(r[4]), int(r[6]) if len(r) > 6 else 0) for r in out]
        return self._impl.update(detections)

    def reset(self):
        if self._kind == "boxmot":
            try:
                from boxmot import ByteTrack, OcSort
                cls = ByteTrack if self.name == "bytetrack" else OcSort
                self._impl = cls()
            except ImportError:
                pass
        else:
            self._impl.reset()
