"""The real-time loop: read frame -> detect -> track -> annotate -> emit.

Modes:
  - "replay": read results JSON (no GPU, for the UI and demos)
  - "live":   run the full pipeline (GPU sessions only)

Design rule: inference and presentation are decoupled. Every live run
writes results/latency traces to results/, and the UI only ever reads them.
You never burn GPU to show someone the demo.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np


class StreamingPipeline:
    def __init__(self, detector, tracker, cache=None, out_path=None,
                 draw=True, classes=None):
        self.detector = detector
        self.tracker = tracker
        self.cache = cache
        self.out_path = Path(out_path) if out_path else None
        self.draw = draw
        self.classes = classes or {0: "person", 32: "ball"}  # COCO ids
        self.latencies_ms = []
        self._writer = None

    # -- drawing ---------------------------------------------------------
    @staticmethod
    def _color(tid):
        rng = np.random.RandomState(tid * 7919)
        return tuple(int(c) for c in rng.randint(60, 255, 3))

    def _annotate(self, frame, tracks, fps):
        for x1, y1, x2, y2, tid, cls in tracks:
            color = tuple(int(c) for c in self._color(tid))[::-1]  # RGB->BGR
            label = f"ID {tid} {self.classes.get(cls, cls)}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"{fps:.1f} FPS", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    # -- main loop --------------------------------------------------------
    def run(self, video_path, max_frames=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open {video_path}")
        writer = None
        if self.out_path:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
            writer = cv2.VideoWriter(str(self.out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_in, (w, h))
        if self.cache:
            self.cache.reset()
        self.tracker.reset()
        self.latencies_ms = []

        frame_id, tracks_out, t_run = 0, [], time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and frame_id >= max_frames):
                break
            t0 = time.perf_counter()
            if self.cache and not self.cache.is_keyframe():
                # Week 2: reuse cached backbone features for the head.
                # For now (CNN backends) this falls through to full inference;
                # the efficient-ViT path wires the real cache in cache.py.
                dets = self.detector.infer(frame)
            else:
                dets = self.detector.infer(frame)
                if self.cache:
                    self.cache.update(None)  # Week 2: pass real features
            tracks = self.tracker.update(dets, frame)
            dt_ms = (time.perf_counter() - t0) * 1000
            self.latencies_ms.append(dt_ms)
            if self.cache:
                self.cache.step()
            tracks_out.append((frame_id, tracks))
            if self.draw:
                inst_fps = 1000 / dt_ms if dt_ms > 0 else 0
                frame = self._annotate(frame, tracks, inst_fps)
            if writer:
                writer.write(frame)
            frame_id += 1

        cap.release()
        if writer:
            writer.release()
        return {
            "frames": frame_id,
            "wall_s": time.perf_counter() - t_run,
            "tracks": [[fid, [[float(x1), float(y1), float(x2), float(y2),
                               int(tid), int(cls)] for x1, y1, x2, y2, tid, cls in tr]]
                       for fid, tr in tracks_out],
            **self.latency_report(),
        }

    def latency_report(self):
        l = np.array(self.latencies_ms, dtype=np.float64)
        if len(l) == 0:
            return {"fps": 0, "latency_p50_ms": 0, "latency_p95_ms": 0, "frames": 0}
        return {
            "fps": float(1000 / l.mean()),
            "latency_p50_ms": float(np.percentile(l, 50)),
            "latency_p95_ms": float(np.percentile(l, 95)),
            "frames": int(len(l)),
        }

    def save(self, result, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # tracks can be large; keep a slim summary + full traces
        slim = {k: v for k, v in result.items() if k != "tracks"}
        slim["tracks"] = result["tracks"]  # keep; JSON is cheap vs GPU time
        path.write_text(json.dumps(slim))
        return path
