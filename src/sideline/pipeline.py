"""The live pipeline: video source in, per-frame game state out at several output delays.

Stages run in their own threads and hand frames on through short queues, so
GPU and CPU work overlap:

  source  decode, resample to 25 FPS / 1920x1080, JPEG preview for viewers
  gpu     detector (players, goalkeepers, referees, ball) and pose model
  cpu     optical flow -> camera motion for the tracker and calibration
          propagation; tracker; pitch projection; torso colour features;
          requests keyframe calibrations
  calib   PnLCalib on keyframes, asynchronous; a late result is carried to the
          current frame with the stored frame-to-frame motions
  jersey  legibility + PARSeq on pose-guided torso crops
  decide  one DecisionEngine per output delay; results go to subscribers

With a real-time source, a frame that arrives while the pipeline is still busy
is dropped at the source, as a live system must.
"""
import collections
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # src/ for calibration.propagation

from calibration.propagation import HomographyPropagator  # noqa: E402

from .decisions import DecisionEngine  # noqa: E402
from .geometry import foot_points, project, torso_feature  # noqa: E402
from .sources import RATE, Source  # noqa: E402

NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
STOP = object()


class SharedCMC:
    """Stand-in for the tracker's camera motion compensation: returns the flow estimate."""
    H = np.eye(2, 3, dtype=np.float32)

    def estimate(self, frame, dets_xyxy=None):
        return self.H

    def reset(self):
        self.H = np.eye(2, 3, dtype=np.float32)


class Models:
    """Loads every model once; reused across sources."""

    def __init__(self, cfg):
        from ultralytics import YOLO
        m = cfg["models"]
        self.imgsz = m.get("imgsz", 1280)
        self.det = YOLO(m["detector"], task="detect")
        self.pose = YOLO(m["pose"], task="pose") if m.get("pose") else None
        self.calib = None
        if m.get("calibrator"):
            from .calibration import CalibWorker
            c = m["calibrator"]
            self.calib = CalibWorker(c["repo"], c["keypoints"], c["lines"])
        self.jersey = None
        if m.get("jersey"):
            from .jersey import JerseyReader
            j = m["jersey"]
            self.jersey = JerseyReader(j["dir"], min_h=j.get("min_height", 40),
                                       leg_thr=j.get("legibility", 0.3), read_conf=j.get("read_conf", 0.97))
        self.warm()

    def warm(self):
        blank = np.zeros((1080, 1920, 3), np.uint8)
        for _ in range(3):
            self.det.predict(blank, imgsz=self.imgsz, device=0, verbose=False)
            if self.pose:
                self.pose.predict(blank, imgsz=self.imgsz, device=0, verbose=False)


class Stats:
    def __init__(self, n=200):
        self.lat = collections.deque(maxlen=n)
        self.done = collections.deque(maxlen=n)
        self.stage = collections.defaultdict(lambda: collections.deque(maxlen=n))
        self.frames = self.dropped = self.keyframes = 0
        self.calibrated = False

    def snapshot(self):
        d = list(self.done)
        fps = (len(d) - 1) / (d[-1] - d[0]) if len(d) > 1 and d[-1] > d[0] else 0.0
        lat = np.asarray(self.lat) if self.lat else np.zeros(1)
        return {"fps": round(fps, 1), "latency_ms_p50": round(float(np.median(lat)), 1),
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
                "frames": self.frames, "dropped": self.dropped, "keyframes": self.keyframes,
                "calibrated": self.calibrated,
                "stage_ms": {k: round(float(np.median(v)), 1) for k, v in self.stage.items() if v}}


class LivePipeline:
    def __init__(self, models, cfg, delays=(0, 25, 125, 250)):
        self.m, self.cfg = models, cfg
        p = cfg.get("pipeline", {})
        self.det_conf = p.get("det_conf", 0.1)
        self.keyframe_k = p.get("keyframe_every", 10)
        self.preview_w = p.get("preview_width", 960)
        self.sync_calib = p.get("sync_calib", False)     # evaluation: calibrate keyframes inline
        self.delays = tuple(sorted(set(int(d) for d in delays)))
        self.dcfg = cfg.get("decisions", {})
        self.subscribers = []
        self.previews = collections.OrderedDict()        # frame -> JPEG bytes, kept for the max delay
        self.preview_lock = threading.Lock()
        self.threads, self.source, self.running = [], None, False
        self.stats = Stats()
        self.error = None

    # -- control -----------------------------------------------------------------
    def subscribe(self, fn):
        self.subscribers.append(fn)

    def preview(self, frame):
        with self.preview_lock:
            return self.previews.get(frame)

    def start(self, spec, realtime=None, loop=False, max_frames=0):
        self.stop()
        self.source = Source(spec, realtime=realtime, loop=loop, max_frames=max_frames)
        self.stats, self.error = Stats(), None
        self.run_id = getattr(self, "run_id", 0) + 1
        with self.preview_lock:
            self.previews.clear()
        self.engines = {d: DecisionEngine(delay=d, tau=self.dcfg.get("tau", 4.0),
                                          ratio=self.dcfg.get("ratio", 0.6),
                                          role_window=self.dcfg.get("role_window", 25),
                                          stitch=self.dcfg.get("stitch")) for d in self.delays}
        self.q_gpu, self.q_cpu = queue.Queue(2), queue.Queue(4)
        self.q_jer, self.q_dec = queue.Queue(4), queue.Queue(8)
        self.cal_results, self.cal_lock, self.cal_sent, self.cal_pending = {}, threading.Lock(), {}, 0
        self.running = True
        stages = [self._source, self._gpu, self._cpu, self._jersey, self._decide]
        if self.m.calib:
            stages.append(self._calib)
        self.threads = [threading.Thread(target=self._guard, args=(f,), daemon=True, name=f.__name__)
                        for f in stages]
        for t in self.threads:
            t.start()

    def stop(self):
        self.running = False
        if self.source:
            self.source.stop()
        for t in self.threads:
            t.join(timeout=10)
        self.threads = []

    def wait(self):
        for t in self.threads:
            t.join()

    @property
    def alive(self):
        return any(t.is_alive() for t in self.threads)

    def _guard(self, fn):
        try:
            fn()
        except Exception as e:                            # surface stage failures to the UI
            import traceback
            traceback.print_exc()
            self.error = f"{fn.__name__}: {e}"
            self.running = False
            if self.source:
                self.source.stop()

    def _put(self, q, item):
        while self.running or item is STOP:
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                if item is STOP and not self.running:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
        return False

    # -- stages ------------------------------------------------------------------
    def _source(self):
        realtime = self.source.realtime
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), 72]
        keep = max(self.delays) + 50
        for idx, frame in self.source.frames():
            if not self.running:
                break
            t_in = time.perf_counter()
            if realtime and self.q_gpu.full():
                self.stats.dropped += 1
                continue
            small = cv2.resize(frame, (self.preview_w, int(self.preview_w * 9 / 16)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", small, enc)
            with self.preview_lock:
                self.previews[idx] = jpg.tobytes()
                while len(self.previews) > keep:
                    self.previews.popitem(last=False)
            if not self._put(self.q_gpu, (idx, frame, t_in)):
                break
        self._put(self.q_gpu, STOP)

    def _gpu(self):
        m = self.m
        while True:
            item = self.q_gpu.get()
            if item is STOP:
                return self._put(self.q_cpu, STOP)
            idx, frame, t_in = item
            t0 = time.perf_counter()
            r = m.det.predict(frame, imgsz=m.imgsz, conf=self.det_conf, device=0, verbose=False)[0]
            det = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))
            pose = None
            if m.pose:
                pr = m.pose.predict(frame, imgsz=m.imgsz, conf=0.1, device=0, verbose=False)[0]
                if pr.keypoints is not None and len(pr.boxes):
                    pose = (pr.boxes.xyxy.cpu().numpy(), pr.keypoints.data.cpu().numpy())
            self.stats.stage["gpu"].append((time.perf_counter() - t0) * 1000)
            self._put(self.q_cpu, (idx, frame, t_in, det, pose))

    def _calib(self):
        """Collects keyframe results from the calibrator process."""
        while self.running or self.cal_pending:
            r = self.m.calib.results(timeout=0.2)
            if r is None:
                if not any(t.is_alive() for t in self.threads if t.name == "_cpu"):
                    return
                continue
            k, H = r
            if k == "ready" or k[0] != self.run_id:
                continue
            k = k[1]
            self.stats.stage["calib"].append((time.perf_counter() - self.cal_sent.pop(k, time.perf_counter())) * 1000)
            with self.cal_lock:
                self.cal_results[k] = H
            self.cal_pending = max(0, self.cal_pending - 1)
            self.stats.keyframes += 1

    def _cpu(self):
        import supervision as sv
        import trackers
        trk = trackers.McByteTracker(frame_rate=int(RATE), enable_cmc=True)
        shared = SharedCMC()
        trk.cmc = shared
        prop = HomographyPropagator()
        motions, H_cur, H_from, prev_boxes, last_req, H_last = {}, None, None, None, None, None
        while True:
            item = self.q_cpu.get()
            if item is STOP:
                return self._put(self.q_jer, STOP)
            idx, frame, t_in, (xyxy, conf, cls), pose = item
            t0 = time.perf_counter()
            M = prop.motion(frame, prev_boxes)
            motions[idx] = M
            shared.H = prop.cmc_affine(M)
            people = cls != 3
            out = trk.update(sv.Detections(xyxy=xyxy[people].astype(np.float32),
                                           confidence=conf[people].astype(np.float32),
                                           class_id=cls[people]), frame)
            prev_boxes = out.xyxy
            # calibration: carry the current homography one step, then take a newer keyframe if done
            if H_cur is not None:
                H_cur = H_cur @ M if M is not None else None
            if self.m.calib:
                due = last_req is None or idx - last_req >= self.keyframe_k or (H_cur is None and M is None)
                if due and self.sync_calib:
                    small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
                    H_sync = self.m.calib.homography(small, (self.run_id, idx))
                    with self.cal_lock:
                        self.cal_results[idx] = H_sync
                    self.stats.keyframes += 1
                    last_req = idx
                elif due:
                    if self.m.calib.request((self.run_id, idx), cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)):
                        self.cal_sent[idx] = time.perf_counter()
                        self.cal_pending += 1
                        last_req = idx
                with self.cal_lock:
                    done = [k for k, H in self.cal_results.items() if H is not None and (H_from is None or k > H_from)]
                    k = max(done) if done else None
                    Hk = self.cal_results[k] if k is not None else None
                    for old in [j for j in self.cal_results if j < idx - 4 * self.keyframe_k]:
                        del self.cal_results[old]
                if k is not None:
                    H = Hk
                    for j in range(k + 1, idx + 1):
                        if j not in motions:              # dropped frame: no motion recorded
                            continue
                        if motions[j] is None:
                            H = None
                            break
                        H = H @ motions[j]
                    if H is not None:
                        H_cur, H_from = H, k
            for j in [j for j in motions if j < idx - 4 * self.keyframe_k - 10]:
                del motions[j]
            self.stats.calibrated = H_cur is not None
            if H_cur is not None:
                H_last = H_cur
            H_use = H_last                                # last good calibration until a new one
            rows = []
            if out.tracker_id is not None:
                for b, tid, c, sc in zip(out.xyxy, out.tracker_id, out.class_id, out.confidence):
                    if tid is None or tid < 0:
                        continue
                    x1, y1, x2, y2 = map(float, b)
                    f = torso_feature(frame, x1, y1, x2 - x1, y2 - y1)
                    rows.append({"track": int(tid), "cls": NAMES[int(c)], "score": float(sc),
                                 "bbox": [x1, y1, x2 - x1, y2 - y1],
                                 "pitch": foot_points(H_use, x1, y1, x2 - x1, y2 - y1) if H_use is not None else None,
                                 "feat": None if f is None else f.tolist()})
            ball = None
            bm = np.flatnonzero(~people)
            if len(bm):
                b = xyxy[bm[int(conf[bm].argmax())]]
                ball = {"bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], "pitch": None}
                if H_use is not None:
                    ball["pitch"] = project(H_use, [((b[0] + b[2]) / 2, b[3])])[0].tolist()
            self.stats.stage["cpu"].append((time.perf_counter() - t0) * 1000)
            self._put(self.q_jer, (idx, frame, t_in, rows, pose, ball, H_use))

    def _jersey(self):
        while True:
            item = self.q_jer.get()
            if item is STOP:
                return self._put(self.q_dec, STOP)
            idx, frame, t_in, rows, pose, ball, H = item
            if self.m.jersey and rows:
                t0 = time.perf_counter()
                ev = self.m.jersey.read(frame, [(r["bbox"][0], r["bbox"][1], r["bbox"][0] + r["bbox"][2],
                                                 r["bbox"][1] + r["bbox"][3]) for r in rows],
                                        [r["cls"] for r in rows],
                                        pose[0] if pose else None, pose[1] if pose else None)
                for r, e in zip(rows, ev):
                    r["jersey"] = e
                self.stats.stage["jersey"].append((time.perf_counter() - t0) * 1000)
            self._put(self.q_dec, (idx, t_in, rows, ball, H))

    def _decide(self):
        while True:
            item = self.q_dec.get()
            if item is STOP:
                for d, eng in self.engines.items():
                    for out in eng.flush():
                        self._publish(out)
                self._publish(None)
                return
            idx, t_in, rows, ball, H = item
            t0 = time.perf_counter()
            fr = {"frame": idx, "image_id": idx, "rows": rows, "ball": ball}
            outs = [o for eng in self.engines.values() for o in eng.push(fr)]
            now = time.perf_counter()
            self.stats.stage["decide"].append((now - t0) * 1000)
            self.stats.lat.append((now - t_in) * 1000)
            self.stats.done.append(now)
            self.stats.frames += 1
            for o in outs:
                o["H"] = H.tolist() if (H is not None and o["delay"] == 0) else None
                self._publish(o)

    def _publish(self, out):
        for fn in list(self.subscribers):
            try:
                fn(out)
            except Exception:
                import traceback
                traceback.print_exc()
