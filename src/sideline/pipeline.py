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

from .calibration_guard import CalibrationGuard, filter_foot_points  # noqa: E402
from .decisions import DecisionEngine  # noqa: E402
from .geometry import foot_points, project, torso_feature  # noqa: E402
from .sources import RATE, SIZE, Source  # noqa: E402

NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
STOP = object()


class SharedCMC:
    """Stand-in for the tracker's camera motion compensation: returns the flow estimate."""
    H = np.eye(2, 3, dtype=np.float32)

    def estimate(self, frame, dets_xyxy=None):
        return self.H

    def reset(self):
        self.H = np.eye(2, 3, dtype=np.float32)


class _CalibState:
    """Calibration state carried across cpu-stage frames.

    The per-frame motions (used to propagate a keyframe homography forward to
    the current frame), the carried homography, keyframe scheduling and the
    guard bookkeeping.  Plain state plus small methods, so the guard wiring
    can be exercised without a video or a calibrator process.
    """

    def __init__(self, keyframe_k):
        self.keyframe_k = keyframe_k
        self.motions = {}            # frame idx -> 2x3 motion, or None
        self.H_cur = None            # carried homography, propagated per frame
        self.H_from = None           # keyframe idx the carry started from
        self.last_req = None         # frame idx of the last keyframe request
        self.last_offered = None     # last keyframe idx offered to the guard
        self.H_last = None           # last accepted homography (feeds the output)
        self.strict = False          # foot filter on: last acceptance was by the full rule

    def motion(self, idx, M):
        """Record one frame's motion estimate, pruning what is no longer needed."""
        self.motions[idx] = M
        for j in [j for j in self.motions if j < idx - 4 * self.keyframe_k - 10]:
            del self.motions[j]

    def carry(self, M):
        """Advance the carried homography one frame.  A missing motion is a
        one-frame propagation miss, not a lost calibration, so H_cur is kept."""
        if self.H_cur is not None and M is not None:
            self.H_cur = self.H_cur @ M


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
        self.rejected_calibrations = self.accepted_calibrations = 0

    def snapshot(self):
        d = list(self.done)
        fps = (len(d) - 1) / (d[-1] - d[0]) if len(d) > 1 and d[-1] > d[0] else 0.0
        lat = np.asarray(self.lat) if self.lat else np.zeros(1)
        return {"fps": round(fps, 1), "latency_ms_p50": round(float(np.median(lat)), 1),
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
                "frames": self.frames, "dropped": self.dropped, "keyframes": self.keyframes,
                "calibrated": self.calibrated,
                "rejected_calibrations": self.rejected_calibrations,
                "accepted_calibrations": self.accepted_calibrations,
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
        self.cal_guard = CalibrationGuard()
        self.cal = _CalibState(self.keyframe_k)
        self.cal_results, self.cal_lock, self.cal_sent, self.cal_pending = {}, threading.Lock(), {}, 0
        self.running = True
        stages = [self._source, self._gpu, self._cpu, self._jersey, self._decide]
        if self.m.calib and not self.sync_calib:
            # in sync mode _cpu is the sole consumer of the calibrator queue
            # (blocking homography()); a second draining stage would race it
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
                        continue
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
        cal, prev_boxes = self.cal, None
        while True:
            item = self.q_cpu.get()
            if item is STOP:
                return self._put(self.q_jer, STOP)
            idx, frame, t_in, (xyxy, conf, cls), pose = item
            t0 = time.perf_counter()
            M = prop.motion(frame, prev_boxes)
            cal.motion(idx, M)
            shared.H = prop.cmc_affine(M)
            cal.carry(M)
            people = cls != 3
            out = trk.update(sv.Detections(xyxy=xyxy[people].astype(np.float32),
                                           confidence=conf[people].astype(np.float32),
                                           class_id=cls[people]), frame)
            prev_boxes = out.xyxy
            if self.m.calib:
                self._schedule_keyframe(idx, frame, M is not None)
                picked = self._pick_next_calibration(idx)
                if picked is not None:
                    Hk, k = picked
                    H_new, dec = self._apply_guard(Hk, cal.H_cur, M is not None)
                    cal.last_offered = k                       # a keyframe is offered once
                    if dec in ("accepted_first", "accepted", "accepted_confirmed"):
                        cal.H_cur, cal.H_from = H_new, k
                        cal.strict = dec != "accepted_first"   # lenient first: no foot filtering
                    else:  # rejected_jump or rejected_invalid: ask again next frame
                        cal.last_req = idx - self.keyframe_k + 1
            self.stats.calibrated = cal.H_cur is not None
            if cal.H_cur is not None:
                cal.H_last = cal.H_cur
            H_use = cal.H_last                               # last good calibration until a new one
            rows = self._project_people(frame, out, H_use, cal.strict)
            ball = self._project_ball(xyxy, conf, people, H_use)
            self.stats.stage["cpu"].append((time.perf_counter() - t0) * 1000)
            self._put(self.q_jer, (idx, frame, t_in, rows, pose, ball, H_use))

    def _schedule_keyframe(self, idx, frame, moving):
        """Request (or, in sync mode, run) a keyframe calibration when due: the
        first frame, every keyframe_k frames, or while no calibration is
        carried and no motion was estimated this frame."""
        st = self.cal
        if not (st.last_req is None or idx - st.last_req >= self.keyframe_k
                or (st.H_cur is None and not moving)):
            return
        small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        if self.sync_calib:
            with self.cal_lock:
                self.cal_results[idx] = self.m.calib.homography(small, (self.run_id, idx))
            self.stats.keyframes += 1
            st.last_req = idx
        elif self.m.calib.request((self.run_id, idx), small):
            self.cal_sent[idx] = time.perf_counter()
            self.cal_pending += 1
            st.last_req = idx

    def _pick_next_calibration(self, idx):
        """The newest keyframe result not yet offered to the guard,
        propagated to frame idx with the recorded motions: (H_candidate, k)
        or None when none is offerable yet.  A motion gap inside the window
        stops the propagation at the gap instead of discarding the
        candidate: a sustained flow failure (blank or static frames, a long
        camera cut) must not withhold a valid calibration forever, and
        cal.carry() resumes the carry once motion returns."""
        with self.cal_lock:
            off = self.cal.last_offered
            done = [k for k, H in self.cal_results.items()
                    if H is not None and (off is None or k > off)]
            k = max(done) if done else None
            Hk = self.cal_results[k] if k is not None else None
            for old in [j for j in self.cal_results if j < idx - 4 * self.keyframe_k]:
                del self.cal_results[old]
        if k is None:
            return None
        H = Hk
        for j in range(k + 1, idx + 1):
            if j not in self.cal.motions:        # dropped frame: no motion recorded
                continue
            if self.cal.motions[j] is None:
                break                            # motion gap: offer up to the gap
            H = H @ self.cal.motions[j]
        return H, k

    def _apply_guard(self, H_candidate, H_cur, moving):
        """Offer one keyframe homography to the guard; returns (H_to_use,
        decision) and updates the calibration counters."""
        H_new, dec = self.cal_guard.update(H_candidate, H_cur, SIZE, moving=moving)
        if dec in ("accepted_first", "accepted", "accepted_confirmed"):
            self.stats.accepted_calibrations += 1
        else:                                    # rejected_invalid / rejected_jump
            self.stats.rejected_calibrations += 1
        return H_new, dec

    def _project_people(self, frame, out, H_use, strict=True):
        """One output row per tracked box: feet on the pitch (None when any
        foot is outside the pitch extended by 15 m, but only while the
        current calibration was accepted by the full rule; a lenient first
        acceptance keeps everyone on the minimap) and the torso feature."""
        rows = []
        if out.tracker_id is None:
            return rows
        for b, tid, c, sc in zip(out.xyxy, out.tracker_id, out.class_id, out.confidence):
            if tid is None or tid < 0:
                continue
            x1, y1, x2, y2 = map(float, b)
            f = torso_feature(frame, x1, y1, x2 - x1, y2 - y1)
            pitch = foot_points(H_use, x1, y1, x2 - x1, y2 - y1) if H_use is not None else None
            if pitch is not None and strict and not all(filter_foot_points(pitch)):
                pitch = None                     # a foot outside pitch + 15 m
            rows.append({"track": int(tid), "cls": NAMES[int(c)], "score": float(sc),
                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "pitch": pitch,
                         "feat": None if f is None else f.tolist()})
        return rows

    def _project_ball(self, xyxy, conf, people, H_use):
        ball = None
        bm = np.flatnonzero(~people)
        if len(bm):
            b = xyxy[bm[int(conf[bm].argmax())]]
            ball = {"bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], "pitch": None}
            if H_use is not None:
                ball["pitch"] = project(H_use, [((b[0] + b[2]) / 2, b[3])])[0].tolist()
        return ball

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
