"""Video sources for the live pipeline.

Every source yields (frame_index, bgr_frame) at the pipeline's working rate
(25 FPS, the rate every frame-based parameter was tuned at) and working size
(1920x1080, what the calibrator expects). frame_index counts working-rate
frames since the start, including frames dropped because the pipeline fell
behind, so time gaps stay physical.

  file.mp4 / file.mkv          a video file; real-time pacing by default
  path/to/img1/ (jpg folder)   an image sequence, e.g. a SoccerNet clip
  rtsp:// rtmp:// http(s)://   a network stream (opened with FFmpeg)
  youtube / twitch page URL    resolved to a stream URL with yt-dlp
"""
import shutil
import subprocess
import time
from pathlib import Path

import cv2

RATE = 25.0
SIZE = (1920, 1080)
PAGE_HOSTS = ("youtube.com", "youtu.be", "twitch.tv")


def resolve(url):
    """Page URL -> direct media URL via yt-dlp (best video stream up to 1080p)."""
    exe = shutil.which("yt-dlp")
    if exe is None:
        raise RuntimeError("yt-dlp is needed for page URLs: pip install yt-dlp")
    r = subprocess.run([exe, "-g", "-f", "best[height<=1080][vcodec!=none]/best", url],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RuntimeError(f"yt-dlp could not resolve {url}: {r.stderr.strip()[-300:]}")
    return r.stdout.strip().splitlines()[0]


class Source:
    def __init__(self, spec, realtime=None, loop=False, max_frames=0):
        self.spec = str(spec)
        self.loop = loop
        self.max_frames = max_frames
        self.is_stream = "://" in self.spec
        self.realtime = self.is_stream if realtime is None else realtime
        self.images = None
        self.fps = RATE
        p = Path(self.spec)
        if p.is_dir():
            d = p / "img1" if (p / "img1").is_dir() else p
            self.images = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))
            if not self.images:
                raise FileNotFoundError(f"no images in {d}")
        else:
            url = resolve(self.spec) if any(h in self.spec for h in PAGE_HOSTS) else self.spec
            if not self.is_stream and not p.exists():
                raise FileNotFoundError(self.spec)
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open {self.spec}")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if 1 < fps < 241 else RATE
        self.stopped = False

    @property
    def name(self):
        return Path(self.spec).name if not self.is_stream else self.spec

    def _raw(self):
        """-> (source_time_seconds, frame) for every decoded frame."""
        while not self.stopped:
            if self.images is not None:
                for i, p in enumerate(self.images):
                    if self.stopped:
                        return
                    yield i / RATE, cv2.imread(str(p))
            else:
                i = 0
                while not self.stopped:
                    ok, fr = self.cap.read()
                    if not ok:
                        break
                    ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    yield (ms / 1000 if ms > 0 else i / self.fps), fr
                    i += 1
                if self.loop and not self.is_stream:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if not self.loop or self.is_stream:
                return

    def frames(self):
        """Working-rate frames. With realtime pacing, frames are released on the
        wall clock; the consumer decides what to drop when it falls behind."""
        t0, offset, last_idx, n = None, 0.0, -1, 0
        prev_ts = -1.0
        for ts, fr in self._raw():
            if fr is None:
                continue
            if ts < prev_ts:                      # file looped: keep time increasing
                offset += prev_ts + 1 / self.fps
            prev_ts = ts
            t = ts + offset
            idx = int(round(t * RATE))
            if idx <= last_idx:                   # downsample sources faster than 25 FPS
                continue
            last_idx = idx
            if self.realtime and not self.is_stream:
                if t0 is None:
                    t0 = time.perf_counter() - t
                wait = t0 + t - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            if fr.shape[1] != SIZE[0] or fr.shape[0] != SIZE[1]:
                fr = cv2.resize(fr, SIZE, interpolation=cv2.INTER_AREA if fr.shape[1] > SIZE[0]
                                else cv2.INTER_LINEAR)
            yield idx, fr
            n += 1
            if self.max_frames and n >= self.max_frames:
                return

    def stop(self):
        self.stopped = True
