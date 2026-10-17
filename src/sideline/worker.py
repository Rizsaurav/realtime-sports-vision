"""Video worker: turn uploaded videos into recorded match files.

A small FastAPI service for the GPU machine. It accepts video uploads,
keeps them in a FIFO queue and processes one at a time with
``python -m sideline run`` in a subprocess, then serves the resulting
JSON lines file. Only the standard library plus FastAPI is imported
here; the heavy pipeline (torch, cv2, numpy) runs in the subprocess, so
importing this module stays light.

Security and observability settings (see ``create_worker_app``):

* ``token``: when set, every route except ``GET /health`` requires
  ``Authorization: Bearer <token>`` (compared with
  ``hmac.compare_digest``; wrong or missing answers 401 with an
  ``{"error": ...}`` body). The CLI form is ``python -m sideline worker
  --token-file FILE``: the file holds one token of 32+ random characters
  and must be mode 600 (refused at start otherwise); without a token the
  command refuses to bind any host other than 127.0.0.1.
* Structured log: one JSON line per request on stdout with ``ts``,
  ``method``, ``path`` (no query string), ``status``, ``ms``,
  ``bytes_in`` and the request id. The id comes from the
  ``X-Request-Id`` request header when present (and sane), else it is
  generated, and it is echoed in the response header. The line never
  carries headers (no tokens), uploaded file names or bodies.
* Metrics: ``GET /metrics`` serves Prometheus text format - requests by
  status class, uploads and uploaded bytes, jobs by state, queue depth
  and seconds spent processing. Counters only go up within a process.
* Rate limit: at most 60 non-GET requests per minute per client, 429
  with a ``Retry-After`` header when exceeded.
* Upload size: uploads are capped at ``max_bytes`` (2 GiB by default,
  ``--max-gb``); the size is enforced while streaming to disk and the
  free disk space is checked before a job is accepted.
* Graceful shutdown: ``Worker.shutdown()`` (called by the CLI on
  SIGTERM/SIGINT) stops accepting jobs (new uploads answer 503), kills
  the running job's process group, marks the job failed with
  "worker stopped", persists the state and joins the processing thread,
  so the process exits within 10 seconds. Jobs left queued stay on disk
  and are picked up again on the next start.

The shared pieces (``Metrics``, ``RateLimiter``, the bearer-token check
and the request-log / rate-limit / auth middleware) are used by the
library app as well.
"""

import collections
import hmac
import inspect
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.datastructures import UploadFile

logger = logging.getLogger("sideline.worker")

__all__ = ["create_worker_app", "Worker", "Metrics", "RateLimiter",
           "bearer_matches", "install_request_log", "install_rate_limit",
           "install_auth"]

_ALLOWED_EXTS = (".avi", ".mkv", ".mov", ".mp4", ".webm")
_JOB_ID_RE = re.compile(r"j_[0-9a-f]{12}\Z")
_CHUNK = 1024 * 1024
_POLL_SECONDS = 2.0
_TIME_LIMIT = 2 * 60 * 60
_KEEP_SECONDS = 24 * 60 * 60
_FPS = 25
_DEFAULT_MAX_BYTES = 2 * 1024 ** 3
_FINISHED = ("done", "failed")


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(stamp):
    """Epoch seconds for an ISO-8601 UTC stamp, or None when unreadable."""
    if not isinstance(stamp, str):
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _count_lines(path):
    try:
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _new_lines(path, offset, total):
    """Advance the line count of ``path`` from ``offset`` bytes on.

    Returns (offset, total): the new offset (the file's current size) and
    the total number of lines - counting only the bytes appended since
    ``offset`` instead of re-reading the whole file on every poll (the
    result of a long job is tens of MB). A file that shrank (truncated or
    replaced) is counted from the start. A missing file keeps the state.
    """
    try:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size < offset:
                offset, total = 0, 0
            if size == offset:
                return offset, total
            fh.seek(offset)
            return size, total + sum(1 for _ in fh)
    except OSError:
        return offset, total


def _tail(path, lines=5):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:]).strip()
    except OSError:
        return ""


def _kill(proc):
    """Kill the processing process and its whole process group.

    The default runner starts the child in its own session, so the group
    kill also reaches the child's children (ffprobe, decoders); a process
    object without a pid (a test double) is killed directly.
    """
    pid = getattr(proc, "pid", None)
    if pid is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError as exc:
            logger.debug("the process group of %s is gone; falling back "
                         "to the process: %s", pid, exc)
    try:
        proc.kill()
    except OSError as exc:
        logger.warning("could not send kill to the processing process: %s", exc)
    try:
        proc.wait(timeout=10)
    except Exception as exc:
        logger.warning("the processing process did not exit within 10 s "
                       "after the kill: %s", exc)


# ---------------------------------------------------------------------------
# shared hardening primitives (also used by the library app)
# ---------------------------------------------------------------------------

def bearer_matches(header_value, token):
    """True when the Authorization header carries the token.

    Compared in constant time with hmac.compare_digest; a missing, empty
    or mistyped header is never equal.
    """
    if not isinstance(header_value, str) or not isinstance(token, str) or not token:
        return False
    return hmac.compare_digest(header_value.encode("utf-8"),
                               ("Bearer " + token).encode("utf-8"))


class Metrics:
    """A small in-process registry rendered as Prometheus text format.

    Counters only go up (``inc``); gauges reflect the current state
    (``set``). Thread-safe. ``render`` returns one ``# HELP``/``# TYPE``
    pair per metric name and one line per label set, in the Prometheus
    text exposition format (version 0.0.4).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._values = {}
        self._names = {}

    def _ensure(self, name, mtype, help_text):
        with self._lock:
            if name not in self._names:
                self._names[name] = (help_text, mtype)

    def inc(self, name, amount=1.0, help_text=None, **labels):
        """Add to a counter (counters never go down within a process)."""
        self._ensure(name, "counter", help_text or name)
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def set(self, name, value, help_text=None, **labels):
        """Set a gauge to its current value."""
        self._ensure(name, "gauge", help_text or name)
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._values[key] = float(value)

    @staticmethod
    def _format(value):
        value = float(value)
        return str(int(value)) if value.is_integer() else repr(value)

    def render(self):
        """The registry as Prometheus text (a trailing newline included)."""
        with self._lock:
            values = sorted(self._values.items(), key=lambda kv: (kv[0][0], kv[0][1]))
            names = dict(self._names)
        lines = []
        emitted = set()
        for (name, labels), value in values:
            if name not in emitted:
                help_text, mtype = names[name]
                lines.append("# HELP %s %s" % (name, help_text))
                lines.append("# TYPE %s %s" % (name, mtype))
                emitted.add(name)
            if labels:
                rendered = ",".join('%s="%s"' % (key, label) for key, label in labels)
                lines.append("%s{%s} %s" % (name, rendered, self._format(value)))
            else:
                lines.append("%s %s" % (name, self._format(value)))
        return "\n".join(lines) + "\n"


class RateLimiter:
    """A sliding-window limit of ``limit`` requests per ``window`` seconds
    per client key.

    ``allow`` returns ``(True, 0)`` for a request that fits the window,
    else ``(False, retry_after)`` where retry_after is the whole number
    of seconds until the oldest counted request leaves the window (the
    value for the Retry-After header). Idle clients are forgotten.
    Thread-safe.
    """

    def __init__(self, limit=60, window=60.0):
        self.limit = limit
        self.window = window
        self._lock = threading.Lock()
        self._hits = {}

    def allow(self, key, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = collections.deque()
            cutoff = now - self.window
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                self._hits[key] = hits
                retry_after = max(1, math.ceil(self.window - (now - hits[0])))
                return False, retry_after
            hits.append(now)
            self._hits[key] = hits
            return True, 0


_REQUEST_ID_RE = re.compile(r"[ -~]{1,128}\Z")


def _request_id_from(header_value):
    """The X-Request-Id to use: the header when it is a sane string
    (printable, 1-128 chars, no whitespace), else a generated id."""
    if isinstance(header_value, str) and _REQUEST_ID_RE.match(header_value):
        return header_value
    return secrets.token_hex(16)


def _emit_request_log(request, status, seconds, request_id):
    """One JSON line per request on stdout.

    Fields: ts (UTC), method, path without the query string, status, ms
    (elapsed), bytes_in (the Content-Length, 0 when absent) and the
    request id. The line never carries request headers (so no tokens),
    uploaded file names or bodies.
    """
    try:
        bytes_in = int(request.headers["content-length"])
    except (KeyError, ValueError):
        bytes_in = 0
    record = {"ts": _now_iso(),
              "method": request.method,
              "path": request.url.path,
              "status": status,
              "ms": int(round(seconds * 1000)),
              "bytes_in": bytes_in,
              "request_id": request_id}
    # straight to the current stdout (no logging format around it), so
    # the line is plain JSON wherever the process's stdout is pointed
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()


def install_request_log(app, metrics):
    """Middleware: the JSON request log plus the X-Request-Id echo, and
    the requests-by-status-class counter."""

    @app.middleware("http")
    async def _log_request(request, call_next):
        request_id = _request_id_from(request.headers.get("x-request-id"))
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.monotonic() - started
            _emit_request_log(request, 500, elapsed, request_id)
            metrics.inc("sideline_requests", 1, "Requests by status class",
                        **{"class": "5xx"})
            raise
        elapsed = time.monotonic() - started
        _emit_request_log(request, response.status_code, elapsed, request_id)
        metrics.inc("sideline_requests", 1, "Requests by status class",
                    **{"class": "%dxx" % (response.status_code // 100)})
        response.headers["x-request-id"] = request_id
        return response


def install_rate_limit(app, limit=60, window=60.0):
    """Middleware: at most ``limit`` non-GET requests per ``window``
    seconds per client; the 429 answer carries a Retry-After header."""
    limiter = RateLimiter(limit, window)

    @app.middleware("http")
    async def _rate_limit(request, call_next):
        if request.method in ("GET", "HEAD"):
            return await call_next(request)
        client = request.client.host if request.client is not None else "unknown"
        allowed, retry_after = limiter.allow(client)
        if not allowed:
            return JSONResponse(
                {"error": "rate limit exceeded: at most %d non-GET requests "
                          "per minute; try again in %d seconds" % (limit, retry_after)},
                status_code=429, headers={"Retry-After": str(retry_after)})
        return await call_next(request)


def install_auth(app, token, exempt):
    """Middleware: bearer-token authentication.

    ``exempt`` is a predicate on the request; every non-exempt request
    needs ``Authorization: Bearer <token>`` and anything else answers
    401 with an ``{"error": ...}`` body.
    """

    @app.middleware("http")
    async def _auth(request, call_next):
        if not exempt(request):
            if not bearer_matches(request.headers.get("authorization"), token):
                return JSONResponse(
                    {"error": "unauthorized: a valid bearer token is required "
                              "(Authorization: Bearer ...)"},
                    status_code=401)
        return await call_next(request)



def _probe_duration(path):
    """Video duration in seconds via ffprobe, or None when it cannot be read."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _runner_takes_cancel(runner):
    """True when an injected runner can receive the job's cancel event.

    A runner that declares a ``cancel`` parameter (or accepts **kwargs)
    is called with ``cancel=<threading.Event>``; every other runner gets
    the plain three-argument call.
    """
    try:
        params = inspect.signature(runner).parameters
    except (TypeError, ValueError):
        return False
    if "cancel" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params.values())


def _display_name(name, filename):
    """The job's display name: the sanitised ``name`` field when given, else
    the stem of the uploaded file name (its directory parts dropped). The
    client's file name never reaches a path or a command line."""
    if name is not None:
        cleaned = str(name).replace("\\", "/").rsplit("/", 1)[-1].strip()
        if cleaned:
            return cleaned
    stem = Path(str(filename or "").replace("\\", "/")).name
    return Path(stem).stem or "video"


def _default_runner(cancel, register=None):
    """Build the standard runner: one ``sideline run`` subprocess per job.

    Progress is read from the number of lines in the output file (polled
    every two seconds; each poll reads only the bytes appended since the
    previous one, so a long job never re-scans its whole result file).
    The child is killed - with its whole process group, in its own
    session - when the job is cancelled or when the total time limit of
    two hours is exceeded. ``register`` (called with the Popen object on
    spawn and with None on exit) lets the worker keep the running process
    so a shutdown can kill the group.
    """

    def run(video, out, on_progress):
        env = dict(os.environ)
        src = Path(__file__).resolve().parent.parent
        env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, "-m", "sideline", "run", str(video),
               "--delay", "5", "--out", str(out)]
        log_path = Path(out).with_name("run.log")
        try:
            with open(log_path, "wb") as log:
                proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                        stderr=log, start_new_session=True)
        except OSError as exc:
            raise RuntimeError("could not start the processing command: %s" % exc)
        if register is not None:
            register(proc)
        started = time.monotonic()
        last = -1
        offset = 0
        lines = 0
        try:
            while proc.poll() is None:
                time.sleep(_POLL_SECONDS)
                if cancel.is_set():
                    _kill(proc)
                    return
                if time.monotonic() - started > _TIME_LIMIT:
                    _kill(proc)
                    raise RuntimeError("processing exceeded the 2 hour time limit")
                offset, lines = _new_lines(out, offset, lines)
                if lines != last:
                    last = lines
                    on_progress(lines)
            if proc.returncode != 0:
                tail = _tail(log_path)
                raise RuntimeError("processing failed (exit %d)%s"
                                   % (proc.returncode, (": " + tail) if tail else ""))
            offset, lines = _new_lines(out, offset, lines)
            if lines != last:
                on_progress(lines)
        finally:
            if proc.poll() is None:
                _kill(proc)
            if register is not None:
                register(None)

    return run


class _UploadError(Exception):
    """A rejected upload with its HTTP status and the client-facing message."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class _Job:
    def __init__(self, job_id, name, created_at, ext):
        self.id = job_id
        self.name = name
        self.created_at = created_at
        self.ext = ext
        self.status = "queued"
        self.progress = 0.0
        self.frames = 0
        self.seconds = None
        self.error = None
        self.finished_at = None
        self.cancel = threading.Event()
        # Not part of the state file: the stored size of the upload (for
        # the upload-bytes metric) and the shutdown flag (a job failed
        # with "worker stopped" must not be overwritten by the late
        # answer of its runner).
        self.bytes = 0
        self.stopped = False

    def to_dict(self):
        return {
            "job_id": self.id,
            "status": self.status,
            "progress": self.progress,
            "frames": self.frames,
            "name": self.name,
            "error": self.error,
            "created_at": self.created_at,
            "seconds": self.seconds,
        }

    def to_state(self):
        state = dict(self.to_dict())
        state["ext"] = self.ext
        state["finished_at"] = self.finished_at
        return state

    @classmethod
    def from_state(cls, state):
        """Rebuild a job from its job.json; raises (OSError, ValueError,
        TypeError, KeyError) on a malformed state file."""
        job = cls(state["job_id"], state.get("name") or state["job_id"],
                  state.get("created_at") or _now_iso(), state.get("ext") or ".mp4")
        job.status = state.get("status") or "queued"
        job.progress = float(state.get("progress") or 0.0)
        job.frames = int(state.get("frames") or 0)
        job.seconds = state.get("seconds")
        job.error = state.get("error")
        job.finished_at = state.get("finished_at")
        return job


class Worker:
    """The job queue, the per-job state on disk and one processing thread.

    ``metrics`` (a ``Metrics`` registry, optional) receives the seconds
    spent processing one job each time a job finishes.
    """

    def __init__(self, root, runner=None, max_bytes=_DEFAULT_MAX_BYTES, metrics=None):
        if max_bytes is None:
            max_bytes = _DEFAULT_MAX_BYTES
        max_bytes = int(max_bytes)
        if max_bytes <= 0:
            raise ValueError("max_bytes must be a positive number of bytes")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.metrics = metrics
        self._runner = runner
        self._pass_cancel = runner is not None and _runner_takes_cancel(runner)
        self._jobs = {}
        self._queue = collections.deque()
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._running_id = None
        self._running_proc = None
        self._stopping = False
        self._thread = None
        self._recover()

    # -- paths -------------------------------------------------------------

    def job_dir(self, job_id):
        return self.root / job_id

    def _video_path(self, job):
        return self.job_dir(job.id) / (job.id + job.ext)

    def _result_path(self, job):
        return self.job_dir(job.id) / "result.jsonl"

    def _state_path(self, job_id):
        return self.job_dir(job_id) / "job.json"

    @staticmethod
    def _atomic_write(path, data):
        path = Path(path)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError as exc:
                logger.debug("could not remove the temporary file %s: %s",
                             tmp, exc)
            raise

    def _write_state(self, job):
        """Persist the job's state atomically (tempfile + rename).

        A failed write is logged and, when the job still exists, turns it
        into a failed one, so the on-disk and in-memory views cannot
        silently diverge (after a crash a job whose state could not be
        saved would otherwise resurrect from a stale state file).
        """
        with self._lock:
            if job.id not in self._jobs:
                return  # the job was removed concurrently: nothing to persist
            data = (json.dumps(job.to_state(), indent=2, sort_keys=True) + "\n").encode("utf-8")
            path = self._state_path(job.id)
        try:
            self._atomic_write(path, data)
            return
        except OSError as exc:
            if not path.parent.is_dir():
                # the job folder was removed concurrently (delete/purge)
                logger.debug("state write for job %s skipped, the job "
                             "folder is gone: %s", job.id, exc)
                return
            logger.error("could not persist the state of job %s: %s", job.id, exc)
            with self._lock:
                if job.id not in self._jobs:
                    return  # deleted in the meantime
                if job.status != "failed":
                    job.status = "failed"
                    job.error = "state could not be persisted: %s" % exc
                    job.finished_at = _now_iso()
            try:
                data = (json.dumps(job.to_state(), indent=2, sort_keys=True)
                        + "\n").encode("utf-8")
                self._atomic_write(path, data)
            except OSError as retry_exc:
                logger.error("could not persist the failure state of job %s: %s",
                             job.id, retry_exc)

    # -- queue lifecycle ---------------------------------------------------

    def start(self):
        """Start the background processing thread (idempotent)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout=5.0):
        """Stop the background thread (for tests and clean shutdown)."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _loop(self):
        while True:
            with self._cond:
                while not self._queue:
                    if self._stopping:
                        return
                    self._cond.wait(timeout=0.2)
                if self._stopping:
                    return  # queued jobs stay on disk; a restart picks them up
                job_id = self._queue.popleft()
            job = self._jobs.get(job_id)
            if job is None:
                continue  # deleted while waiting
            self._run(job)

    def _register_proc(self, proc):
        with self._lock:
            self._running_proc = proc

    def _run(self, job):
        with self._cond:
            job.status = "running"
            job.error = None
            self._running_id = job.id
        self._write_state(job)
        runner = self._runner
        if runner is None:
            runner = _default_runner(job.cancel, register=self._register_proc)
        started = time.monotonic()
        try:
            try:
                if self._pass_cancel:
                    runner(self._video_path(job), self._result_path(job),
                           lambda frames: self._report(job, frames),
                           cancel=job.cancel)
                else:
                    runner(self._video_path(job), self._result_path(job),
                           lambda frames: self._report(job, frames))
            except Exception as exc:
                self._finish(job, ok=False,
                             error=str(exc) or exc.__class__.__name__)
            else:
                self._finish(job, ok=True, frames=_count_lines(self._result_path(job)))
        finally:
            with self._cond:
                if self._running_id == job.id:
                    self._running_id = None
            if self.metrics is not None:
                self.metrics.inc("sideline_processing_seconds",
                                 time.monotonic() - started,
                                 "Seconds spent processing jobs")

    def _report(self, job, frames):
        try:
            frames = int(frames)
        except (TypeError, ValueError):
            logger.debug("ignoring non-numeric progress report %r for job %s",
                         frames, job.id)
            return
        if frames < 0:
            frames = 0
        with self._cond:
            if job.id not in self._jobs or job.stopped:
                return
            job.frames = frames
            if job.seconds:
                total = int(job.seconds * _FPS)
                job.progress = max(0.0, min(1.0, frames / total)) if total > 0 else 1.0
        self._write_state(job)

    def _finish(self, job, ok, error=None, frames=None):
        with self._cond:
            if job.id not in self._jobs:
                return  # deleted (cancelled) while running: drop the result
            if job.stopped:
                return  # the shutdown already failed it with "worker stopped"
            if ok:
                job.status = "done"
                job.progress = 1.0
                job.error = None
                if frames:
                    job.frames = frames
                if job.seconds is None:
                    job.seconds = round(job.frames / _FPS, 2)
            else:
                job.status = "failed"
                job.error = error
            job.finished_at = _now_iso()
        self._write_state(job)

    def shutdown(self, timeout=8.0):
        """Stop the worker for a process exit (the CLI calls it on SIGTERM).

        New uploads answer 503 from this point on, the running job's
        process group is killed, that job is marked failed with
        "worker stopped" and persisted, queued jobs stay on disk (a
        restart picks them up), and the processing thread is joined
        within ``timeout`` seconds.
        """
        running = None
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
            if self._running_id is not None:
                running = self._jobs.get(self._running_id)
        with self._lock:
            proc = self._running_proc
        if running is not None:
            running.stopped = True
            running.cancel.set()
        if proc is not None:
            _kill(proc)
        if running is not None:
            with self._cond:
                if running.status == "running":
                    running.status = "failed"
                    running.error = "worker stopped"
                    running.finished_at = _now_iso()
            self._write_state(running)
        self.stop(timeout=timeout)

    # -- upload ------------------------------------------------------------

    def _new_id(self):
        while True:
            job_id = "j_" + secrets.token_hex(6)
            with self._lock:
                if job_id not in self._jobs and not self.job_dir(job_id).exists():
                    return job_id

    def _purge_old(self):
        """Remove finished jobs older than 24 hours (lazy, on each upload)."""
        cutoff = time.time() - _KEEP_SECONDS
        with self._lock:
            stale = [job for job in self._jobs.values()
                     if job.status in _FINISHED
                     and (_parse_iso(job.finished_at or job.created_at) or time.time()) < cutoff]
        for job in stale:
            with self._lock:
                self._jobs.pop(job.id, None)
                if job.id in self._queue:
                    self._queue.remove(job.id)
            shutil.rmtree(self.job_dir(job.id), ignore_errors=True)

    async def upload(self, file, name=None):
        """Store one uploaded video and queue its job; raises _UploadError.

        The bytes are streamed to disk in chunks and the stored file name
        is generated; the client's file name only supplies the extension
        (and, sanitised, the display name). While the worker is stopping
        (a process exit), new jobs are not accepted.
        """
        if self._stopping:
            raise _UploadError(503, "the worker is stopping and is not "
                                    "accepting new jobs; try again later")
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()
        if ext not in _ALLOWED_EXTS:
            raise _UploadError(400, "file extension must be one of: %s; "
                                     "choose a video file" % " ".join(_ALLOWED_EXTS))
        free = shutil.disk_usage(self.root).free
        if free < self.max_bytes:
            raise _UploadError(507, "not enough free disk space on the worker "
                                     "(%d bytes free, the limit is %d bytes); "
                                     "free up space or choose a smaller recording"
                                     % (free, self.max_bytes))
        self._purge_old()
        job = _Job(self._new_id(), _display_name(name, filename), _now_iso(), ext)
        self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
        video = self._video_path(job)
        size = 0
        try:
            with open(video, "wb") as dst:
                while True:
                    chunk = await file.read(_CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise _UploadError(413,
                                           "file exceeds the maximum size of %d bytes; "
                                           "choose a smaller recording" % self.max_bytes)
                    dst.write(chunk)
        except _UploadError:
            shutil.rmtree(self.job_dir(job.id), ignore_errors=True)
            raise
        except OSError as exc:
            # the write failed mid-stream (e.g. the disk filled up): remove
            # the partial job and report a proper error instead of leaving
            # an orphaned job folder behind
            shutil.rmtree(self.job_dir(job.id), ignore_errors=True)
            raise _UploadError(507, "the upload could not be written to disk "
                                     "(%s); free up space and try again" % exc)
        except Exception as exc:
            # a client disconnect or other I/O surprise mid-stream
            shutil.rmtree(self.job_dir(job.id), ignore_errors=True)
            raise _UploadError(500, "the upload could not be stored: %s" % exc)
        if size == 0:
            shutil.rmtree(self.job_dir(job.id), ignore_errors=True)
            raise _UploadError(400, "file is empty; choose a video file with content")
        # probe in a worker thread: ffprobe can take up to 30 s and must not
        # block the event loop (health checks, other uploads, job lookups)
        job.seconds = await anyio.to_thread.run_sync(_probe_duration, video)
        job.bytes = size
        with self._cond:
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._cond.notify_all()
        self._write_state(job)
        return job

    # -- access ------------------------------------------------------------

    def get(self, job_id):
        """The job object, or None for unknown or malformed ids (the file
        system is never consulted for a malformed id)."""
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            return None
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id):
        """Remove a job (video, result and state); a running job is cancelled.

        Cancellation sets the job's cancel event: the default runner kills
        its subprocess on the next poll, and an injected runner that
        declares a ``cancel`` parameter receives the event and should
        return promptly so the queue can move on.

        Returns True when a job was removed.
        """
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            return False
        with self._cond:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return False
            if job_id in self._queue:
                self._queue.remove(job_id)
            job.cancel.set()
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        return True

    # -- startup recovery ---------------------------------------------------

    def _recover(self):
        """Load the jobs left on disk by an earlier process.

        A job still ``running`` was interrupted by the restart and is
        marked failed with a clear error; a job still ``queued`` is put
        back in the FIFO queue.
        """
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir() or not _JOB_ID_RE.match(entry.name):
                continue
            try:
                state = json.loads((entry / "job.json").read_text(encoding="utf-8"))
                if not isinstance(state, dict):
                    continue
                if state.get("job_id") != entry.name:
                    continue
                job = _Job.from_state(state)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                # one corrupted state file must not stop the service from
                # starting: skip (and log) the malformed entry
                logger.warning("skipping malformed job state in %s: %s",
                               entry.name, exc)
                continue
            if job.status == "running":
                job.status = "failed"
                job.error = "interrupted by a worker restart; the job did not finish. " \
                            "Delete the job and upload the video again."
                job.finished_at = _now_iso()
                self._jobs[job.id] = job
                self._write_state(job)
            else:
                self._jobs[job.id] = job
                if job.status == "queued":
                    self._queue.append(job.id)


async def _create_job(request, worker, metrics):
    """The POST /jobs logic; the route wires in the worker and metrics.

    Stores the uploaded video (the extension and display name come from
    the client's file name, sanitised) and queues its job; answers 202
    with the job id, else the worker's own error.
    """
    form = await request.form()
    try:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            return JSONResponse({"error": "file field is required"}, status_code=400)
        name = form.get("name")
        if not isinstance(name, str):
            name = None
        job = await worker.upload(file, name)
        metrics.inc("sideline_uploads", 1, "Accepted video uploads")
        metrics.inc("sideline_upload_bytes", job.bytes,
                    "Bytes uploaded as videos")
    except _UploadError as exc:
        return JSONResponse({"error": exc.message}, status_code=exc.status)
    finally:
        await form.close()
    return JSONResponse({"job_id": job.id, "status": "queued"}, status_code=202)


def _worker_metrics(worker, metrics):
    """The Prometheus text answer for the worker's /metrics: jobs by
    state, the queue depth, plus the shared counters and gauges."""
    with worker._lock:
        counts = collections.Counter(job.status for job in worker._jobs.values())
        queue_depth = len(worker._queue)
    for state in ("queued", "running", "done", "failed"):
        metrics.set("sideline_jobs", counts.get(state, 0),
                    "Jobs by state", state=state)
    metrics.set("sideline_queue_depth", queue_depth, "Jobs waiting in the queue")
    return Response(content=metrics.render(),
                    media_type="text/plain; version=0.0.4")


def create_worker_app(root, runner=None, max_bytes=_DEFAULT_MAX_BYTES, token=None):
    """Build the FastAPI app for the video worker and start its thread.

    root: folder for the uploaded videos and results (created if missing).
    runner: optional ``runner(video_path, out_path, on_progress)`` that
        processes one video; the default starts ``python -m sideline run``
        in a subprocess. DELETE cancels a running job by setting its cancel
        event: a runner that declares a ``cancel`` parameter is called with
        ``cancel=<threading.Event>`` and must return promptly once it is set
        (the single worker thread would otherwise be blocked and the whole
        FIFO queue would stall behind the deleted job).
    max_bytes: maximum upload size in bytes.
    token: bearer token string; when not None, every route except
        ``GET /health`` needs ``Authorization: Bearer <token>`` (compared
        with hmac.compare_digest; wrong or missing answers 401).

    The app also installs, around every route: the JSON request log
    (stdout) with the X-Request-Id echo, the per-client rate limit for
    non-GET requests (60 per minute, 429 with Retry-After), and the
    bearer authentication when a token is given. ``GET /metrics`` serves
    the Prometheus text registry: requests by status class, uploads and
    uploaded bytes, jobs by state, queue depth and processing seconds.
    """
    worker = Worker(root, runner=runner, max_bytes=max_bytes)
    worker.start()
    app = FastAPI(title="sideline video worker")
    app.state.worker = worker
    metrics = Metrics()
    worker.metrics = metrics

    def not_found():
        return JSONResponse({"error": "job not found"}, status_code=404)

    @app.get("/health")
    def health():
        with worker._lock:
            return {"status": "ok",
                    "busy": worker._running_id is not None,
                    "queued": len(worker._queue)}

    @app.get("/metrics", include_in_schema=False)
    def metrics_route():
        return _worker_metrics(worker, metrics)

    @app.post("/jobs")
    async def create_job(request: Request):
        return await _create_job(request, worker, metrics)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = worker.get(job_id)
        if job is None:
            return not_found()
        return job.to_dict()

    @app.get("/jobs/{job_id}/result")
    def get_result(job_id: str):
        job = worker.get(job_id)
        if job is None:
            return not_found()
        if job.status != "done":
            return JSONResponse(
                {"error": "job is %s; the result is available once the job is done"
                 % job.status}, status_code=409)
        path = worker._result_path(job)
        if not path.is_file():
            return JSONResponse({"error": "result file is missing"}, status_code=409)
        return FileResponse(path, media_type="application/x-ndjson")

    @app.delete("/jobs/{job_id}")
    def delete_job(job_id: str):
        if not worker.delete(job_id):
            return not_found()
        return Response(status_code=204)

    # Middleware order: added last, runs first (outermost). The request
    # log sees every answer (including 401/429) and echoes the request id
    # on all of them; the rate limit counts non-GET requests; the token
    # gate keeps every route but GET /health behind the bearer token.
    if token is not None:
        install_auth(app, token,
                     exempt=lambda request: request.method == "GET"
                     and request.url.path == "/health")
    install_rate_limit(app)
    install_request_log(app, metrics)
    return app
