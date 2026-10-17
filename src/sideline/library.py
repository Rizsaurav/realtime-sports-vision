"""Match library web app: the match API, the two library pages and the video upload.

``create_library_app(root, worker_url=None, worker_client=None,
admin_token=None)`` builds a FastAPI application that serves the
``/api/matches`` endpoints (from ``matches.create_router``) together with
the static files in ``src/sideline/web`` under ``/static``. Only the
standard library, FastAPI/Starlette and the light ``sideline`` modules
are imported, so the app stays importable without the heavy pipeline
dependencies (torch, cv2, numpy).

With a worker configured (``worker_url`` or an injected ``worker_client``),
the app also accepts video uploads: it streams them to the video worker,
tracks the jobs and stores the finished results through the same path as
ordinary match uploads.

The plug-in features of ``sideline.features`` register their routes on
this app (mounted under ``/api/matches/{match_id}``) and add
``GET /api/features``; a broken feature is skipped, never fatal.

Security and observability settings:

* ``admin_token``: optional bearer token. When set, every non-GET
  request (``POST /api/matches``, ``POST /api/videos``, the DELETE routes)
  needs ``Authorization: Bearer <token>`` (compared with
  hmac.compare_digest; wrong or missing answers 401); GETs stay open.
  Without it everything stays open, and the CLI then refuses a ``--host``
  other than 127.0.0.1 unless ``--allow-open`` is given.
* Request size limit, enforced before the body is read: with a
  Content-Length header an oversized request is answered 413 at once -
  nothing is spooled or forwarded. The video route is held to the
  worker's video limit (the ``max_video_bytes`` published by
  ``/api/limits``); every other body is held to the match store limit.
  A chunked video upload without a Content-Length is capped at the same
  limit while it is spooled. The worker's result download is capped the
  same way and its stream is closed on exhaustion and on an early break.
* Structured log: one JSON line per request on stdout with ``ts``,
  ``method``, ``path`` (no query string), ``status``, ``ms``,
  ``bytes_in`` and the request id from (or generated for) the
  ``X-Request-Id`` header, which is echoed in the response. The line
  never carries tokens, uploaded file names or bodies.
* Metrics: ``GET /metrics`` serves Prometheus text format - requests by
  status class, uploads and uploaded bytes, and for the library the
  matches stored and the disk bytes used. Counters only go up within a
  process.
* Rate limit: at most 60 non-GET requests per minute per client, 429
  with a ``Retry-After`` header when exceeded.
* Security headers on the two pages: a Content-Security-Policy that
  allows only same-origin scripts and styles (plus ``img-src 'self'
  data:``) - each page's single inline module script is pinned by its
  sha256 hash, since the pages are served byte-for-byte from the web
  folder - plus ``X-Content-Type-Options: nosniff`` and
  ``Referrer-Policy: no-referrer``.
* Graceful shutdown: uvicorn's default SIGTERM handling finishes the
  in-flight requests before the process exits (no token to hold, nothing
  to kill).
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.datastructures import UploadFile
from starlette.staticfiles import StaticFiles

from .features import FeatureContext, register_all
from .matches import _RequestError
from .videos import VideoWorkerClient, WorkerClientError, WORKER_EXTENSIONS
from .worker import (Metrics, install_auth, install_rate_limit,
                     install_request_log)

_log = logging.getLogger(__name__)

# WEB_DIR: the web asset folder shipped with the package (pages, stylesheets,
# lib/*.js), served both as the two top-level pages and under /static.
WEB_DIR = Path(__file__).resolve().parent / "web"

_JOB_ID_RE = re.compile(r"j_[0-9a-f]{12}\Z")
_MATCH_ID_RE = re.compile(r"m_[0-9a-f]{10}\Z")
_CHUNK = 1024 * 1024
# The worker's default upload limit (worker.py), reported by /api/limits
# when the worker client does not carry its own limit.
_DEFAULT_MAX_VIDEO_BYTES = 2 * 1024 ** 3
# Serializes the done-job resolution (download + store) and the job delete
# across concurrent requests; both run in worker threads, so a threading
# lock never blocks the event loop.
_RESOLVE_LOCK = threading.Lock()

_UNREACHABLE = ("the video processing machine is not reachable; "
                "check that the worker is running and try again")


class _SizeLimitError(Exception):
    """A staged video upload that exceeded the worker's video limit."""


def _store_bytes(root):
    """The total size in bytes of everything under the store root."""
    total = 0
    root = Path(root)
    if not root.is_dir():
        return 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, filename))
            except OSError as exc:
                _log.debug("the store walk skipped %s: %s",
                           os.path.join(dirpath, filename), exc)
    return total


class _StoreStats:
    """The store gauges for GET /metrics.

    The match count is kept in memory and updated on every add and
    delete; the disk bytes are the store walk's result, re-run at most
    once per store change (the cached walk result is kept until an add or
    delete dirties it), so a Prometheus scrape never walks the whole
    store.
    """

    def __init__(self, store):
        self._store = store
        self._lock = threading.Lock()
        self._stored = len(store.list())
        self._bytes = _store_bytes(store.root)
        self._dirty = False

    def note_added(self):
        with self._lock:
            self._stored += 1
            self._dirty = True

    def note_deleted(self):
        with self._lock:
            self._stored -= 1
            self._dirty = True

    def snapshot(self):
        """(matches stored, disk bytes); the walk runs only when dirty."""
        with self._lock:
            if self._dirty:
                self._bytes = _store_bytes(self._store.root)
                self._dirty = False
            return self._stored, self._bytes


def _track_store_stats(store, stats):
    """Route the store's add and delete through the gauge tracker, so the
    /metrics gauges stay current without a disk walk per scrape."""
    add = store.add
    delete = store.delete

    def _add(raw, name, source=None):
        meta, created = add(raw, name, source=source)
        if created:
            stats.note_added()
        return meta, created

    def _delete(match_id):
        removed = delete(match_id)
        if removed:
            stats.note_deleted()
        return removed

    store.add = _add
    store.delete = _delete


def _csp_for_page(html):
    """The Content-Security-Policy for one served page.

    Same-origin scripts and styles only (plus ``img-src 'self' data:``):
    the pages link no remote asset. The page's single inline module
    script - kept inline so the page file stays what it is on disk and
    what the smoke tests run - is pinned by its sha256 hash, so no other
    inline script (unsafe-inline) can run.
    """
    policy = "default-src 'self'; script-src 'self'"
    match = re.search(r'<script type="module">([\s\S]*?)</script>', html)
    if match is not None:
        digest = base64.b64encode(
            hashlib.sha256(match.group(1).encode("utf-8")).digest()).decode("ascii")
        policy += " 'sha256-%s'" % digest
    return policy + "; style-src 'self'; img-src 'self' data:"


def _oversize_check(request, video_limit, match_limit):
    """The 413 JSONResponse for an oversized Content-Length, else None.

    The size is enforced before the body is read (nothing is spooled or
    forwarded). The video route is held to the worker's video limit;
    every other body to the match store limit. A missing or non-numeric
    length gives no early answer (a chunked body is still capped while it
    is read by the route that spools it).
    """
    if request.method not in ("POST", "PUT", "PATCH"):
        return None
    if request.url.path == "/api/videos":
        limit = video_limit
        message = ("file exceeds the maximum size of %d bytes; "
                   "choose a smaller recording")
    else:
        limit = match_limit
        message = "file exceeds the maximum size of %d bytes"
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        length = int(raw)
    except ValueError:
        _log.debug("non-numeric Content-Length %r: the limit is enforced "
                   "while the body is read", raw)
        return None
    if length > limit:
        return JSONResponse({"error": message % limit}, status_code=413)
    return None


def _serve_page(name, page_headers):
    """The route handler for one page: the file served byte-for-byte
    with its security headers (computed from the same file, so the CSP
    hash and the script it pin cannot drift apart)."""
    def page():
        return Response(content=(WEB_DIR / name).read_bytes(),
                        media_type="text/html", headers=page_headers[name])
    page.__name__ = name.replace(".", "_")
    return page


def _page_security_headers(page_path):
    """The security header dict for one page, computed from the file that
    is served (the served bytes are the file bytes, so the CSP hash and
    the script it pins can never drift apart)."""
    try:
        html = page_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    return {"Content-Security-Policy": _csp_for_page(html),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer"}


def _parse_job_mapping(data):
    """The video_jobs.json payload as a job-id to outcome mapping.

    Non-dict payloads give an empty mapping, and one malformed entry
    never hides the others.
    """
    entries = {}
    if not isinstance(data, dict):
        return entries
    for job_id, entry in data.items():
        try:
            if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
                continue
            if not isinstance(entry, dict):
                continue
            match_id = entry.get("match_id")
            if isinstance(match_id, str) and _MATCH_ID_RE.match(match_id):
                entries[job_id] = {"match_id": match_id}
                continue
            error = entry.get("error")
            if isinstance(error, str) and error:
                entries[job_id] = {"error": error}
        except (AttributeError, TypeError):
            continue  # skip this entry, keep the rest
    return entries


class _VideoJobLog:
    """The job-id to outcome mapping, persisted in the store root.

    Every job this app resolved - stored as a match, or rejected because
    its result failed validation - is remembered in ``video_jobs.json`` in
    the store root, so a restart (or the worker's own 24 h purge) cannot
    lose the mapping. A mapping is removed when the match is deleted or
    the job is cancelled.
    """

    def __init__(self, root):
        self.path = Path(root) / "video_jobs.json"
        self._lock = threading.Lock()
        self._entries = self._load()

    def _load(self):
        """The on-disk mapping.

        A missing file is the legitimate empty state (a fresh store): an
        empty mapping. A read or parse failure is not - the file stays on
        disk, so the loss must be visible in the log; the process keeps
        working from an empty mapping for this run.
        """
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            _log.debug("no video job mapping at %s yet: starting empty", self.path)
            return {}
        except (OSError, ValueError) as exc:
            _log.warning("could not read or parse the video job mapping %s: %s; "
                         "continuing with an empty mapping for this run "
                         "(the file stays on disk)", self.path, exc)
            return {}
        return _parse_job_mapping(data)

    def get(self, job_id):
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            return None
        with self._lock:
            return self._entries.get(job_id)

    def set_done(self, job_id, match_id):
        self._put(job_id, {"match_id": match_id})

    def set_failed(self, job_id, error):
        self._put(job_id, {"error": error})

    def remove(self, job_id):
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            return False
        with self._lock:
            existed = self._entries.pop(job_id, None) is not None
            self._write_locked()
            return existed

    def _put(self, job_id, entry):
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            return
        with self._lock:
            self._entries[job_id] = entry
            self._write_locked()

    def _write_locked(self):
        if not self._entries:
            if self.path.exists():
                try:
                    self.path.unlink()
                except OSError as exc:
                    _log.warning("could not remove the emptied video job "
                                 "mapping %s: %s", self.path, exc)
            return
        data = (json.dumps(self._entries, indent=2, sort_keys=True) + "\n").encode("utf-8")
        try:
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".video-jobs-")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, self.path)
            except BaseException:
                if os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError as exc:
                        _log.warning("could not remove the temporary job "
                                     "mapping %s: %s", tmp, exc)
                raise
        except OSError as exc:
            _log.error("could not persist the video job mapping: %s", exc)


def _download_result(worker_client, job_id, cap):
    """The worker's result file as bytes, capped at cap (blocking).

    The cap is store.max_bytes + 1: an oversized result then reaches
    MatchStore.add as exactly one byte over its limit, which answers with
    the same 413 message as an oversized direct upload. The stream is
    closed when it is exhausted and when the cap stops the iteration
    early, so the worker's connection is released either way instead of
    being left to the garbage collector.
    """
    parts = []
    total = 0
    stream = worker_client.fetch_result(job_id)
    try:
        for chunk in stream:
            remaining = cap - total
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
            total += len(chunk)
            parts.append(bytes(chunk))
            if total == cap:
                break
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    return b"".join(parts)


async def _forget_worker_copy(worker_client, job_id):
    """Delete the worker's copy of a resolved job; best effort, because the
    local mapping already records the outcome (a leftover is purged by the
    worker's 24 h rule)."""
    try:
        await asyncio.to_thread(worker_client.delete, job_id)
    except WorkerClientError as exc:
        _log.warning("could not delete job %s on the worker: %s", job_id, exc.message)


async def _spool_upload(file, ext, first, limit):
    """Write a browser upload to a temporary file in chunks (async).

    first: the first byte already read (the empty-file check). limit: the
    maximum number of bytes to stage (the worker's video limit, enforced
    on the first hop before anything is forwarded to the worker). Returns
    the spool file's path and the number of bytes written; when the
    upload exceeds the limit the spool is removed and _SizeLimitError is
    raised.
    """
    fd, tmp = tempfile.mkstemp(prefix=".video-upload-", suffix=ext)
    try:
        dst = os.fdopen(fd, "wb")
    except Exception:
        os.close(fd)
        _remove_spool(tmp)
        raise
    size = len(first)
    try:
        with dst:
            dst.write(first)
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise _SizeLimitError()
                dst.write(chunk)
        return tmp, size
    except BaseException:
        _remove_spool(tmp)
        raise


def _submit_to_worker(worker_client, tmp, filename, name):
    """Stream the spooled upload to the worker (blocking, run threaded)."""
    with open(tmp, "rb") as handle:
        return worker_client.submit(handle, filename, name)


def _remove_spool(tmp):
    """Remove the staged upload file if it still exists (best effort)."""
    if tmp is None or not os.path.exists(tmp):
        return
    try:
        os.unlink(tmp)
    except OSError as exc:
        _log.warning("could not remove the staged upload file %s: %s", tmp, exc)


def _worker_error_response(exc):
    """The JSONResponse for a WorkerClientError.

    status None means the worker could not be reached at all: a 502 with
    the fixed reachability message (and how to fix it). Any other status
    - including the 502 the client reports for a reachable worker that
    sent a malformed body - is passed through with the worker's message.
    """
    if exc.status is None:
        return JSONResponse({"error": _UNREACHABLE}, status_code=502)
    return JSONResponse({"error": exc.message}, status_code=exc.status)


def _resolve_done_job(worker_client, store, joblog, job_id, name, cap):
    """Download and store a done job's result, exactly once (blocking).

    Runs in a worker thread. The resolve lock serializes racing calls,
    which re-check the log before doing any work. Returns {"match_id":
    ...} or {"error": ...} (a result that failed validation); raises
    WorkerClientError/OSError when the download or the storage I/O fails
    (the job stays resolvable, the next poll retries).
    """
    with _RESOLVE_LOCK:
        outcome = joblog.get(job_id)
        if outcome is not None:
            return outcome
        raw = _download_result(worker_client, job_id, cap)
        try:
            meta, _created = store.add(raw, name, source=job_id)
        except _RequestError as exc:
            joblog.set_failed(job_id, exc.message)
            return {"error": exc.message}
        joblog.set_done(job_id, meta["id"])
        return {"match_id": meta["id"]}


def _delete_job_from_worker(worker_client, joblog, job_id):
    """Forget the job locally and delete the worker's copy (blocking).

    Returns (had_local, error) where error is None or the WorkerClientError
    the worker's delete raised (the local forget already happened).
    """
    with _RESOLVE_LOCK:
        had_local = joblog.remove(job_id)
        try:
            worker_client.delete(job_id)
        except WorkerClientError as exc:
            return had_local, exc
        return had_local, None


async def _video_upload(request, worker_client, video_max_bytes, metrics):
    """The POST /api/videos logic; the route wires in the worker.

    Answers 503 without a worker, 400 for a wrong extension or an empty
    file, 413 for a video over the worker's limit (before it is spooled
    when a Content-Length says so, and while spooling when it does not),
    202 ``{"job_id": ..., "status": "queued"}`` when the video reached the
    worker, otherwise the worker's own error (via
    ``_worker_error_response``) or a 500 with guidance when the upload
    could not be staged on this server.
    """
    if worker_client is None:
        return JSONResponse({"error": "video processing is not configured"},
                             status_code=503)
    form = await request.form()
    tmp = None
    try:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            return JSONResponse({"error": "file field is required"}, status_code=400)
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()
        if ext not in WORKER_EXTENSIONS:
            return JSONResponse(
                {"error": "file extension must be one of: %s; choose a video file"
                 % " ".join(WORKER_EXTENSIONS)}, status_code=400)
        first = await file.read(1)
        if not first:
            return JSONResponse(
                {"error": "file is empty; choose a video file with content"},
                status_code=400)
        name = form.get("name")
        if not isinstance(name, str):
            name = None
        # Spool the browser upload to a temporary file, then stream that
        # file to the worker from a worker thread (the client does
        # blocking socket I/O and must not run on the event loop). The
        # worker's limit is enforced while spooling, so an oversized
        # upload is not forwarded and not left on disk.
        tmp, size = await _spool_upload(file, ext, first, video_max_bytes)
        job_id = await asyncio.to_thread(_submit_to_worker, worker_client, tmp,
                                         filename, name)
        metrics.inc("sideline_uploads", 1, "Accepted video uploads")
        metrics.inc("sideline_upload_bytes", size, "Bytes uploaded as videos")
        return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)
    except _SizeLimitError:
        return JSONResponse(
            {"error": "file exceeds the maximum size of %d bytes; choose a "
                      "smaller recording" % video_max_bytes},
            status_code=413)
    except WorkerClientError as exc:
        return _worker_error_response(exc)
    except OSError as exc:
        return JSONResponse(
            {"error": "the video could not be staged for the worker: %s; "
                      "check that the server has enough free disk space, "
                      "then try the upload again" % exc},
            status_code=500)
    finally:
        _remove_spool(tmp)
        await form.close()


def _stored_outcome(store, joblog, job_id):
    """The answer for a remembered job, or None when there is none.

    A remembered match is served without touching the worker; a stale
    mapping (the match was deleted) is dropped, so the job is unknown
    again; a remembered failed validation is served as failed.
    """
    outcome = joblog.get(job_id)
    if outcome is None:
        return None
    if "match_id" not in outcome:
        return {"job_id": job_id, "status": "failed", "error": outcome["error"]}
    if store.get(outcome["match_id"]) is None:
        joblog.remove(job_id)  # the match was deleted: stale mapping
        return None
    return {"job_id": job_id, "status": "done", "progress": 1.0,
            "match_id": outcome["match_id"]}


async def _video_job_status(store, worker_client, joblog, job_id):
    """The GET /api/videos/{job_id} logic; the route wires in the store,
    the worker and the job log.

    Serves the remembered outcome first (a stored match, or a failed
    validation) without touching the worker; otherwise asks the worker
    and passes its status through. When the status is done for the first
    time the result is downloaded and stored as a normal match (exactly
    once even if several polls race), the mapping is persisted in the
    store root, the worker's copy is deleted and ``match_id`` is added.
    """
    if not _JOB_ID_RE.match(job_id):
        return JSONResponse({"error": "job not found"}, status_code=404)
    stored = _stored_outcome(store, joblog, job_id)
    if stored is not None:
        return stored
    if worker_client is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    try:
        status = await asyncio.to_thread(worker_client.status, job_id)
    except WorkerClientError as exc:
        return _worker_error_response(exc)
    if status.get("status") != "done":
        return status
    # The status is done for the first time: download the result and
    # store it through the same path as POST /api/matches (validation
    # included), exactly once even if several polls race.
    name = str(status.get("name") or "video")
    try:
        outcome = await asyncio.to_thread(
            _resolve_done_job, worker_client, store, joblog, job_id, name,
            store.max_bytes + 1)
    except WorkerClientError as exc:
        return _worker_error_response(exc)
    except OSError as exc:
        return JSONResponse(
            {"error": "could not store the processed match: %s; check that "
                      "the server has enough free disk space, then poll the "
                      "job again" % exc},
            status_code=500)
    if "match_id" not in outcome:
        # A result that failed validation: the job is failed with the
        # validation message, and the worker's copy is deleted.
        await _forget_worker_copy(worker_client, job_id)
        return {"job_id": job_id, "status": "failed", "error": outcome["error"]}
    await _forget_worker_copy(worker_client, job_id)
    response = dict(status)
    response["match_id"] = outcome["match_id"]
    return response


async def _video_job_delete(worker_client, joblog, job_id):
    """The DELETE /api/videos/{job_id} logic; the route wires in the
    worker and the job log.

    Cancels the job on the worker and forgets it locally. The local
    forget happens even when the worker's copy survives (the worker
    purges it after 24 h).
    """
    if not _JOB_ID_RE.match(job_id):
        return JSONResponse({"error": "job not found"}, status_code=404)
    if worker_client is None:
        if joblog.remove(job_id):
            return Response(status_code=204)
        return JSONResponse({"error": "job not found"}, status_code=404)
    had_local, exc = await asyncio.to_thread(
        _delete_job_from_worker, worker_client, joblog, job_id)
    if exc is None:
        return Response(status_code=204)
    if exc.status == 404:
        if had_local:
            return Response(status_code=204)
        return JSONResponse({"error": "job not found"}, status_code=404)
    return _worker_error_response(exc)


def _build_video_stack(root, worker_url, worker_client):
    """The video-upload stack for create_library_app.

    Returns (worker_client, joblog, video_max_bytes): the worker client
    (built from worker_url when no client is injected; it stays None,
    i.e. video processing off, without a URL), the worker's video limit
    (the worker's default when the client carries none) and the
    persisted job log in the store root.
    """
    if worker_client is None and worker_url is not None and str(worker_url).strip():
        worker_client = VideoWorkerClient(str(worker_url).strip())
    video_max_bytes = getattr(worker_client, "max_bytes", None)
    if not isinstance(video_max_bytes, int) or video_max_bytes <= 0:
        video_max_bytes = _DEFAULT_MAX_VIDEO_BYTES
    return worker_client, _VideoJobLog(root), video_max_bytes


def _install_middlewares(app, store, metrics, video_max_bytes, admin_token):
    """The four middlewares, added in this order: the size limit (413
    before the body is read, from the Content-Length header), the write
    protection when an admin token is set, the per-client rate limit for
    non-GET requests (60 per minute, 429 with Retry-After) and the JSON
    request log on stdout (it sees every answer, 401/413/429 included,
    and echoes the X-Request-Id).
    """

    @app.middleware("http")
    async def _reject_oversized_bodies(request, call_next):
        rejected = _oversize_check(request, video_max_bytes, store.max_bytes)
        if rejected is not None:
            return rejected
        return await call_next(request)

    if admin_token is not None:
        install_auth(app, admin_token,
                     exempt=lambda request: request.method in ("GET", "HEAD"))
    install_rate_limit(app)
    install_request_log(app, metrics)


def _video_routes(app, store, worker_client, joblog, video_max_bytes, metrics):
    """The three video endpoints; the logic lives in the module-level
    ``_video_upload``, ``_video_job_status`` and ``_video_job_delete``,
    which take the store, the worker client and the job log explicitly."""

    @app.post("/api/videos", include_in_schema=False)
    async def upload_video(request: Request):
        return await _video_upload(request, worker_client, video_max_bytes, metrics)

    @app.get("/api/videos/{job_id}", include_in_schema=False)
    async def get_video_job(job_id: str):
        return await _video_job_status(store, worker_client, joblog, job_id)

    @app.delete("/api/videos/{job_id}", include_in_schema=False)
    async def delete_video_job(job_id: str):
        return await _video_job_delete(worker_client, joblog, job_id)


def create_library_app(root, worker_url=None, worker_client=None, admin_token=None):
    """Create the FastAPI application for the match library.

    root: str or pathlib.Path folder for the match store; created
    (including parents) if missing. worker_url: URL of the video worker
    service (the app then streams video uploads to it); worker_client:
    an injected client object with submit(stream, filename, name),
    status(job_id), fetch_result(job_id) and delete(job_id) - used instead
    of worker_url (in tests it stands in for the real
    ``videos.VideoWorkerClient``). admin_token: bearer token string; when
    set, every non-GET request needs ``Authorization: Bearer <token>``
    (wrong or missing answers 401); GETs stay open.

    Returns: the FastAPI app: the ``/api/matches`` router over a
    ``MatchStore(root, smooth=True)``, the plug-in feature routes plus
    ``GET /api/features`` (``sideline.features``), ``GET /api/limits``,
    the three video endpoints, ``GET /metrics`` (requests by status
    class, uploads and bytes, matches stored and store disk bytes),
    ``GET /`` (a 307 to /library.html), ``GET /healthz``, the two pages
    served byte-for-byte with their security headers (CSP, nosniff,
    no-referrer) and the ``/static`` mount. Around every route: the size
    limit, the write protection, the rate limit and the request log
    (``_install_middlewares``).
    """
    from .matches import MatchStore, create_router

    app = FastAPI(title="Sideline match library")
    # smooth=True: the analytics summary uses the tracker's jitter
    # smoothing, so the speed and sprint figures stay realistic.
    store = MatchStore(root, smooth=True)
    app.include_router(create_router(store))
    # The plug-in features (their routes mount under /api/matches/{match_id},
    # plus GET /api/features); a broken feature is skipped, never fatal.
    register_all(app, FeatureContext(store))
    worker_client, joblog, video_max_bytes = _build_video_stack(root, worker_url, worker_client)
    metrics = Metrics()
    stats = _StoreStats(store)
    _track_store_stats(store, stats)
    page_headers = {name: _page_security_headers(WEB_DIR / name)
                    for name in ("library.html", "match.html")}

    @app.get("/", include_in_schema=False)
    def root_redirect():
        return RedirectResponse(url="/library.html", status_code=307)

    @app.get("/api/limits", include_in_schema=False)
    def limits():
        return {"max_upload_bytes": store.max_bytes,
                "video_enabled": worker_client is not None,
                "max_video_bytes": video_max_bytes}

    @app.get("/metrics", include_in_schema=False)
    def metrics_route():
        stored, bytes_used = stats.snapshot()
        metrics.set("sideline_matches_stored", stored, "Matches in the store")
        metrics.set("sideline_store_bytes", bytes_used,
                    "Bytes used by the match store on disk")
        return Response(content=metrics.render(),
                        media_type="text/plain; version=0.0.4")

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok"}

    for name in ("library.html", "match.html"):
        app.add_api_route("/" + name, _serve_page(name, page_headers),
                          methods=["GET"], include_in_schema=False)

    _video_routes(app, store, worker_client, joblog, video_max_bytes, metrics)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    _install_middlewares(app, store, metrics, video_max_bytes, admin_token)
    return app
