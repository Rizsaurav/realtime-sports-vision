"""HTTP client for the video worker service (``src/sideline/worker.py``).

The library app forwards a browser video upload to the worker, polls the
job status and downloads the finished result. This module implements that
client with the standard library only (``http.client``): the upload is
streamed to the worker in chunks with chunked transfer encoding, so a
whole video is never read into memory.

Settings (all on ``VideoWorkerClient``):

* ``timeout``: socket timeout for establishing the connection (30 s by
default); a worker that does not accept the connection fails fast instead
of stalling the upload.
* ``idle_timeout``: socket timeout for an established connection (120 s by
default); a transfer that stops making progress is aborted instead of
hanging, on the upload and on every other request.
* ``token``: when set (the library passes it from ``--worker-token-file``),
every request carries ``Authorization: Bearer <token>``; the worker
answers 401 without it when it was started with its own ``--token-file``.
* ``max_bytes``: the worker's upload limit in bytes; informational only,
reported by the library's /api/limits (the library also enforces it on the
first hop, before it stages or forwards anything, and the worker enforces
its own copy and answers 413/507 for oversized uploads).

The blocking methods of ``VideoWorkerClient`` are meant to be called from
a worker thread (``asyncio.to_thread``); the library app never performs
this I/O on the event loop. ``fetch_result`` hands out a stream with
documented close semantics: ``close()`` releases the connection and is
safe to call both after the stream is exhausted and after an early break.
"""

import http.client
import json
import logging
import secrets
import ssl

__all__ = ["VideoWorkerClient", "WorkerClientError", "WORKER_EXTENSIONS"]

logger = logging.getLogger(__name__)

# The video extensions the worker accepts (kept in one place with the
# worker; imported so the library's 400 message can never drift from it).
from .worker import _ALLOWED_EXTS as WORKER_EXTENSIONS

_CHUNK = 1024 * 1024
_CONNECT_TIMEOUT = 30.0
_IDLE_TIMEOUT = 120.0
# The worker's own default upload limit (worker.py); used when the client
# is constructed without an explicit limit for /api/limits reporting.
_DEFAULT_MAX_BYTES = 2 * 1024 ** 3
# A reachable worker that answered with a body that is not the promised
# JSON: not a reachability failure, so it is reported with the 502 the
# library uses for bad worker answers and a message that says what is
# wrong and what to do.
_MALFORMED_ANSWER = ("the video processing machine sent a malformed answer; "
                     "check the worker's logs and version, then try again")


class WorkerClientError(Exception):
    """A failed worker request, with the worker's own message when it answered.

    status: the worker's HTTP status when the worker answered, or None when
    the worker could not be reached at all (connection refused, DNS, a
    timeout). A reachable worker that sends a malformed body is reported
    with status 502 and ``_MALFORMED_ANSWER``. The library app maps None
    to a 502 with the reachability message and other statuses to the
    worker's status with its message.
    """

    def __init__(self, message, status=None):
        super().__init__(message)
        self.message = message
        self.status = status


def _error_message(body, status):
    """The worker's own error text from a JSON answer, else a fallback."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str) and payload["error"]:
        return payload["error"]
    return "the video processing machine answered %d" % status


class _StreamedBody:
    """Iterator over the body of an open ``http.client`` response.

    Close semantics: ``close()`` releases the connection. Call it when the
    stream is exhausted as well as when iteration stops early (a cap, a
    break) - the connection must not be left open for the garbage
    collector to find. Calling it twice, or after exhaustion, is a no-op.
    """

    def __init__(self, response):
        self._response = response

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self._response.read(_CHUNK)
        if not chunk:
            raise StopIteration
        return chunk

    def close(self):
        response, self._response = self._response, None
        if response is None:
            return
        close = getattr(response, "close", None)
        if close is not None:
            try:
                close()
            except OSError as exc:
                logger.debug("the worker connection is already gone; "
                             "nothing left to release: %s", exc)


def _connect(host, port, uses_https, timeout):
    try:
        if uses_https:
            context = ssl.create_default_context()
            return http.client.HTTPSConnection(host, port, timeout=timeout, context=context)
        return http.client.HTTPConnection(host, port, timeout=timeout)
    except (OSError, ssl.SSLError) as exc:
        raise WorkerClientError(
            "could not reach the video processing machine at %s:%s: %s" % (host, port, exc))


def _apply_idle_timeout(conn, idle):
    """Switch an established socket from the connect timeout to the idle
    timeout, so a stalled transfer is aborted after the idle limit. A
    connection that does not expose its socket (a test double) is kept as
    is."""
    sock = getattr(conn, "sock", None)
    if sock is None:
        return
    try:
        sock.settimeout(idle)
    except (OSError, AttributeError) as exc:
        logger.debug("could not switch the connection to the idle timeout "
                     "(%s): %s", idle, exc)


def _read_error(conn, response):
    """Raise WorkerClientError for a non-success worker answer."""
    try:
        body = response.read()
    finally:
        conn.close()
    raise WorkerClientError(_error_message(body, response.status), status=response.status)


class VideoWorkerClient:
    """A minimal blocking client for the worker's HTTP API.

    url: the worker base URL (a scheme is required; a missing one is
    assumed to be http), optionally with a path prefix.
    max_bytes: the worker's upload limit in bytes; informational only,
    reported by the library's /api/limits (the library also enforces the
    limit on the first hop and the worker enforces its own copy, answering
    413/507 for oversized uploads).
    timeout: seconds to wait for the connection to be established (30 s).
    idle_timeout: seconds an established connection may be idle before the
    request is aborted (120 s); it applies to the upload and to every
    other request.
    token: bearer token sent as ``Authorization: Bearer <token>`` on every
    request when the worker requires it (its --token-file).
    """

    def __init__(self, url, max_bytes=_DEFAULT_MAX_BYTES, timeout=_CONNECT_TIMEOUT,
                 idle_timeout=_IDLE_TIMEOUT, token=None):
        url = str(url).strip()
        if "://" not in url:
            url = "http://" + url
        parts = url.split("://", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("worker URL needs a scheme and a host: %r" % url)
        self.uses_https = parts[0] == "https"
        rest = parts[1]
        if rest.startswith("//"):
            rest = rest[2:]
        hostport, _, path = rest.partition("/")
        if ":" in hostport:
            host, _, port = hostport.partition(":")
        else:
            host, port = hostport, None
        if not host:
            raise ValueError("worker URL has no host: %r" % url)
        self.host = host
        try:
            self.port = int(port) if port else (443 if self.uses_https else 80)
        except ValueError:
            raise ValueError("worker URL has a bad port: %r" % url)
        self.base = ("/" + path).rstrip("/") if path else ""
        self.timeout = timeout
        self.idle_timeout = idle_timeout
        self.max_bytes = max_bytes
        self.token = token

    # -- internal ----------------------------------------------------------

    def _open(self, method, path):
        conn = _connect(self.host, self.port, self.uses_https, self.timeout)
        try:
            conn.putrequest(method, self.base + path)
            if self.token is not None:
                conn.putheader("Authorization", "Bearer " + self.token)
            conn.endheaders()
            _apply_idle_timeout(conn, self.idle_timeout)
            return conn, conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            raise WorkerClientError(
                "could not reach the video processing machine at %s:%s: %s"
                % (self.host, self.port, exc))

    # -- the worker API ------------------------------------------------------

    def submit(self, stream, filename, name=None):
        """POST the video to the worker's /jobs; returns the job id.

        stream: a sync file-like object with read(size) -> bytes; it is
        read in chunks and sent with chunked transfer encoding, so the
        video is never held in memory. filename supplies the extension and
        the display name fallback (as on the worker itself); name is the
        optional user-supplied job name.
        """
        boundary = "sideline" + secrets.token_hex(16)
        # the client's file name only ever goes into the multipart header
        # (the worker itself never uses it for a path); keep the header
        # unambiguous by dropping quote and line-break characters
        safe_name = str(filename or "").replace('"', "").replace("\r", "").replace("\n", "")
        head = (
            "--%s\r\n"
            'Content-Disposition: form-data; name="file"; filename="%s"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n" % (boundary, safe_name)
        ).encode("utf-8")
        tail = b""
        if name is not None:
            tail += (
                "\r\n--%s\r\n"
                'Content-Disposition: form-data; name="name"\r\n\r\n'
                "%s\r\n" % (boundary, str(name))
            ).encode("utf-8")
        tail += ("\r\n--%s--\r\n" % boundary).encode("utf-8")

        conn = _connect(self.host, self.port, self.uses_https, self.timeout)
        try:
            conn.putrequest("POST", self.base + "/jobs")
            conn.putheader("Content-Type", "multipart/form-data; boundary=%s" % boundary)
            if self.token is not None:
                conn.putheader("Authorization", "Bearer " + self.token)
            conn.putheader("Transfer-Encoding", "chunked")
            try:
                conn.endheaders(chunked=True)
            except TypeError:
                # The standard library's HTTPConnection.endheaders takes no
                # chunked flag; the explicit Transfer-Encoding header above
                # carries the framing, which the body below applies itself.
                conn.endheaders()
            _apply_idle_timeout(conn, self.idle_timeout)
            # Every part of the body is sent as its own chunk, so the video
            # is never held in memory (chunked transfer encoding).
            def send_chunk(data):
                conn.send(b"%X\r\n" % len(data) + data + b"\r\n")
            send_chunk(head)
            while True:
                chunk = stream.read(_CHUNK)
                if not chunk:
                    break
                send_chunk(chunk)
            send_chunk(tail)
            conn.send(b"0\r\n\r\n")
            response = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            raise WorkerClientError(
                "could not reach the video processing machine at %s:%s: %s"
                % (self.host, self.port, exc))
        if response.status != 202:
            _read_error(conn, response)
        try:
            body = response.read()
        finally:
            conn.close()
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise WorkerClientError(_MALFORMED_ANSWER, status=502) from exc
        job_id = payload.get("job_id") if isinstance(payload, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise WorkerClientError(_MALFORMED_ANSWER, status=502)
        return job_id

    def status(self, job_id):
        """GET /jobs/<id>; the job status dict, or raises WorkerClientError."""
        conn, response = self._open("GET", "/jobs/%s" % job_id)
        if response.status != 200:
            _read_error(conn, response)
        try:
            body = response.read()
        finally:
            conn.close()
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise WorkerClientError(_MALFORMED_ANSWER, status=502) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
            raise WorkerClientError(_MALFORMED_ANSWER, status=502)
        return payload

    def fetch_result(self, job_id):
        """GET /jobs/<id>/result; an iterator of bytes of the result file."""
        conn, response = self._open("GET", "/jobs/%s/result" % job_id)
        if response.status != 200:
            _read_error(conn, response)
        return _StreamedBody(response)

    def delete(self, job_id):
        """DELETE /jobs/<id>; None on success, WorkerClientError otherwise."""
        conn, response = self._open("DELETE", "/jobs/%s" % job_id)
        if response.status == 204:
            conn.close()
            return None
        _read_error(conn, response)
