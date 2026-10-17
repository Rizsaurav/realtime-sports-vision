"""Hardening tests: auth, write protection, logs, metrics, limits,
timeouts, graceful shutdown, security headers and the doctor command.

The app-level checks use FastAPI's TestClient (no sockets), the CLI
checks run ``sideline.__main__`` in-process with uvicorn stubbed where a
server would actually start, and the process-level checks (process group
kill, signal wiring) use fakes, because the test sandbox may refuse to
send signals.
"""
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import types
import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sideline import worker as W
from sideline import library as L
from sideline import videos as V
import sideline.__main__ as M


TOKEN = "hardening-" + "a" * 44
ADMIN_TOKEN = "admin-" + "b" * 44

WORKER_ROUTES = ("/jobs/j_1234567890ab", "/jobs/j_1234567890ab/result")


def _token_file(tmp_path, token=TOKEN, mode=0o600, name="token"):
    path = tmp_path / name
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token + "\n")
    os.chmod(path, mode)
    return path


def _worker_app(tmp_path, runner=None, max_bytes=None, token=None):
    if max_bytes is None:
        return W.create_worker_app(str(tmp_path), runner=runner, token=token)
    return W.create_worker_app(str(tmp_path), runner=runner,
                               max_bytes=max_bytes, token=token)


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _log_lines(capsys):
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


class _AsyncUpload:
    """A minimal UploadFile double for Worker.upload outside a request."""

    def __init__(self, data, filename):
        self.filename = filename
        self._data = data

    async def read(self, n=-1):
        if not self._data:
            return b""
        data, self._data = self._data[:n], self._data[n:]
        return data


@pytest.fixture
def argv(monkeypatch):
    def _argv(*parts):
        monkeypatch.setattr(sys, "argv", ["sideline"] + list(parts))
    return _argv


@pytest.fixture
def make_uvicorn(monkeypatch):
    def _make():
        uvicorn = mock.MagicMock(name="uvicorn")
        monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
        return uvicorn
    return _make


def _fake_disk_usage(free):
    return mock.patch.object(shutil, "disk_usage",
                             return_value=types.SimpleNamespace(
                                 total=2 * free, used=free, free=free))


# ---------------------------------------------------------------------------
# 1. worker authentication
# ---------------------------------------------------------------------------

def test_worker_token_protects_all_routes_except_health(tmp_path):
    app = _worker_app(tmp_path, token=TOKEN)
    with TestClient(app) as client:
        # open: health
        assert client.get("/health").status_code == 200
        assert client.get("/health",
                          headers={"Authorization": "Bearer wrong"}).status_code == 200
        # protected: everything else, missing or wrong token
        for path in WORKER_ROUTES + ("/metrics",):
            assert client.get(path).status_code == 401
            assert client.get(path,
                              headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.post("/jobs").status_code == 401
        assert client.delete("/jobs/j_1234567890ab").status_code == 401
        # good token
        headers = {"Authorization": "Bearer " + TOKEN}
        assert client.get("/jobs/j_1234567890ab",
                          headers=headers).status_code == 404
        assert client.get("/metrics", headers=headers).status_code == 200


def test_worker_401_body_is_json_error(tmp_path):
    app = _worker_app(tmp_path, token=TOKEN)
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 401
    body = response.json()
    assert set(body) == {"error"}
    assert body["error"]


def test_worker_token_comparison_is_constant_time(tmp_path):
    import inspect
    assert "compare_digest" in inspect.getsource(W.bearer_matches)
    app = _worker_app(tmp_path, token=TOKEN)
    with TestClient(app) as client:
        # a token of a different length is a clean 401, not an error
        assert client.get("/metrics",
                          headers={"Authorization": "Bearer short"}).status_code == 401
        # without the Bearer prefix it is not the token
        assert client.get("/metrics",
                          headers={"Authorization": TOKEN}).status_code == 401


def test_token_subcommand_creates_a_mode_600_file(tmp_path, argv):
    target = tmp_path / "new" / "token"
    argv("token", str(target))
    M.main()
    assert target.is_file()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    token = target.read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"[0-9a-f]{64}", token)
    # a second run writes a fresh token
    argv("token", str(target))
    M.main()
    assert target.read_text(encoding="utf-8").strip() != token


def test_worker_token_file_checks(tmp_path):
    root = tmp_path / "w"
    # missing file
    with pytest.raises(SystemExit) as exc:
        M._cmd_worker(M.build_parser().parse_args(
            ["worker", "--root", str(root), "--token-file", str(tmp_path / "nope")]))
    assert "does not exist" in str(exc.value)
    # group-readable
    bad = _token_file(tmp_path, mode=0o644, name="group-readable")
    with pytest.raises(SystemExit) as exc:
        M._cmd_worker(M.build_parser().parse_args(
            ["worker", "--root", str(root), "--token-file", str(bad)]))
    assert "readable" in str(exc.value) and "600" in str(exc.value)
    # too short
    short = _token_file(tmp_path, token="x" * 10, name="short")
    with pytest.raises(SystemExit) as exc:
        M._cmd_worker(M.build_parser().parse_args(
            ["worker", "--root", str(root), "--token-file", str(short)]))
    assert "32" in str(exc.value)


def test_worker_refuses_non_loopback_host_without_token(tmp_path):
    pytest.importorskip("uvicorn")  # the refusal is a real-bind guard
    with pytest.raises(SystemExit) as exc:
        M._cmd_worker(M.build_parser().parse_args(
            ["worker", "--root", str(tmp_path), "--host", "0.0.0.0"]))
    message = str(exc.value)
    assert "127.0.0.1" in message and "token" in message


def test_worker_token_file_allows_other_hosts(tmp_path, monkeypatch, make_uvicorn):
    uvicorn = make_uvicorn()
    token_file = _token_file(tmp_path)
    created = {}

    def factory(root, runner=None, max_bytes=None, token=None):
        created["token"] = token
        created["max_bytes"] = max_bytes
        return mock.sentinel.app

    with mock.patch.object(W, "create_worker_app", factory):
        M._cmd_worker(M.build_parser().parse_args(
            ["worker", "--root", str(tmp_path), "--host", "0.0.0.0",
             "--port", "9111", "--token-file", str(token_file)]))
    assert created["token"] == TOKEN
    assert created["max_bytes"] == int(2.0 * 1024 ** 3)
    assert uvicorn.run.call_args.args[0] is mock.sentinel.app
    assert uvicorn.run.call_args.kwargs["host"] == "0.0.0.0"
    assert uvicorn.run.call_args.kwargs["timeout_graceful_shutdown"] == 5


# ---------------------------------------------------------------------------
# 2. library write protection
# ---------------------------------------------------------------------------

def test_library_writes_require_admin_token(tmp_path):
    app = L.create_library_app(str(tmp_path), admin_token=ADMIN_TOKEN)
    match = ("{\"frame\": 0, \"people\": []}\n"
             "{\"frame\": 1, \"people\": []}\n")
    with TestClient(app) as client:
        # GET stays open
        assert client.get("/api/matches").status_code == 200
        assert client.get("/healthz").status_code == 200
        # writes are closed without the token
        assert client.post("/api/matches",
                           files={"file": ("a.jsonl", match.encode())}).status_code == 401
        # ...and open with it
        headers = {"Authorization": "Bearer " + ADMIN_TOKEN}
        created = client.post("/api/matches",
                              files={"file": ("a.jsonl", match.encode())},
                              headers=headers)
        assert created.status_code == 201
        match_id = created.json()["id"]
        assert client.delete("/api/matches/" + match_id).status_code == 401
        assert client.delete("/api/matches/" + match_id,
                             headers=headers).status_code == 204


def test_library_refuses_non_loopback_without_admin_token(tmp_path):
    pytest.importorskip("uvicorn")  # the refusal is a real-bind guard
    with pytest.raises(SystemExit) as exc:
        M._cmd_library(M.build_parser().parse_args(
            ["library", "--root", str(tmp_path), "--host", "0.0.0.0"]))
    assert "127.0.0.1" in str(exc.value)


def test_library_allow_open_permits_other_hosts(tmp_path, monkeypatch, make_uvicorn):
    uvicorn = make_uvicorn()
    with mock.patch.object(L, "create_library_app", return_value=mock.sentinel.app):
        M._cmd_library(M.build_parser().parse_args(
            ["library", "--root", str(tmp_path), "--host", "0.0.0.0",
             "--port", "9123", "--allow-open"]))
    assert uvicorn.run.call_args.args[0] is mock.sentinel.app
    assert uvicorn.run.call_args.kwargs == {
        "host": "0.0.0.0", "port": 9123, "log_level": "warning"}


def test_library_admin_token_file_checks(tmp_path):
    bad = _token_file(tmp_path, token=ADMIN_TOKEN, mode=0o640, name="admin")
    with pytest.raises(SystemExit) as exc:
        M._cmd_library(M.build_parser().parse_args(
            ["library", "--root", str(tmp_path), "--admin-token-file", str(bad)]))
    assert "readable" in str(exc.value)


def test_library_worker_token_file_forwards_the_token(tmp_path, monkeypatch,
                                                      make_uvicorn):
    uvicorn = make_uvicorn()
    token_file = _token_file(tmp_path, token=TOKEN, name="worker-token")
    admin_file = _token_file(tmp_path, token=ADMIN_TOKEN, name="admin")
    seen = {}

    def factory(root, worker_client=None, admin_token=None):
        seen["worker_client"] = worker_client
        seen["admin_token"] = admin_token
        return mock.sentinel.app

    with mock.patch.object(L, "create_library_app", factory):
        M._cmd_library(M.build_parser().parse_args(
            ["library", "--root", str(tmp_path), "--host", "127.0.0.1",
             "--worker", "http://127.0.0.1:9111",
             "--worker-token-file", str(token_file),
             "--admin-token-file", str(admin_file)]))
    client = seen["worker_client"]
    assert isinstance(client, V.VideoWorkerClient)
    assert client.token == TOKEN
    assert seen["admin_token"] == ADMIN_TOKEN


# ---------------------------------------------------------------------------
# 3. structured request logs
# ---------------------------------------------------------------------------

def test_request_log_is_one_json_line_per_request(tmp_path, capsys):
    app = L.create_library_app(str(tmp_path))
    with TestClient(app) as client:
        client.get("/healthz")
    records = _log_lines(capsys)
    assert len(records) == 1
    record = records[0]
    assert set(record) == {"ts", "method", "path", "status", "ms",
                           "bytes_in", "request_id"}
    assert record["method"] == "GET"
    assert record["path"] == "/healthz"
    assert record["status"] == 200
    assert isinstance(record["ms"], int) and record["ms"] >= 0
    assert record["bytes_in"] == 0
    assert re.fullmatch(r"[0-9a-f]{32}", record["request_id"])
    assert record["ts"].endswith("Z")


def test_request_log_path_drops_the_query(tmp_path, capsys):
    app = L.create_library_app(str(tmp_path))
    with TestClient(app) as client:
        client.get("/api/matches?from=2024-01-01&secret=leaked")
    record = _log_lines(capsys)[0]
    assert record["path"] == "/api/matches"
    assert "from=2024-01-01" not in json.dumps(record)


def test_request_id_is_echoed_and_generated(tmp_path, capsys):
    app = L.create_library_app(str(tmp_path))
    with TestClient(app) as client:
        response = client.get("/healthz", headers={"X-Request-Id": "req-42"})
    assert response.headers["x-request-id"] == "req-42"
    record = _log_lines(capsys)[0]
    assert record["request_id"] == "req-42"
    capsys.readouterr()
    with TestClient(app) as client:
        response = client.get("/healthz",
                              headers={"X-Request-Id": "bad id\nwith junk"})
    generated = response.headers["x-request-id"]
    assert re.fullmatch(r"[0-9a-f]{32}", generated)
    record = _log_lines(capsys)[-1]
    assert record["request_id"] == generated


def test_request_log_bytes_in_counts_the_body(tmp_path, capsys):
    app = L.create_library_app(str(tmp_path))
    body = b"x" * 1234
    with TestClient(app) as client:
        client.post("/healthz", content=body,
                    headers={"content-type": "application/octet-stream"})
    record = _log_lines(capsys)[0]
    assert record["method"] == "POST"
    assert record["bytes_in"] == len(body)


def test_request_log_never_contains_tokens_or_filenames(tmp_path, capsys):
    secret_token = "s3cret-" + "z" * 44
    app = W.create_worker_app(str(tmp_path), token=secret_token)
    with TestClient(app) as client:
        client.post("/jobs", data={"file": "secret-video.mp4"},
                    headers={"Authorization": "Bearer " + secret_token})
    line = capsys.readouterr().out
    assert secret_token not in line
    assert "secret-video.mp4" not in line
    assert "Bearer" not in line


# ---------------------------------------------------------------------------
# 4. metrics
# ---------------------------------------------------------------------------

def test_worker_metrics_endpoint_format(tmp_path):
    app = _worker_app(tmp_path)
    with TestClient(app) as client:
        client.get("/health")
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    text = response.text
    assert "# TYPE sideline_requests counter" in text
    assert '# TYPE sideline_jobs gauge' in text
    assert '# TYPE sideline_queue_depth gauge' in text
    assert 'sideline_requests{class="2xx"}' in text
    for state in ("queued", "running", "done", "failed"):
        assert 'sideline_jobs{state="%s"}' % state in text
    assert "sideline_queue_depth" in text


def _count_metric(text, name, label):
    for line in text.splitlines():
        if line.startswith(name + "{" + label + "}"):
            return int(line.rsplit(" ", 1)[1])
        if line.startswith(name + " ") and not label:
            return int(line.rsplit(" ", 1)[1])
    raise AssertionError("%s{%s} not in:\n%s" % (name, label, text))


def test_worker_metrics_count_requests_monotonically(tmp_path):
    app = _worker_app(tmp_path)
    with TestClient(app) as client:
        client.get("/health")  # warm up: the registry starts empty
        first = client.get("/metrics").text
        client.get("/health")
        second = client.get("/metrics").text
    a = _count_metric(first, "sideline_requests", 'class="2xx"')
    b = _count_metric(second, "sideline_requests", 'class="2xx"')
    assert b > a >= 0


def test_worker_metrics_requires_token_when_configured(tmp_path):
    app = _worker_app(tmp_path, token=TOKEN)
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics",
                          headers={"Authorization": "Bearer " + TOKEN}).status_code == 200


def test_worker_metrics_gauges_track_jobs(tmp_path):
    release = threading.Event()

    def runner(video, out, on_progress, cancel=None):
        release.wait(timeout=10)
        out.write_text("{}\n")

    app = _worker_app(tmp_path, runner=runner)
    with TestClient(app) as client:
        jobs = [client.post("/jobs", files={"file": ("v%d.mp4" % i, b"123")}).json()
                for i in range(2)]
        worker = app.state.worker

        def both_busy():
            states = {worker.get(j["job_id"]).status for j in jobs}
            return states == {"running", "queued"}
        assert _wait_for(both_busy)
        text = client.get("/metrics").text
        assert 'sideline_jobs{state="running"} 1' in text
        assert 'sideline_jobs{state="queued"} 1' in text
        assert "sideline_queue_depth 1" in text
        release.set()

        def both_done():
            states = {worker.get(j["job_id"]).status for j in jobs}
            return states == {"done"}
        assert _wait_for(both_done)
        text = client.get("/metrics").text
        assert 'sideline_jobs{state="done"} 2' in text
        assert "sideline_queue_depth 0" in text
        # the accepted uploads and their bytes are counted
        assert "sideline_uploads 2" in text
        assert "sideline_upload_bytes 6" in text
        # the processing time is counted (the jobs each took a moment)
        assert "sideline_processing_seconds" in text


def test_library_metrics_endpoint(tmp_path):
    app = L.create_library_app(str(tmp_path))
    match = ("{\"frame\": 0, \"people\": []}\n"
             "{\"frame\": 1, \"people\": []}\n")
    with TestClient(app) as client:
        text = client.get("/metrics").text
        assert "sideline_matches_stored 0" in text
        assert "sideline_store_bytes" in text
        assert client.post("/api/matches",
                           files={"file": ("a.jsonl", match.encode())}).status_code == 201
        text = client.get("/metrics").text
        assert "sideline_matches_stored 1" in text
        assert _count_metric(text, "sideline_store_bytes", "") > 0
        assert 'sideline_requests{class="2xx"}' in text
        # the store contents (file names) never leak into the registry
        assert "a.jsonl" not in text


# ---------------------------------------------------------------------------
# 5. limits and timeouts
# ---------------------------------------------------------------------------

def test_library_rejects_oversized_video_before_forwarding(tmp_path):
    class FakeWorker:
        max_bytes = 100

        def submit(self, stream, filename, name=None):
            raise AssertionError("the oversized upload must not be forwarded")

    app = L.create_library_app(str(tmp_path), worker_client=FakeWorker())
    with TestClient(app) as client:
        response = client.post("/api/videos",
                               files={"file": ("a.mp4", io.BytesIO(b"x" * 101))})
    assert response.status_code == 413
    assert "100 bytes" in response.json()["error"]
    assert not list(tmp_path.glob(".video-upload-*"))


def test_library_rejects_oversized_match_upload(tmp_path, monkeypatch):
    import sideline.matches as matches_mod

    class SmallStore(matches_mod.MatchStore):
        def __init__(self, root, max_bytes=1000, smooth=False):
            super().__init__(root, max_bytes=max_bytes, smooth=smooth)

    monkeypatch.setattr(matches_mod, "MatchStore", SmallStore)
    app = L.create_library_app(str(tmp_path))
    with TestClient(app) as client:
        limit = client.get("/api/limits").json()["max_upload_bytes"]
        assert limit == 1000
        # one byte over the limit: rejected from the Content-Length,
        # before the body is read (nothing is stored)
        payload = b"0" * (limit + 1)
        response = client.post("/api/matches",
                               files={"file": ("big.jsonl", io.BytesIO(payload))})
    assert response.status_code == 413
    assert "maximum size" in response.json()["error"]
    assert not list(tmp_path.glob("*.jsonl"))


def test_library_counts_video_uploads(tmp_path):
    class FakeWorker:
        max_bytes = 10000

        def submit(self, stream, filename, name=None):
            data = stream.read()
            self.sent = len(data)
            return "j_aabbccddeeff"

    app = L.create_library_app(str(tmp_path), worker_client=FakeWorker())
    with TestClient(app) as client:
        response = client.post("/api/videos",
                               files={"file": ("a.mp4", io.BytesIO(b"x" * 42))})
        assert response.status_code == 202
        assert response.json()["job_id"] == "j_aabbccddeeff"
        text = client.get("/metrics").text
    assert "sideline_uploads 1" in text
    assert "sideline_upload_bytes 42" in text


def test_worker_upload_limit_413(tmp_path):
    app = _worker_app(tmp_path, max_bytes=10)
    with TestClient(app) as client:
        response = client.post("/jobs", files={"file": ("a.mp4", b"x" * 11)})
    assert response.status_code == 413
    assert "maximum size of 10 bytes" in response.json()["error"]


def test_client_connect_and_idle_timeouts():
    assert V._CONNECT_TIMEOUT == 30.0
    assert V._IDLE_TIMEOUT == 120.0
    client = V.VideoWorkerClient("http://127.0.0.1:9111")
    assert client.timeout == 30.0
    assert client.idle_timeout == 120.0
    custom = V.VideoWorkerClient("http://127.0.0.1:9111",
                                 timeout=5, idle_timeout=7)
    assert custom.timeout == 5
    assert custom.idle_timeout == 7

    class FakeSocket:
        def __init__(self):
            self.timeout = 30.0  # the connect timeout, set by http.client

        def settimeout(self, value):
            self.timeout = value

    class FakeConn:
        def __init__(self, socket):
            self.sock = socket

    socket = FakeSocket()
    V._apply_idle_timeout(FakeConn(socket), 7.0)
    assert socket.timeout == 7.0
    # a connection without a socket (a test double) is kept as is
    V._apply_idle_timeout(object(), 7.0)


def test_client_sends_the_bearer_token_to_the_worker():
    client = V.VideoWorkerClient("http://127.0.0.1:9111", token=TOKEN)
    assert client.token == TOKEN
    headers = []

    class FakeConn:
        def __init__(self, *args, **kwargs):
            pass

        def putrequest(self, method, path, *args):
            pass

        def putheader(self, name, value):
            headers.append((name, value))

        def endheaders(self, *args):
            pass

        def getresponse(self):
            raise OSError("the connection is a stub")

        def close(self):
            pass

    import http.client
    with mock.patch.object(http.client, "HTTPConnection", FakeConn):
        try:
            client.status("j_1234567890ab")
        except Exception:
            pass
    assert ("Authorization", "Bearer " + TOKEN) in headers


# ---------------------------------------------------------------------------
# 6. graceful shutdown
# ---------------------------------------------------------------------------

class _FakeProc:
    """A Popen double: a pid (so the group-kill path is taken) and a kill
    that records the call instead of signalling."""

    def __init__(self, pid=4242):
        self.pid = pid
        self.returncode = None
        self.kills = 0

    def kill(self):
        self.kills += 1
        self.returncode = -9

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def test_worker_shutdown_fails_running_job_and_refuses_new_ones(tmp_path):
    release = threading.Event()

    def runner(video, out, on_progress, cancel=None):
        cancel.wait(timeout=10)
        out.write_text("{}\n")

    app = _worker_app(tmp_path, runner=runner)
    with TestClient(app) as client:
        job_id = client.post("/jobs",
                             files={"file": ("a.mp4", b"123")}).json()["job_id"]
        worker = app.state.worker
        assert _wait_for(lambda: worker.get(job_id).status == "running")
        started = time.monotonic()
        worker.shutdown()
        assert time.monotonic() - started < 5.0
        job = worker.get(job_id)
        assert job.status == "failed"
        assert job.error == "worker stopped"
        # new jobs are refused from here on
        response = client.post("/jobs", files={"file": ("b.mp4", b"123")})
        assert response.status_code == 503
        assert "stopping" in response.json()["error"]
        # idempotent
        worker.shutdown()


def test_default_runner_registers_and_kills_the_process_group(tmp_path, monkeypatch):
    killpg_calls = []

    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        if kwargs.get("start_new_session") is not True:
            return real_popen(cmd, **kwargs)  # e.g. the ffprobe duration probe
        assert kwargs.get("start_new_session") is True
        return _FakeProc(pid=777)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(W.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(W.os, "killpg",
                        lambda pgid, sig: killpg_calls.append((pgid, sig)))
    monkeypatch.setattr(W, "_POLL_SECONDS", 0.01)
    worker = W.Worker(str(tmp_path)).start()
    try:
        loop = asyncio.new_event_loop()
        job = loop.run_until_complete(
            worker.upload(_AsyncUpload(b"123", "a.mp4")))
        loop.close()
        assert _wait_for(lambda: worker.get(job.id).status == "running")

        def proc_registered():
            with worker._lock:
                return worker._running_proc is not None

        # the default runner resolves its own path before spawning (slow on
        # some sandboxes); give it room before asserting the registration
        assert _wait_for(
            lambda: proc_registered() or worker.get(job.id).status != "running",
            timeout=10.0)
        assert proc_registered()
        assert killpg_calls == []
        worker.shutdown()
        assert _wait_for(lambda: worker.get(job.id).status == "failed")
        assert worker.get(job.id).error == "worker stopped"
        assert killpg_calls and all(
            call == (777, signal.SIGKILL) for call in killpg_calls)
    finally:
        worker.shutdown()


def test_worker_kill_falls_back_when_the_group_is_gone(tmp_path, monkeypatch):
    proc = _FakeProc()

    def gone(pid):
        raise OSError("the process is already gone")

    monkeypatch.setattr(W.os, "getpgid", gone)
    monkeypatch.setattr(W.os, "killpg",
                        lambda pgid, sig: pytest.fail("group must be gone"))
    W._kill(proc)
    assert proc.kills == 1


def test_cli_sigterm_handler_shuts_down_the_worker(tmp_path, monkeypatch,
                                                   make_uvicorn):
    uvicorn = make_uvicorn()
    token_file = _token_file(tmp_path)
    created = {}
    real_factory = W.create_worker_app

    def factory(root, runner=None, max_bytes=None, token=None):
        app = real_factory(str(root), runner=_blocking_runner, token=token)
        created["app"] = app
        return app

    monkeypatch.setattr(W, "_POLL_SECONDS", 0.01)
    with mock.patch.object(W, "create_worker_app", factory):
        M._cmd_worker(M.build_parser().parse_args(
            ["worker", "--root", str(tmp_path),
             "--token-file", str(token_file)]))
    assert uvicorn.run.called
    worker = created["app"].state.worker
    handler = signal.getsignal(signal.SIGTERM)
    assert handler is not None and callable(handler)
    headers = {"Authorization": "Bearer " + TOKEN}
    with TestClient(created["app"]) as client:
        job_id = client.post("/jobs",
                             files={"file": ("a.mp4", b"123")},
                             headers=headers).json()["job_id"]
        assert _wait_for(lambda: worker.get(job_id).status == "running")
        handler(signal.SIGTERM, None)
    assert _wait_for(lambda: worker.get(job_id).status == "failed")
    assert worker.get(job_id).error == "worker stopped"


def _blocking_runner(video, out, on_progress, cancel=None):
    cancel.wait(timeout=10)
    out.write_text("{}\n")


# ---------------------------------------------------------------------------
# 7. security headers and CSP
# ---------------------------------------------------------------------------

def _csp_hash_of_inline_script(html):
    match = re.search(r'<script type="module">([\s\S]*?)</script>', html)
    assert match, "the page keeps its inline module script"
    digest = hashlib.sha256(match.group(1).encode("utf-8")).digest()
    return "sha256-" + base64.b64encode(digest).decode("ascii")


def test_pages_send_security_headers_and_per_page_csp(tmp_path):
    app = L.create_library_app(str(tmp_path))
    with TestClient(app) as client:
        lib = client.get("/library.html")
        match = client.get("/match.html")
    for response in (lib, match):
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
    csp = lib.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "style-src 'self'" in csp
    assert "'unsafe-inline'" not in csp
    assert _csp_hash_of_inline_script(lib.text) in csp
    # each page pins its own inline script (different hashes)
    match_hash = _csp_hash_of_inline_script(match.text)
    assert match_hash in match.headers["content-security-policy"]
    assert match_hash != _csp_hash_of_inline_script(lib.text)


def test_pages_are_served_byte_for_byte(tmp_path):
    app = L.create_library_app(str(tmp_path))
    web = os.path.join(os.path.dirname(L.__file__), "web")
    with TestClient(app) as client:
        for name in ("library.html", "match.html"):
            served = client.get("/" + name).content
            on_disk = open(os.path.join(web, name), "rb").read()
            assert served == on_disk


def test_match_css_is_external_and_served(tmp_path):
    app = L.create_library_app(str(tmp_path))
    web = os.path.join(os.path.dirname(L.__file__), "web")
    html = open(os.path.join(web, "match.html"), encoding="utf-8").read()
    assert "<style" not in html
    assert '<link rel="stylesheet" href="/static/match.css">' in html
    css = open(os.path.join(web, "match.css"), encoding="utf-8").read()
    assert len(css.strip()) > 100  # the moved styles are in the file
    with TestClient(app) as client:
        response = client.get("/static/match.css")
        assert response.status_code == 200
        assert response.text == css


def test_pages_have_no_inline_event_handlers(tmp_path):
    app = L.create_library_app(str(tmp_path))
    web = os.path.join(os.path.dirname(L.__file__), "web")
    with TestClient(app) as client:
        for name in ("library.html", "match.html"):
            html = client.get("/" + name).text
            # an HTML attribute that starts with on<...>= is an inline
            # event handler (no other HTML attribute names look like that)
            assert not re.search(r"\son[a-z]+\s*=", html, re.I), name
    # and the scripts the two pages load do not assign handlers onto
    # elements either (app.js belongs to the live stream viewer, a
    # different service)
    scripts = set()

    def collect(text):
        for url in re.findall(r'<script[^>]+src="(/[^"]+)"', text):
            scripts.add(url.lstrip("/"))
        for url in re.findall(r'from\s+["\'](/[^"\']+)["\']', text):
            scripts.add(url.lstrip("/"))

    for name in ("library.html", "match.html"):
        collect(open(os.path.join(web, name), encoding="utf-8").read())
    todo = list(scripts)

    def to_path(url):
        # the /static mount is the web folder itself
        rel = url.lstrip("/")
        if rel.startswith("static/"):
            rel = rel[len("static/"):]
        return os.path.join(web, rel)

    while todo:
        rel = todo.pop()
        path = to_path(rel)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        before = set(scripts)
        collect(text)
        todo += sorted(scripts - before)
    assert scripts, "the pages load their modules from /static"
    for rel in sorted(scripts):
        text = open(to_path(rel), encoding="utf-8").read()
        assert not re.search(r"\.\s*on[a-z]+\s*=", text, re.I), rel


# ---------------------------------------------------------------------------
# 8. doctor
# ---------------------------------------------------------------------------

def test_doctor_ok_rows_exit_zero(tmp_path, capsys):
    token_file = _token_file(tmp_path)
    root = tmp_path / "store"
    with _fake_disk_usage(10 * 1024 ** 3):
        M._cmd_doctor(M.build_parser().parse_args(
            ["doctor", "--root", str(root), "--token-file", str(token_file)]))
    out = capsys.readouterr().out
    for name in ("python version", "fastapi installed", "writable root",
                 "free disk space", "worker reachable", "token file"):
        assert name in out
    assert "FAIL" not in out
    assert re.search(r"python version\s+ok\s+3\.", out)
    assert re.search(r"worker reachable\s+skip", out)
    assert re.search(r"token file\s+ok", out)


def test_doctor_fails_on_bad_token_permissions(tmp_path, capsys):
    bad = _token_file(tmp_path, mode=0o644, name="loose")
    with _fake_disk_usage(10 * 1024 ** 3):
        with pytest.raises(SystemExit) as exc:
            M._cmd_doctor(M.build_parser().parse_args(
                ["doctor", "--root", str(tmp_path / "s"),
                 "--token-file", str(bad)]))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert re.search(r"token file\s+FAIL", out)


def test_doctor_fails_on_low_disk(tmp_path, capsys):
    with _fake_disk_usage(1 * 1024 ** 3):
        with pytest.raises(SystemExit) as exc:
            M._cmd_doctor(M.build_parser().parse_args(
                ["doctor", "--root", str(tmp_path / "s")]))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert re.search(r"free disk space\s+FAIL", out)


def test_doctor_fails_on_unwritable_root(tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the root's parent should be")
    with _fake_disk_usage(10 * 1024 ** 3):
        with pytest.raises(SystemExit) as exc:
            M._cmd_doctor(M.build_parser().parse_args(
                ["doctor", "--root", str(blocker / "child")]))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert re.search(r"writable root\s+FAIL", out)


def test_doctor_worker_unreachable_fails(tmp_path, capsys):
    with _fake_disk_usage(10 * 1024 ** 3):
        with pytest.raises(SystemExit) as exc:
            M._cmd_doctor(M.build_parser().parse_args(
                ["doctor", "--root", str(tmp_path / "s"),
                 "--worker", "not-a-url"]))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert re.search(r"worker reachable\s+FAIL", out)


def test_doctor_worker_reachable_ok(tmp_path, capsys):
    import urllib.request

    class FakeResponse:
        status = 200

        def read(self):
            return b'{"status": "ok"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with _fake_disk_usage(10 * 1024 ** 3), \
            mock.patch.object(urllib.request, "urlopen",
                              return_value=FakeResponse()):
        M._cmd_doctor(M.build_parser().parse_args(
            ["doctor", "--root", str(tmp_path / "s"),
             "--worker", "http://127.0.0.1:9111"]))
    out = capsys.readouterr().out
    assert re.search(r"worker reachable\s+ok", out)
    assert "FAIL" not in out
