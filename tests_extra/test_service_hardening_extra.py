"""Extra adversarial tests for service hardening (tasks/service_hardening.md).

The acceptance suite (tests/test_hardening.py) covers the happy paths; these
tests probe the specification's boundary values and the rules it does not
check there:

- auth: the exact ``Authorization: Bearer <token>`` form (wrong case, a
  doubled space, trailing whitespace and a same-length wrong token are all
  401s); the 31/32-character token-file boundary; a token file that holds
  two tokens (or none at all) is refused; an other-readable (mode 604)
  token file is refused; a file written by ``python -m sideline token``
  round-trips into the worker CLI unchanged; HEAD /health is not on the
  exemption list (only GET /health is open);
- write protection: with an admin token, POST /api/videos and
  DELETE /api/videos/{id} are gated while GETs (including
  GET /api/videos/{id} and GET /api/limits) stay open; without an admin
  token the writes stay open; an admin token allows a non-loopback host
  without --allow-open, and the default 127.0.0.1 host needs nothing;
- structured logs: the worker also emits one JSON line per request,
  including 401/404/413 answers; the logged line of an upload carries no
  token, file name or body; the X-Request-Id is echoed on 401 and on 429
  answers too;
- metrics: counters (not gauges) never go down within a process across a
  mixed ok/failed workload; the 4xx class is counted; the jobs gauge
  includes the failed state; the seconds spent processing accumulate; the
  registry is valid Prometheus text on both services; the library's
  matches-stored and store-bytes gauges track the store;
- limits: the exact Content-Length boundary (exactly the limit is
  accepted, one byte over is a 413 before the body is read and nothing is
  stored); a chunked (no Content-Length) oversized video is rejected
  before it is forwarded to the worker; per-client rate limit: exactly 60
  non-GET requests pass and the 61st is a 429 with a positive integer
  Retry-After; GETs are never counted; distinct client keys have distinct
  budgets; the sliding window releases exactly at the window edge;
  concurrent requests never admit more than the limit;
- client timeouts: the connection is opened with the 30 s connect timeout
  and the socket is switched to the 120 s idle timeout after the request
  headers go out, on plain requests and on chunked uploads;
- graceful shutdown: even when the running job's runner never honours the
  cancel event, the worker's shutdown is bounded (it can always exit
  within the spec's 10 s), the job stays failed with "worker stopped",
  the late runner answer must not overwrite that state, and new uploads
  answer 503 from the shutdown on;
- security headers: the CSP's img-src is 'self' data: and the script-src
  allows only 'self' plus the pinned hash (no remote, no unsafe-inline);
  each page keeps exactly one inline script (the pinned module) and no
  <style> element;
- doctor: a reachable worker whose health is not ok fails; free disk
  exactly at the 5 GiB threshold fails (the spec says "above 5 GB") while
  one byte more passes; a 31-character token file fails and a 32-character
  one passes; a missing token file fails; a missing FastAPI fails.

The sandbox refuses to send signals to other processes, so the process
level of the graceful-shutdown rule is covered through the Worker's
bounded shutdown instead of a real SIGTERM (as in the acceptance suite).
"""

import json
import os
import re
import shutil
import stat
import sys
import threading
import time
import types
import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sideline import worker as W  # noqa: E402
from sideline import library as L  # noqa: E402
from sideline import videos as V  # noqa: E402
import sideline.__main__ as M  # noqa: E402
import sideline.matches as matches_mod  # noqa: E402
import http.client  # noqa: E402

TOKEN = "extra-worker-" + "a" * 40
ADMIN = "extra-admin-" + "b" * 40


def _token_file(tmp_path, token, mode=0o600, name="token"):
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


def _wait_for(predicate, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _log_records(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def _fake_disk_usage(free):
    return mock.patch.object(shutil, "disk_usage",
                             return_value=types.SimpleNamespace(
                                 total=2 * free, used=free, free=free))


def _multipart(payload, filename):
    """A hand-built one-field multipart body, so its exact size (and thus
    the Content-Length the limit check sees) is known byte for byte."""
    head = ('--B0\r\n'
            'Content-Disposition: form-data; name="file"; filename="%s"\r\n'
            '\r\n' % filename).encode("utf-8")
    return head + payload + b"\r\n--B0--\r\n"


# ---------------------------------------------------------------------------
# 1. worker authentication
# ---------------------------------------------------------------------------

def test_token_file_length_boundary_31_refused_32_accepted(tmp_path, monkeypatch):
    root = tmp_path / "w"
    short = _token_file(tmp_path, token="s" * 31, name="short")
    with pytest.raises(SystemExit) as exc:
        M._cmd_worker(M.build_parser().parse_args(
            ["worker", "--root", str(root), "--token-file", str(short)]))
    assert "32" in str(exc.value)

    ok = _token_file(tmp_path, token="k" * 32, name="ok")
    uvicorn = mock.MagicMock(name="uvicorn")
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    created = {}

    def factory(root, runner=None, max_bytes=None, token=None):
        created["token"] = token
        return mock.sentinel.app

    monkeypatch.setattr(W, "create_worker_app", factory)
    M._cmd_worker(M.build_parser().parse_args(
        ["worker", "--root", str(root), "--token-file", str(ok)]))
    assert created["token"] == "k" * 32


def test_token_file_must_hold_one_token(tmp_path):
    root = tmp_path / "w"
    two = tmp_path / "two"
    fd = os.open(str(two), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("t" * 33 + "\n" + "u" * 33 + "\n")
    with pytest.raises(SystemExit) as exc:
        M._cmd_worker(M.build_parser().parse_args(
            ["worker", "--root", str(root), "--token-file", str(two)]))
    assert "one token" in str(exc.value)

    empty = tmp_path / "empty"
    fd = os.open(str(empty), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.close(fd)
    with pytest.raises(SystemExit) as exc:
        M._cmd_worker(M.build_parser().parse_args(
            ["worker", "--root", str(root), "--token-file", str(empty)]))
    assert "32" in str(exc.value)


def test_other_readable_token_file_refused(tmp_path):
    bad = _token_file(tmp_path, token="o" * 40, mode=0o604, name="other-readable")
    with pytest.raises(SystemExit) as exc:
        M._cmd_worker(M.build_parser().parse_args(
            ["worker", "--root", str(tmp_path), "--token-file", str(bad)]))
    message = str(exc.value)
    assert "readable" in message and "600" in message


def test_token_subcommand_file_round_trips_into_the_worker(tmp_path, monkeypatch):
    target = tmp_path / "new" / "token"
    monkeypatch.setattr(sys, "argv", ["sideline", "token", str(target)])
    M.main()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600

    uvicorn = mock.MagicMock(name="uvicorn")
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    created = {}

    def factory(root, runner=None, max_bytes=None, token=None):
        created["token"] = token
        return mock.sentinel.app

    monkeypatch.setattr(W, "create_worker_app", factory)
    M._cmd_worker(M.build_parser().parse_args(
        ["worker", "--root", str(tmp_path / "w"), "--token-file", str(target)]))
    token = target.read_text(encoding="utf-8").strip()
    assert len(token) == 64
    assert created["token"] == token


def test_bearer_header_format_is_strict(tmp_path):
    app = _worker_app(tmp_path, token=TOKEN)
    try:
        client = TestClient(app)
        assert client.get("/metrics",
                          headers={"Authorization": "Bearer " + TOKEN}).status_code == 200
        # the spec's header is `Authorization: Bearer <token>`; anything
        # else is a wrong token
        assert client.get("/metrics",
                          headers={"Authorization": "bearer " + TOKEN}).status_code == 401
        assert client.get("/metrics",
                          headers={"Authorization": "Bearer  " + TOKEN}).status_code == 401
        assert client.get("/metrics",
                          headers={"Authorization": "Bearer " + TOKEN + " "}).status_code == 401
        wrong = TOKEN[:-1] + ("b" if TOKEN[-1] != "b" else "a")
        assert client.get("/metrics",
                          headers={"Authorization": "Bearer " + wrong}).status_code == 401
    finally:
        app.state.worker.stop()


def test_health_is_the_only_exempt_route(tmp_path):
    app = _worker_app(tmp_path, token=TOKEN)
    try:
        client = TestClient(app)
        # GET /health is open
        assert client.get("/health").status_code == 200
        # HEAD is not GET: the exemption is `GET /health` only
        assert client.head("/health").status_code == 401
        # and every other route needs the token, with or without a body
        assert client.get("/jobs/j_1234567890ab").status_code == 401
        assert client.delete("/jobs/j_1234567890ab").status_code == 401
    finally:
        app.state.worker.stop()


def test_repeated_uploads_with_token_all_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_probe_duration", lambda path: None)

    def runner(video, out, on_progress):
        out.write_text("{}\n")
        on_progress(1)

    app = _worker_app(tmp_path, runner=runner, token=TOKEN)
    try:
        client = TestClient(app)
        headers = {"Authorization": "Bearer " + TOKEN}
        ids = []
        for i in range(5):
            response = client.post("/jobs",
                                   files={"file": ("v%d.mp4" % i, b"v" * (i + 1))},
                                   headers=headers)
            assert response.status_code == 202
            ids.append(response.json()["job_id"])
        assert len(set(ids)) == 5
        for job_id in ids:
            assert _wait_for(
                lambda: client.get("/jobs/" + job_id, headers=headers)
                .json()["status"] == "done")
        assert client.get("/health", headers=headers).json() == {
            "status": "ok", "busy": False, "queued": 0}
    finally:
        app.state.worker.stop()


# ---------------------------------------------------------------------------
# 2. library write protection
# ---------------------------------------------------------------------------

def test_library_video_routes_need_the_admin_token(tmp_path):
    # with a worker: POST /api/videos is gated
    class FakeWorker:
        max_bytes = 100000

        def submit(self, stream, filename, name=None):
            stream.read()
            return "j_111111111111"

        def status(self, job_id):
            return {"job_id": job_id, "status": "queued", "progress": 0.0,
                    "frames": 0, "name": "n", "error": None,
                    "created_at": "2026-01-01T00:00:00Z", "seconds": None}

        def delete(self, job_id):
            return None

    app = L.create_library_app(str(tmp_path), worker_client=FakeWorker(),
                               admin_token=ADMIN)
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + ADMIN}
    assert client.post("/api/videos",
                       files={"file": ("a.mp4", b"123")}).status_code == 401
    response = client.post("/api/videos", files={"file": ("a.mp4", b"123")},
                           headers=headers)
    assert response.status_code == 202
    assert response.json() == {"job_id": "j_111111111111", "status": "queued"}
    assert client.delete("/api/videos/j_111111111111").status_code == 401
    assert client.delete("/api/videos/j_111111111111",
                         headers=headers).status_code == 204

    # without a worker: GETs stay open, the DELETE needs the token
    app2 = L.create_library_app(str(tmp_path / "r2"), admin_token=ADMIN)
    client2 = TestClient(app2)
    assert client2.get("/api/limits").status_code == 200
    assert client2.get("/api/matches").status_code == 200
    # an unknown job is a 404, not a 401: GETs are open
    assert client2.get("/api/videos/j_222222222222").status_code == 404
    assert client2.delete("/api/videos/j_222222222222").status_code == 401
    assert client2.delete("/api/videos/j_222222222222",
                          headers={"Authorization": "Bearer " + ADMIN}).status_code == 404


def test_library_no_admin_token_keeps_writes_open(tmp_path):
    app = L.create_library_app(str(tmp_path))
    client = TestClient(app)
    # no Authorization header at all
    response = client.post("/api/matches",
                           files={"file": ("a.jsonl",
                                           b'{"frame": 0, "people": []}\n')})
    assert response.status_code == 201


def test_library_admin_token_allows_non_loopback_host(tmp_path, monkeypatch):
    uvicorn = mock.MagicMock(name="uvicorn")
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    token_file = _token_file(tmp_path, token=ADMIN, name="admin")
    created = {}

    def factory(root, worker_client=None, admin_token=None):
        created["admin_token"] = admin_token
        return mock.sentinel.app

    monkeypatch.setattr(L, "create_library_app", factory)
    M._cmd_library(M.build_parser().parse_args(
        ["library", "--root", str(tmp_path), "--host", "0.0.0.0",
         "--port", "9321", "--admin-token-file", str(token_file)]))
    assert created["admin_token"] == ADMIN
    assert uvicorn.run.call_args.kwargs["host"] == "0.0.0.0"


def test_library_default_loopback_host_without_token_runs(tmp_path, monkeypatch):
    uvicorn = mock.MagicMock(name="uvicorn")
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    with mock.patch.object(L, "create_library_app",
                           return_value=mock.sentinel.app):
        M._cmd_library(M.build_parser().parse_args(
            ["library", "--root", str(tmp_path)]))
    assert uvicorn.run.call_args.kwargs["host"] == "127.0.0.1"


def test_library_401_body_is_json_error(tmp_path):
    app = L.create_library_app(str(tmp_path), admin_token=ADMIN)
    client = TestClient(app)
    response = client.post("/api/matches", content=b"",
                           headers={"content-type": "application/octet-stream"})
    assert response.status_code == 401
    body = response.json()
    assert set(body) == {"error"}
    assert body["error"]


# ---------------------------------------------------------------------------
# 3. structured logs
# ---------------------------------------------------------------------------

def test_worker_logs_one_json_line_per_request_with_status(tmp_path, capsys):
    app = _worker_app(tmp_path, token=TOKEN, max_bytes=10)
    try:
        client = TestClient(app)
        headers = {"Authorization": "Bearer " + TOKEN}
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 401  # no token
        assert client.get("/jobs/j_1234567890ab", headers=headers).status_code == 404
        upload = client.post("/jobs",
                             files={"file": ("leak-me.mp4", b"x" * 11)},
                             headers=headers)
        assert upload.status_code == 413
    finally:
        app.state.worker.stop()
    out = capsys.readouterr().out
    records = _log_records(out)
    assert [rec["status"] for rec in records] == [200, 401, 404, 413]
    assert [rec["path"] for rec in records] == [
        "/health", "/metrics", "/jobs/j_1234567890ab", "/jobs"]
    assert [rec["method"] for rec in records] == ["GET", "GET", "GET", "POST"]
    for rec in records:
        assert set(rec) == {"ts", "method", "path", "status", "ms",
                            "bytes_in", "request_id"}
    assert records[3]["bytes_in"] > 0  # the upload's Content-Length
    # the log line never carries the token, the uploaded file name or the body
    assert TOKEN not in out
    assert "leak-me" not in out
    assert "Bearer" not in out


def test_request_id_echoed_on_401_and_429(tmp_path):
    app = _worker_app(tmp_path, token=TOKEN)
    try:
        client = TestClient(app)
        response = client.get("/metrics", headers={"X-Request-Id": "echo-401"})
        assert response.status_code == 401
        assert response.headers["x-request-id"] == "echo-401"
    finally:
        app.state.worker.stop()

    app2 = _worker_app(tmp_path / "r2")
    try:
        client2 = TestClient(app2)
        for _ in range(60):
            assert client2.post("/jobs", data={"x": "y"}).status_code == 400
        response = client2.post("/jobs", data={"x": "y"},
                                headers={"X-Request-Id": "echo-429"})
        assert response.status_code == 429
        assert response.headers["x-request-id"] == "echo-429"
        assert int(response.headers["retry-after"]) >= 1
    finally:
        app2.state.worker.stop()


def test_library_log_never_contains_filename_or_body(tmp_path, capsys):
    app = L.create_library_app(str(tmp_path))
    client = TestClient(app)
    body = b'{"frame": 0, "people": [{"id": "BODY-MARKER-ZZZ"}]}\n'
    response = client.post("/api/matches", files={"file": ("leaky-name.jsonl", body)})
    assert response.status_code == 201
    out = capsys.readouterr().out
    assert "leaky-name" not in out
    assert "BODY-MARKER-ZZZ" not in out
    records = _log_records(out)
    assert len(records) == 1
    assert records[0]["status"] == 201
    assert records[0]["bytes_in"] == int(response.request.headers["content-length"])


# ---------------------------------------------------------------------------
# 4. metrics
# ---------------------------------------------------------------------------

def _parse_prom(text):
    """(types, values): name -> metric type, (name, labels) -> value."""
    types = {}
    values = {}
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            parts = line.split()
            types[parts[2]] = parts[3]
            continue
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})? (-?[0-9.eE+]+)$", line)
        assert match, "not a prometheus sample line: %r" % line
        labels = ()
        if match.group(2):
            labels = tuple(sorted(
                tuple(pair.split("=", 1))
                for pair in match.group(2)[1:-1].split(", ")))
        values[(match.group(1), labels)] = float(match.group(3))
    return types, values


def test_worker_counters_never_go_down_across_mixed_workload(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_probe_duration", lambda path: None)

    def runner(video, out, on_progress):
        out.write_text("{}\n{}\n")
        on_progress(2)

    app = _worker_app(tmp_path, runner=runner, max_bytes=10)
    try:
        client = TestClient(app)
        assert client.get("/health").status_code == 200  # warm the registry
        first_types, first = _parse_prom(client.get("/metrics").text)
        assert client.get("/jobs/j_1234567890ab").status_code == 404
        assert client.post("/jobs",
                           files={"file": ("a.mp4", b"x" * 11)}).status_code == 413
        assert client.post("/jobs", data={"name": "no file"}).status_code == 400
        upload = client.post("/jobs", files={"file": ("b.mp4", b"y" * 7)})
        assert upload.status_code == 202
        job_id = upload.json()["job_id"]
        assert _wait_for(lambda: client.get("/jobs/" + job_id).json()["status"] == "done")
        second_types, second = _parse_prom(client.get("/metrics").text)
        # counters never go down within a process
        for key, value in first.items():
            name = key[0]
            if second_types.get(name) != "counter":
                continue  # gauges may move either way
            assert second.get(key) is not None, "counter %r disappeared" % (key,)
            assert second[key] >= value, (key, value, second.get(key))
        # the failed answers landed in the 4xx class
        assert second[("sideline_requests", (("class", '"4xx"'),))] >= 3
        # exactly one accepted upload, with its bytes
        assert second[("sideline_uploads", ())] - first.get(("sideline_uploads", ()), 0.0) == 1
        assert second[("sideline_upload_bytes", ())] - first.get(("sideline_upload_bytes", ()), 0.0) == 7
    finally:
        app.state.worker.stop()


def test_worker_metrics_shows_failed_state(tmp_path):
    def runner(video, out, on_progress):
        raise RuntimeError("boom")

    app = _worker_app(tmp_path, runner=runner)
    try:
        client = TestClient(app)
        job_id = client.post("/jobs", files={"file": ("a.mp4", b"1")}).json()["job_id"]
        assert _wait_for(lambda: client.get("/jobs/" + job_id).json()["status"] == "failed")
        text = client.get("/metrics").text
        assert 'sideline_jobs{state="failed"} 1' in text
    finally:
        app.state.worker.stop()


def test_worker_processing_seconds_accumulates(tmp_path):
    def runner(video, out, on_progress):
        time.sleep(0.2)
        out.write_text("{}\n")

    app = _worker_app(tmp_path, runner=runner)
    try:
        client = TestClient(app)
        job_id = client.post("/jobs", files={"file": ("a.mp4", b"1")}).json()["job_id"]
        assert _wait_for(lambda: client.get("/jobs/" + job_id).json()["status"] == "done")
        match = re.search(r"^sideline_processing_seconds (\S+)$",
                          client.get("/metrics").text, re.M)
        assert match, "sideline_processing_seconds missing from the registry"
        assert float(match.group(1)) >= 0.1
    finally:
        app.state.worker.stop()


def test_metrics_are_valid_prometheus_text(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_probe_duration", lambda path: None)
    app = _worker_app(tmp_path)
    try:
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        assert client.get("/jobs/j_1234567890ab").status_code == 404
        worker_text = client.get("/metrics").text
    finally:
        app.state.worker.stop()

    lib_app = L.create_library_app(str(tmp_path / "lib"))
    lib_client = TestClient(lib_app)
    assert lib_client.get("/healthz").status_code == 200
    library_text = lib_client.get("/metrics").text

    sample = re.compile(
        r"^([a-zA-Z_:][a-zA-Z0-9_:]*)"
        r"(\{[a-zA-Z_][a-zA-Z0-9_]*=\"[^\"]*\""
        r"(, [a-zA-Z_][a-zA-Z0-9_]*=\"[^\"]*\")*\})? "
        r"([-+]?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?|[-+]?\.[0-9]+([eE][-+]?[0-9]+)?"
        r"|-?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?|-?[0-9]+[eE][-+]?[0-9]+)$")
    for text, what in ((worker_text, "worker"), (library_text, "library")):
        assert text.endswith("\n"), what
        announced = set()
        rendered = 0
        for line in text.splitlines():
            if line.startswith("# HELP "):
                announced.add(line.split()[2])
            elif line.startswith("# TYPE "):
                parts = line.split()
                assert parts[3] in ("counter", "gauge"), line
                assert parts[2] in announced, "TYPE before HELP: " + line
            else:
                match = sample.match(line)
                assert match, what + ": bad sample line: " + line
                name = match.group(1)
                assert name in announced, what + ": sample before HELP/TYPE: " + line
                rendered += 1
        assert rendered > 0, what


def test_library_metrics_matches_and_bytes_track_the_store(tmp_path):
    app = L.create_library_app(str(tmp_path))
    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    first_types, first = _parse_prom(client.get("/metrics").text)
    match = b'{"frame": 0, "people": []}\n{"frame": 1, "people": []}\n'
    response = client.post("/api/matches", files={"file": ("a.jsonl", match)})
    assert response.status_code == 201
    mid_text = client.get("/metrics").text
    assert "sideline_matches_stored 1" in mid_text
    store_bytes = re.search(r"^sideline_store_bytes (\S+)$", mid_text, re.M)
    assert store_bytes and float(store_bytes.group(1)) >= len(match)
    _, mid = _parse_prom(mid_text)
    for key, value in first.items():
        if first_types[key[0]] == "counter":
            assert mid.get(key, 0.0) >= value, key
    match_id = response.json()["id"]
    assert client.delete("/api/matches/" + match_id).status_code == 204
    end_text = client.get("/metrics").text
    assert "sideline_matches_stored 0" in end_text
    assert "sideline_store_bytes 0" in end_text
    # the request counters kept their monotonic increase
    end_types, end = _parse_prom(end_text)
    for key, value in first.items():
        if first_types[key[0]] == "counter":
            assert end.get(key, 0.0) >= value, key
    assert end[("sideline_requests", (("class", '"2xx"'),))] >= 3  # upload, delete, metrics


# ---------------------------------------------------------------------------
# 5. limits and timeouts
# ---------------------------------------------------------------------------

def test_library_size_limit_exact_boundary(tmp_path, monkeypatch):
    class SmallStore(matches_mod.MatchStore):
        def __init__(self, root, max_bytes=1000, smooth=False):
            super().__init__(root, max_bytes=max_bytes, smooth=smooth)

    monkeypatch.setattr(matches_mod, "MatchStore", SmallStore)
    app = L.create_library_app(str(tmp_path))
    client = TestClient(app)
    content_type = {"content-type": "multipart/form-data; boundary=B0"}

    base = b'{"frame": 0, "people": []}'
    overhead = len(_multipart(b"", "a.jsonl")) - 0
    exact = base + b" " * (1000 - overhead - len(base))
    body = _multipart(exact, "a.jsonl")
    assert len(body) == 1000  # exactly the limit
    response = client.post("/api/matches", content=body, headers=content_type)
    assert response.status_code == 201
    stored = {p.name for p in tmp_path.iterdir()}

    over = _multipart(exact + b" ", "a.jsonl")
    assert len(over) == 1001  # one byte over
    response = client.post("/api/matches", content=over, headers=content_type)
    assert response.status_code == 413
    assert "file exceeds the maximum size of 1000 bytes" in response.json()["error"]
    # the oversized body was rejected before it was read: nothing was stored
    assert {p.name for p in tmp_path.iterdir()} == stored


def test_library_video_limit_exact_boundary(tmp_path):
    class FakeWorker:
        max_bytes = 100
        submitted = []

        def submit(self, stream, filename, name=None):
            FakeWorker.submitted.append(len(stream.read()))
            return "j_1234567890ab"

    app = L.create_library_app(str(tmp_path), worker_client=FakeWorker())
    client = TestClient(app)
    content_type = {"content-type": "multipart/form-data; boundary=B0"}

    overhead = len(_multipart(b"", "a.mp4"))
    exact = _multipart(b"x" * (100 - overhead), "a.mp4")
    assert len(exact) == 100
    response = client.post("/api/videos", content=exact, headers=content_type)
    assert response.status_code == 202
    assert response.json() == {"job_id": "j_1234567890ab", "status": "queued"}
    assert FakeWorker.submitted == [100 - overhead]

    over = _multipart(b"x" * (101 - overhead), "a.mp4")
    assert len(over) == 101
    response = client.post("/api/videos", content=over, headers=content_type)
    assert response.status_code == 413
    assert "maximum size of 100 bytes" in response.json()["error"]
    assert FakeWorker.submitted == [100 - overhead]  # not forwarded


def test_library_chunked_oversized_video_rejected_before_forwarding(tmp_path):
    class FakeWorker:
        max_bytes = 100

        def submit(self, stream, filename, name=None):
            raise AssertionError("an oversized upload must not be forwarded")

    app = L.create_library_app(str(tmp_path), worker_client=FakeWorker())
    client = TestClient(app)
    body = _multipart(b"x" * 120, "a.mp4")  # the file part is 120 > 100
    # a chunked body has no Content-Length: the limit is enforced while the
    # upload is staged, before anything reaches the worker
    response = client.post("/api/videos", content=iter([body[:40], body[40:]]),
                           headers={"content-type":
                                    "multipart/form-data; boundary=B0"})
    assert response.status_code == 413
    assert "maximum size of 100 bytes" in response.json()["error"]


def test_library_chunked_oversized_match_upload_rejected(tmp_path, monkeypatch):
    class SmallStore(matches_mod.MatchStore):
        def __init__(self, root, max_bytes=100, smooth=False):
            super().__init__(root, max_bytes=max_bytes, smooth=smooth)

    monkeypatch.setattr(matches_mod, "MatchStore", SmallStore)
    app = L.create_library_app(str(tmp_path))
    client = TestClient(app)
    body = _multipart(b"x" * 150, "a.jsonl")
    response = client.post("/api/matches", content=iter([body[:40], body[40:]]),
                           headers={"content-type":
                                    "multipart/form-data; boundary=B0"})
    assert response.status_code == 413
    assert "maximum size of 100 bytes" in response.json()["error"]
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("m_")]


def test_worker_rate_limit_61st_non_get_is_429_with_retry_after(tmp_path):
    app = _worker_app(tmp_path)
    try:
        client = TestClient(app)
        codes = [client.post("/jobs", data={"x": "y"}).status_code
                 for _ in range(60)]
        assert codes == [400] * 60  # rejected for lacking a file, not for rate
        response = client.post("/jobs", data={"x": "y"})
        assert response.status_code == 429
        assert "error" in response.json()
        retry_after = int(response.headers["retry-after"])
        assert 1 <= retry_after <= 60
    finally:
        app.state.worker.stop()


def test_rate_limit_counts_only_non_get(tmp_path):
    app = _worker_app(tmp_path)
    try:
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        for _ in range(30):
            assert client.post("/jobs", data={"x": "y"}).status_code == 400
        for _ in range(50):
            assert client.get("/health").status_code == 200
        for _ in range(30):
            assert client.post("/jobs", data={"x": "y"}).status_code == 400
        # 60 non-GET requests went through while the 50 GETs in between
        # were never counted: the next non-GET is the 61st
        assert client.post("/jobs", data={"x": "y"}).status_code == 429
    finally:
        app.state.worker.stop()


def test_library_rate_limit_61st_non_get_is_429(tmp_path):
    app = L.create_library_app(str(tmp_path))
    client = TestClient(app)
    headers = {"content-type": "application/octet-stream"}
    for _ in range(60):
        assert client.post("/api/matches", content=b"",
                           headers=headers).status_code == 400
    response = client.post("/api/matches", content=b"", headers=headers)
    assert response.status_code == 429
    assert "error" in response.json()
    assert int(response.headers["retry-after"]) >= 1


def test_rate_limiter_window_and_per_client_keys():
    limiter = W.RateLimiter(limit=2, window=10.0)
    assert limiter.allow("a", now=0.0) == (True, 0)
    assert limiter.allow("a", now=5.0) == (True, 0)
    allowed, retry = limiter.allow("a", now=6.0)
    assert allowed is False
    assert retry == 4  # the oldest hit (t=0) leaves the window at t=10
    # a different client has its own budget
    assert limiter.allow("b", now=6.0) == (True, 0)
    # the sliding window releases exactly when the oldest hit leaves
    assert limiter.allow("a", now=10.0)[0] is True
    assert limiter.allow("a", now=10.0)[0] is False  # t=5 and t=10 fill it
    assert limiter.allow("a", now=15.0)[0] is True  # t=5 left at t=15


def test_rate_limiter_admits_exactly_limit_under_contention():
    limiter = W.RateLimiter(limit=60, window=60.0)
    results = []
    lock = threading.Lock()

    def hammer():
        for _ in range(40):
            allowed, _ = limiter.allow("client")
            with lock:
                results.append(allowed)

    threads = [threading.Thread(target=hammer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 80
    assert sum(results) == 60  # never more than the limit in the window


# ---------------------------------------------------------------------------
# 5b. client socket timeouts
# ---------------------------------------------------------------------------

def test_client_applies_connect_then_idle_timeout_on_requests(monkeypatch):
    class FakeSocket:
        def __init__(self):
            self.timeout = None

        def settimeout(self, value):
            self.timeout = value

    class FakeConn:
        last = None

        def __init__(self, host, port, timeout=None, **kwargs):
            self.sock = FakeSocket()
            self.connect_timeout = timeout
            FakeConn.last = self

        def putrequest(self, *args):
            pass

        def putheader(self, *args):
            pass

        def endheaders(self, *args, **kwargs):
            pass

        def getresponse(self):
            raise OSError("stub connection")

        def send(self, data):
            pass

        def close(self):
            pass

    monkeypatch.setattr(http.client, "HTTPConnection", FakeConn)
    client = V.VideoWorkerClient("http://127.0.0.1:9111")

    try:
        client.status("j_1234567890ab")
    except V.WorkerClientError:
        pass
    assert FakeConn.last.connect_timeout == 30.0  # opened with the connect timeout
    assert FakeConn.last.sock.timeout == 120.0    # switched to the idle timeout

    class Stream:
        def read(self, size):
            return b""

    FakeConn.last = None
    try:
        client.submit(Stream(), "a.mp4")  # the chunked upload path
    except V.WorkerClientError:
        pass
    assert FakeConn.last.connect_timeout == 30.0
    assert FakeConn.last.sock.timeout == 120.0


# ---------------------------------------------------------------------------
# 6. graceful shutdown
# ---------------------------------------------------------------------------

def test_shutdown_is_bounded_when_the_runner_ignores_cancel(tmp_path):
    """Even if the running job's process never reacts, the shutdown must
    stay bounded (the spec's 10 s exit budget): the join waits only the
    given timeout, the job keeps its "worker stopped" failure, and the
    runner's late answer must not overwrite it."""
    started = threading.Event()

    def runner(video, out, on_progress, cancel=None):
        started.set()
        time.sleep(2.5)  # never honours the cancel event
        out.write_text("{}\n")  # the late answer

    app = _worker_app(tmp_path, runner=runner)
    try:
        client = TestClient(app)
        job_id = client.post("/jobs",
                             files={"file": ("a.mp4", b"1")}).json()["job_id"]
        assert started.wait(5)
        assert _wait_for(
            lambda: client.get("/jobs/" + job_id).json()["status"] == "running")
        started_at = time.monotonic()
        app.state.worker.shutdown(timeout=1.0)
        assert time.monotonic() - started_at < 5.0  # bounded, not stuck
        job = client.get("/jobs/" + job_id).json()
        assert job["status"] == "failed"
        assert job["error"] == "worker stopped"
        # new uploads are refused from the shutdown on
        response = client.post("/jobs", files={"file": ("b.mp4", b"1")})
        assert response.status_code == 503
        assert "stopping" in response.json()["error"]
        # once the unresponsive runner finally returns, its answer must not
        # resurrect the job
        time.sleep(2.5)
        job = client.get("/jobs/" + job_id).json()
        assert job["status"] == "failed"
        assert job["error"] == "worker stopped"
    finally:
        app.state.worker.stop()


# ---------------------------------------------------------------------------
# 7. security headers and the pages
# ---------------------------------------------------------------------------

def test_csp_img_src_and_script_src_are_same_origin(tmp_path):
    app = L.create_library_app(str(tmp_path))
    client = TestClient(app)
    for name in ("library.html", "match.html"):
        response = client.get("/" + name)
        csp = response.headers["content-security-policy"]
        assert "img-src 'self' data:" in csp, name
        match = re.search(r"script-src ([^;]+);", csp)
        assert match, name
        entries = match.group(1).split()
        assert entries, name
        for entry in entries:
            # only same-origin scripts and the pinned inline hash may run
            assert entry == "'self'" or entry.startswith("'sha256-"), (name, entry)
        assert "'unsafe-inline'" not in csp, name
        assert not any(entry.startswith(("http://", "https://"))
                       for entry in entries), name


def test_pages_keep_only_the_pinned_inline_script_and_no_inline_style(tmp_path):
    app = L.create_library_app(str(tmp_path))
    client = TestClient(app)
    for name in ("library.html", "match.html"):
        html = client.get("/" + name).text
        tags = re.findall(r"<script\b[^>]*>", html)
        inline = [tag for tag in tags if not re.search(r"\bsrc\s*=", tag)]
        # the per-page CSP hash pins exactly one inline script: the module
        assert len(inline) == 1, name
        assert 'type="module"' in inline[0], name
        content = re.search(
            r"<script type=\"module\">([\s\S]*?)</script>", html).group(1)
        assert content.strip(), name
        # every other script is a same-origin file
        for tag in tags:
            if re.search(r"\bsrc\s*=", tag):
                src = re.search(r"\bsrc\s*=\s*[\"']([^\"']+)[\"']", tag).group(1)
                assert src.startswith("/static/"), (name, src)
        # no inline <style>: the style-src has no 'unsafe-inline'
        assert "<style" not in html, name


# ---------------------------------------------------------------------------
# 8. doctor
# ---------------------------------------------------------------------------

def test_doctor_worker_reachable_but_unhealthy_fails(tmp_path, capsys):
    import urllib.request

    class FakeResponse:
        status = 200

        def read(self):
            return b'{"status": "degraded"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with _fake_disk_usage(10 * 1024 ** 3), \
            mock.patch.object(urllib.request, "urlopen",
                              return_value=FakeResponse()):
        with pytest.raises(SystemExit) as exc:
            M._cmd_doctor(M.build_parser().parse_args(
                ["doctor", "--root", str(tmp_path / "s"),
                 "--worker", "http://127.0.0.1:9111"]))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert re.search(r"worker reachable\s+FAIL", out)


def test_doctor_free_disk_boundary_at_five_gibibytes(tmp_path, capsys):
    # "free disk space above 5 GB": exactly the threshold is not above it
    with _fake_disk_usage(5 * 1024 ** 3):
        with pytest.raises(SystemExit) as exc:
            M._cmd_doctor(M.build_parser().parse_args(
                ["doctor", "--root", str(tmp_path / "s")]))
    assert exc.value.code == 1
    assert re.search(r"free disk space\s+FAIL", capsys.readouterr().out)

    with _fake_disk_usage(5 * 1024 ** 3 + 1):
        M._cmd_doctor(M.build_parser().parse_args(
            ["doctor", "--root", str(tmp_path / "s")]))
    assert "FAIL" not in capsys.readouterr().out


def test_doctor_token_file_boundaries(tmp_path, capsys):
    short = _token_file(tmp_path, token="x" * 31, name="short")
    with _fake_disk_usage(10 * 1024 ** 3):
        with pytest.raises(SystemExit) as exc:
            M._cmd_doctor(M.build_parser().parse_args(
                ["doctor", "--root", str(tmp_path / "s"),
                 "--token-file", str(short)]))
    assert exc.value.code == 1
    assert re.search(r"token file\s+FAIL", capsys.readouterr().out)

    missing = tmp_path / "missing"
    with _fake_disk_usage(10 * 1024 ** 3):
        with pytest.raises(SystemExit):
            M._cmd_doctor(M.build_parser().parse_args(
                ["doctor", "--root", str(tmp_path / "s"),
                 "--token-file", str(missing)]))
    capsys.readouterr()  # drop the failing run's table

    ok = _token_file(tmp_path, token="x" * 32, name="ok")
    with _fake_disk_usage(10 * 1024 ** 3):
        M._cmd_doctor(M.build_parser().parse_args(
            ["doctor", "--root", str(tmp_path / "s"),
             "--token-file", str(ok)]))
    out = capsys.readouterr().out
    assert re.search(r"token file\s+ok", out)
    assert "FAIL" not in out


def test_doctor_fails_when_fastapi_is_missing(tmp_path, capsys, monkeypatch):
    import importlib.util
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        return None if name == "fastapi" else real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    with _fake_disk_usage(10 * 1024 ** 3):
        with pytest.raises(SystemExit) as exc:
            M._cmd_doctor(M.build_parser().parse_args(
                ["doctor", "--root", str(tmp_path / "s")]))
    assert exc.value.code == 1
    assert re.search(r"fastapi installed\s+FAIL", capsys.readouterr().out)


def test_doctor_all_ok_with_worker_and_token(tmp_path, capsys):
    import urllib.request

    class FakeResponse:
        status = 200

        def read(self):
            return b'{"status": "ok"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    token_file = _token_file(tmp_path, token="x" * 40, name="tok")
    with _fake_disk_usage(10 * 1024 ** 3), \
            mock.patch.object(urllib.request, "urlopen",
                              return_value=FakeResponse()):
        M._cmd_doctor(M.build_parser().parse_args(
            ["doctor", "--root", str(tmp_path / "store"),
             "--worker", "http://127.0.0.1:9111",
             "--token-file", str(token_file)]))
    out = capsys.readouterr().out
    for name in ("python version", "fastapi installed", "writable root",
                 "free disk space", "worker reachable", "token file"):
        assert name in out
    assert "FAIL" not in out
