"""Video upload through the library app (task: video_client).

A fake worker client (same interface as ``sideline.videos.VideoWorkerClient``)
stands in for the worker service: no sockets, no real video. The whole path
video -> job -> match -> analytics is covered, plus every error case, the
restart persistence of the job-to-match mapping, and the real client's HTTP
behaviour (its ``http.client`` connection is faked, so no sockets either).
"""
import io
import json
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sideline import library  # noqa: E402
from sideline.videos import VideoWorkerClient, WorkerClientError  # noqa: E402

SAMPLE = REPO / "tests" / "fixtures" / "sample_match.jsonl"
STORE_LIMIT = 100 * 1024 * 1024
WORKER_LIMIT = 2 * 1024 ** 3

UNREACHABLE = ("the video processing machine is not reachable; "
               "check that the worker is running and try again")
EXTENSION_ERROR = ("file extension must be one of: .avi .mkv .mov .mp4 .webm; "
                   "choose a video file")


class FakeWorker:
    """A worker client with the same interface as videos.VideoWorkerClient."""

    max_bytes = WORKER_LIMIT

    def __init__(self):
        self.jobs = {}
        self.submissions = []
        self.deletes = []
        self.result_fetches = 0
        self.down = False
        self.submit_error = None
        self.result = SAMPLE.read_bytes()

    def submit(self, stream, filename, name=None):
        if self.submit_error is not None:
            raise self.submit_error
        if self.down:
            raise WorkerClientError("connection refused", status=None)
        data = b""
        while True:
            chunk = stream.read(1024 * 64)
            if not chunk:
                break
            data += chunk
        self.submissions.append({"filename": filename, "name": name, "data": data})
        job_id = "j_%012x" % (len(self.jobs) + 1)
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "frames": 0,
            "name": name or "video",
            "error": None,
            "created_at": "2026-01-01T00:00:00Z",
            "seconds": None,
        }
        return job_id

    def finish(self, job_id):
        job = self.jobs[job_id]
        job["status"] = "done"
        job["progress"] = 1.0
        job["frames"] = 750

    def status(self, job_id):
        if self.down:
            raise WorkerClientError("connection refused", status=None)
        job = self.jobs.get(job_id)
        if job is None:
            raise WorkerClientError("job not found", status=404)
        return dict(job)

    def fetch_result(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            raise WorkerClientError("job not found", status=404)
        if job["status"] != "done":
            raise WorkerClientError(
                "job is %s; the result is available once the job is done" % job["status"],
                status=409)
        self.result_fetches += 1
        data = self.result
        return iter([data[:1024], data[1024:]])

    def delete(self, job_id):
        if self.down:
            raise WorkerClientError("connection refused", status=None)
        if job_id not in self.jobs:
            raise WorkerClientError("job not found", status=404)
        del self.jobs[job_id]
        self.deletes.append(job_id)


@pytest.fixture()
def fake_worker():
    return FakeWorker()


def make_client(root, fake=None):
    return TestClient(library.create_library_app(root, worker_client=fake) if fake is not None
                      else library.create_library_app(root))


def post_video(client, filename="clip.mp4", data=b"fake video bytes", job_name=None):
    kwargs = {"files": {"file": (filename, data, "video/mp4")}}
    if job_name is not None:
        kwargs["data"] = {"name": job_name}
    return client.post("/api/videos", **kwargs)


def spool_under(monkeypatch, tmp_path):
    """Route the upload spool file (mkstemp without an explicit dir) into a
    test-owned folder so the tests can assert it is cleaned up."""
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    real_mkstemp = tempfile.mkstemp

    def mkstemp(prefix="", suffix="", dir=None, **kwargs):
        if dir is None:
            dir = str(spool_dir)
        return real_mkstemp(prefix=prefix, suffix=suffix, dir=dir, **kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", mkstemp)
    return spool_dir


# -- /api/limits ---------------------------------------------------------------


def test_limits_reports_the_video_configuration(tmp_path):
    body = make_client(tmp_path / "a").get("/api/limits").json()
    assert body == {"max_upload_bytes": STORE_LIMIT, "video_enabled": False,
                    "max_video_bytes": WORKER_LIMIT}
    body = make_client(tmp_path / "b", fake=FakeWorker()).get("/api/limits").json()
    assert body == {"max_upload_bytes": STORE_LIMIT, "video_enabled": True,
                    "max_video_bytes": WORKER_LIMIT}


# -- the whole path: video -> job -> match -> analytics -------------------------


def test_video_job_lifecycle_stores_the_match_as_a_normal_match(tmp_path, fake_worker):
    root = tmp_path / "m"
    client = make_client(root, fake=fake_worker)
    response = post_video(client, data=b"A" * 4096, job_name="Friday night")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert re.fullmatch(r"j_[0-9a-f]{12}", body["job_id"])
    job_id = body["job_id"]
    # The upload went to the worker with the client's file name and the name.
    assert fake_worker.submissions == [
        {"filename": "clip.mp4", "name": "Friday night", "data": b"A" * 4096}]

    # Queued and running statuses pass the worker through.
    status = client.get("/api/videos/" + job_id).json()
    assert status["status"] == "queued" and status["name"] == "Friday night"
    fake_worker.jobs[job_id]["status"] = "running"
    fake_worker.jobs[job_id]["progress"] = 0.42
    assert client.get("/api/videos/" + job_id).json()["progress"] == 0.42

    # done for the first time: the result is downloaded, stored through the
    # normal match path, the worker's copy is deleted and match_id is added.
    fake_worker.finish(job_id)
    body = client.get("/api/videos/" + job_id).json()
    assert body["status"] == "done"
    assert body["name"] == "Friday night"
    match_id = body["match_id"]
    assert re.fullmatch(r"m_[0-9a-f]{10}", match_id)
    assert fake_worker.result_fetches == 1
    assert fake_worker.deletes == [job_id]

    # The match is a normal stored match: listed, openable, analytics works.
    meta = client.get("/api/matches/%s" % match_id).json()
    assert meta["name"] == "Friday night" and meta["source"] == job_id
    assert meta["frame_count"] == 750
    assert [m["id"] for m in client.get("/api/matches").json()["items"]] == [match_id]
    assert client.get("/api/matches/%s/analytics" % match_id).json()["frame_count"] == 750

    # The mapping is remembered on disk in the store root.
    mapping = json.loads((root / "video_jobs.json").read_text(encoding="utf-8"))
    assert mapping == {job_id: {"match_id": match_id}}

    # Later calls return the same match_id without downloading again.
    again = client.get("/api/videos/" + job_id).json()
    assert again == {"job_id": job_id, "status": "done", "progress": 1.0,
                     "match_id": match_id}
    assert fake_worker.result_fetches == 1

    # Deleting the job forgets it (the worker's copy is already gone).
    assert client.delete("/api/videos/" + job_id).status_code == 204
    assert client.get("/api/videos/" + job_id).status_code == 404
    assert client.delete("/api/videos/" + job_id).status_code == 404
    assert not (root / "video_jobs.json").exists()
    # The match itself is untouched.
    assert client.get("/api/matches/%s" % match_id).status_code == 200


def test_two_racing_polls_download_the_result_exactly_once(tmp_path, fake_worker):
    client = make_client(tmp_path / "m", fake=fake_worker)
    job_id = post_video(client).json()["job_id"]
    fake_worker.finish(job_id)
    with ThreadPoolExecutor(2) as pool:
        bodies = list(pool.map(lambda _: client.get("/api/videos/" + job_id).json(),
                               range(2)))
    assert bodies[0] == bodies[1]
    assert bodies[0]["status"] == "done" and bodies[0]["match_id"]
    assert fake_worker.result_fetches == 1
    assert fake_worker.deletes.count(job_id) == 1


def test_job_to_match_mapping_survives_a_restart(tmp_path, fake_worker):
    root = tmp_path / "m"
    client = make_client(root, fake=fake_worker)
    job_id = post_video(client, job_name="Friday night").json()["job_id"]
    fake_worker.finish(job_id)
    match_id = client.get("/api/videos/" + job_id).json()["match_id"]
    assert fake_worker.deletes == [job_id]  # the worker's copy is already gone

    # A new app on the same store root, with a worker that has never seen
    # the job: the answer must come from the mapping in the store root.
    fresh = make_client(root, fake=FakeWorker())
    assert fresh.get("/api/videos/" + job_id).json() == {
        "job_id": job_id, "status": "done", "progress": 1.0, "match_id": match_id}

    # The mapping is served even without a worker at all.
    assert make_client(root).get("/api/videos/" + job_id).json()["match_id"] == match_id


# -- errors ---------------------------------------------------------------------


def test_post_video_without_a_worker_is_503_and_jobs_are_404(tmp_path):
    client = make_client(tmp_path / "m")
    response = post_video(client)
    assert response.status_code == 503
    assert response.json() == {"error": "video processing is not configured"}
    assert client.get("/api/videos/j_000000000001").status_code == 404
    assert client.delete("/api/videos/j_000000000001").status_code == 404


@pytest.mark.parametrize("filename", ["clip.txt", "clip.jsonl", "clip", "CLIP"])
def test_post_video_rejects_wrong_extensions(tmp_path, fake_worker, filename):
    root = tmp_path / "m"
    client = make_client(root, fake=fake_worker)
    response = post_video(client, filename=filename)
    assert response.status_code == 400
    assert response.json()["error"] == EXTENSION_ERROR
    assert fake_worker.submissions == []
    assert list(root.iterdir()) == []  # nothing staged, nothing stored


@pytest.mark.parametrize("filename", ["clip.MP4", "CLIP.MOV"])
def test_post_video_extensions_are_case_insensitive(tmp_path, fake_worker, filename):
    client = make_client(tmp_path / "m", fake=fake_worker)
    assert post_video(client, filename=filename).status_code == 202


def test_post_video_rejects_an_empty_file(tmp_path, fake_worker):
    root = tmp_path / "m"
    client = make_client(root, fake=fake_worker)
    response = post_video(client, data=b"")
    assert response.status_code == 400
    assert response.json()["error"] == "file is empty; choose a video file with content"
    assert fake_worker.submissions == []
    assert list(root.iterdir()) == []


def test_post_video_without_a_file_field_is_404_or_400_and_stores_nothing(tmp_path, fake_worker):
    client = make_client(tmp_path / "m", fake=fake_worker)
    response = client.post("/api/videos", data={"name": "no file"})
    assert response.status_code == 400
    assert fake_worker.submissions == []


def test_worker_down_at_submit_time_is_502_with_the_reachability_message(
        tmp_path, fake_worker, monkeypatch):
    spool = spool_under(monkeypatch, tmp_path)
    client = make_client(tmp_path / "m", fake=fake_worker)
    fake_worker.down = True
    response = post_video(client, data=b"B" * 2048)
    assert response.status_code == 502
    assert response.json() == {"error": UNREACHABLE}
    assert fake_worker.submissions == []
    assert list(spool.iterdir()) == []  # the spool file is cleaned up


def test_worker_down_while_polling_is_502(tmp_path, fake_worker):
    client = make_client(tmp_path / "m", fake=fake_worker)
    job_id = post_video(client).json()["job_id"]
    fake_worker.down = True
    response = client.get("/api/videos/" + job_id)
    assert response.status_code == 502
    assert response.json() == {"error": UNREACHABLE}


def test_worker_rejects_the_upload_and_the_error_passes_through(
        tmp_path, fake_worker, monkeypatch):
    spool = spool_under(monkeypatch, tmp_path)
    client = make_client(tmp_path / "m", fake=fake_worker)
    fake_worker.submit_error = WorkerClientError(
        "file exceeds the maximum size of 1048576 bytes; choose a smaller recording",
        status=413)
    response = post_video(client, data=b"0" * 2048)
    assert response.status_code == 413
    assert response.json()["error"] == (
        "file exceeds the maximum size of 1048576 bytes; choose a smaller recording")
    assert list(spool.iterdir()) == []  # the spool file is cleaned up


def test_worker_delete_failure_is_502(tmp_path, fake_worker):
    client = make_client(tmp_path / "m", fake=fake_worker)
    job_id = post_video(client).json()["job_id"]
    fake_worker.down = True
    response = client.delete("/api/videos/" + job_id)
    assert response.status_code == 502
    assert response.json() == {"error": UNREACHABLE}
    # The job is forgotten locally even though the worker's copy survived.
    fake_worker.down = False
    fake_worker.jobs[job_id] = {"job_id": job_id, "status": "queued", "progress": 0.0,
                                "frames": 0, "name": "video", "error": None,
                                "created_at": "2026-01-01T00:00:00Z", "seconds": None}
    assert client.get("/api/videos/" + job_id).json()["status"] == "queued"


# -- results that fail validation ------------------------------------------------


def test_invalid_result_fails_the_job_with_the_validation_message(tmp_path, fake_worker):
    root = tmp_path / "m"
    client = make_client(root, fake=fake_worker)
    job_id = post_video(client, job_name="bad").json()["job_id"]
    fake_worker.result = b"this is not json\n"
    fake_worker.finish(job_id)
    body = client.get("/api/videos/" + job_id).json()
    assert body == {"job_id": job_id, "status": "failed", "error": "line 1: invalid JSON"}
    assert fake_worker.deletes == [job_id]  # the worker's copy is deleted
    assert client.get("/api/matches").json()["items"] == []
    mapping = json.loads((root / "video_jobs.json").read_text(encoding="utf-8"))
    assert mapping == {job_id: {"error": "line 1: invalid JSON"}}
    # The failed outcome is served from the mapping even though the worker
    # has lost the job (its copy was deleted with the failed result).
    assert job_id not in fake_worker.jobs
    assert client.get("/api/videos/" + job_id).json() == body


def test_oversized_result_fails_the_job_with_the_store_limit_message(tmp_path, fake_worker):
    client = make_client(tmp_path / "m", fake=fake_worker)
    job_id = post_video(client).json()["job_id"]
    fake_worker.result = b"0" * (STORE_LIMIT + 16)
    fake_worker.finish(job_id)
    body = client.get("/api/videos/" + job_id).json()
    assert body["status"] == "failed"
    assert body["error"] == "file exceeds the maximum size of %d bytes" % STORE_LIMIT
    assert client.get("/api/matches").json()["items"] == []


# -- delete and malformed ids -----------------------------------------------------


def test_delete_cancels_the_job_and_forgets_it(tmp_path, fake_worker):
    root = tmp_path / "m"
    client = make_client(root, fake=fake_worker)
    job_id = post_video(client).json()["job_id"]
    assert client.delete("/api/videos/" + job_id).status_code == 204
    assert job_id in fake_worker.deletes
    assert client.get("/api/videos/" + job_id).status_code == 404
    assert client.delete("/api/videos/" + job_id).status_code == 404
    assert list(root.iterdir()) == []  # no mapping file left behind


@pytest.mark.parametrize("job_id", ["j_short", "j_00000000000G", "J_000000000001", "x1"])
def test_malformed_job_ids_are_404_without_touching_the_worker(tmp_path, fake_worker, job_id):
    client = make_client(tmp_path / "m", fake=fake_worker)
    assert client.get("/api/videos/" + job_id).status_code == 404
    assert client.delete("/api/videos/" + job_id).status_code == 404
    assert fake_worker.submissions == []


def test_stale_mapping_for_a_deleted_match_is_dropped(tmp_path, fake_worker):
    root = tmp_path / "m"
    client = make_client(root, fake=fake_worker)
    job_id = post_video(client).json()["job_id"]
    fake_worker.finish(job_id)
    match_id = client.get("/api/videos/" + job_id).json()["match_id"]
    assert client.delete("/api/matches/" + match_id).status_code == 204
    # The mapping points at a match that no longer exists: the job is
    # unknown again (the worker's copy is gone too).
    assert client.get("/api/videos/" + job_id).status_code == 404
    assert not (root / "video_jobs.json").exists()


# -- the command line ---------------------------------------------------------------


def test_library_command_has_the_worker_flag():
    from sideline import __main__ as cli
    parser = cli.build_parser()
    args = parser.parse_args(["library", "--root", "r", "--worker", "http://127.0.0.1:8200"])
    assert args.worker == "http://127.0.0.1:8200"
    args = parser.parse_args(["library", "--root", "r"])
    assert args.worker is None


def test_blank_worker_url_leaves_video_disabled(tmp_path):
    client = TestClient(library.create_library_app(tmp_path / "m", worker_url="   "))
    assert client.get("/api/limits").json()["video_enabled"] is False
    assert post_video(client).status_code == 503


def test_worker_url_without_a_scheme_defaults_to_http(tmp_path):
    client = TestClient(library.create_library_app(tmp_path / "m", worker_url="127.0.0.1:8200"))
    assert client.get("/api/limits").json()["video_enabled"] is True


def test_library_command_rejects_a_bad_worker_url(monkeypatch, tmp_path):
    from unittest import mock
    from sideline import __main__ as cli
    monkeypatch.setitem(sys.modules, "uvicorn", mock.MagicMock(name="uvicorn"))
    monkeypatch.setattr(sys, "argv",
                        ["sideline", "library", "--root", str(tmp_path),
                         "--worker", "http://:8200"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "[sideline] error:" in str(exc.value.code)
    assert "worker URL" in str(exc.value.code)


# -- the real worker client (faked connection, no sockets) ------------------------


class _FakeHTTPResponse:
    def __init__(self, status, body=b""):
        self.status = status
        self._body = body
        self._pos = 0

    def read(self, n=-1):
        if n is None or n < 0:
            data = self._body[self._pos:]
            self._pos = len(self._body)
            return data
        data = self._body[self._pos:self._pos + n]
        self._pos += len(data)
        return data


class _FakeConnection:
    def __init__(self, response):
        self.response = response
        self.sent = []
        self.headers = {}
        self.method = None
        self.path = None
        self.chunked = False
        self.closed = False

    def putrequest(self, method, path):
        self.method, self.path = method, path

    def putheader(self, key, value):
        self.headers[key] = value

    def endheaders(self):
        self.chunked = any(k.lower() == "transfer-encoding" and str(v).lower() == "chunked" for k, v in self.headers.items())

    def send(self, data):
        self.sent.append(bytes(data))

    def request(self, method, path):
        self.method, self.path = method, path

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


def _with_fake_connection(monkeypatch, response=None):
    conn = _FakeConnection(response or _FakeHTTPResponse(200))
    monkeypatch.setattr("http.client.HTTPConnection", lambda *args, **kwargs: conn)
    return conn


def _with_refused_connection(monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionRefusedError(61, "Connection refused")

    monkeypatch.setattr("http.client.HTTPConnection", boom)


def _decode_chunked(data):
    out = []
    pos = 0
    while data[pos:pos + 1] != b"0":
        end = data.index(b"\r\n", pos)
        size = int(data[pos:end], 16)
        pos = end + 2
        out.append(data[pos:pos + size])
        pos += size + 2
    return b"".join(out)


def test_client_submit_streams_multipart_chunked_and_returns_the_job_id(monkeypatch):
    conn = _with_fake_connection(
        monkeypatch, _FakeHTTPResponse(202, b'{"job_id": "j_1234567890ab", "status": "queued"}'))
    client = VideoWorkerClient("http://10.0.0.5:8200")
    payload = b"0123456789" * 300_000  # 3 MB: several chunks of 1 MB
    assert client.submit(io.BytesIO(payload), "clip.mp4", "Friday night") == "j_1234567890ab"

    assert conn.method == "POST" and conn.path == "/jobs"
    assert conn.chunked is True and conn.closed
    boundary = conn.headers["Content-Type"].split("boundary=")[1]
    raw = _decode_chunked(b"".join(conn.sent))
    expected = (
        ("--%s\r\n"
         'Content-Disposition: form-data; name="file"; filename="clip.mp4"\r\n'
         "Content-Type: application/octet-stream\r\n\r\n" % boundary).encode("utf-8")
        + payload
        + ("\r\n--%s\r\n"
           'Content-Disposition: form-data; name="name"\r\n\r\n'
           "Friday night\r\n"
           "\r\n--%s--\r\n" % (boundary, boundary)).encode("utf-8")
    )
    assert raw == expected
    # The raw video appears exactly once in the (decoded) body - it is
    # streamed, chunked, and never buffered whole in the client.
    assert raw.count(payload) == 1


def test_client_submit_without_a_name_sends_only_the_file_field(monkeypatch):
    conn = _with_fake_connection(
        monkeypatch, _FakeHTTPResponse(202, b'{"job_id": "j_1234567890ab"}'))
    client = VideoWorkerClient("http://10.0.0.5:8200")
    client.submit(io.BytesIO(b"abcd"), "clip.mp4")
    raw = _decode_chunked(b"".join(conn.sent))
    assert b'name="name"' not in raw
    assert b"abcd" in raw


def test_client_status_parses_the_answer_and_maps_errors(monkeypatch):
    conn = _with_fake_connection(
        monkeypatch,
        _FakeHTTPResponse(200, b'{"job_id": "j_1", "status": "running", "progress": 0.5}'))
    client = VideoWorkerClient("http://10.0.0.5:8200")
    assert client.status("j_1") == {"job_id": "j_1", "status": "running", "progress": 0.5}
    conn.response = _FakeHTTPResponse(404, b'{"error": "job not found"}')
    with pytest.raises(WorkerClientError) as exc:
        client.status("j_1")
    assert exc.value.status == 404 and exc.value.message == "job not found"


def test_client_fetch_result_streams_the_body(monkeypatch):
    body = (b"0123456789abcdef" * 100000) + b"END"  # ~1.6 MB: several read chunks
    conn = _with_fake_connection(monkeypatch, _FakeHTTPResponse(200, body))
    client = VideoWorkerClient("http://10.0.0.5:8200")
    chunks = list(client.fetch_result("j_1"))
    assert len(chunks) > 1
    assert b"".join(chunks) == body


def test_client_delete_returns_none_on_204_and_raises_on_404(monkeypatch):
    conn = _with_fake_connection(monkeypatch, _FakeHTTPResponse(204))
    client = VideoWorkerClient("http://10.0.0.5:8200")
    assert client.delete("j_1") is None
    conn.response = _FakeHTTPResponse(404, b'{"error": "job not found"}')
    with pytest.raises(WorkerClientError) as exc:
        client.delete("j_1")
    assert exc.value.status == 404


def test_client_unreachable_maps_to_status_none(monkeypatch):
    _with_refused_connection(monkeypatch)
    client = VideoWorkerClient("http://10.0.0.5:8200")
    for call in (
        lambda: client.submit(io.BytesIO(b"x"), "clip.mp4"),
        lambda: client.status("j_1"),
        lambda: client.fetch_result("j_1"),
        lambda: client.delete("j_1"),
    ):
        with pytest.raises(WorkerClientError) as exc:
            call()
        assert exc.value.status is None
        assert "could not reach the video processing machine" in exc.value.message


def test_client_multipart_body_is_accepted_by_the_worker_app(tmp_path, monkeypatch):
    """Protocol check: the exact body the client builds (decoded from its
    chunked framing) is parsed by the real worker app (a fake runner, no
    sockets, no real video)."""
    from sideline.worker import create_worker_app

    seen = {}

    def runner(video, out, on_progress):
        seen["video"] = Path(video).read_bytes()
        out.write_text('{"frame": 0, "people": []}\n', encoding="utf-8")
        on_progress(1)

    worker_client = TestClient(create_worker_app(tmp_path / "worker", runner=runner))

    # The client's own answer (job id) is irrelevant here: only the body it
    # sends is under test.
    conn = _with_fake_connection(monkeypatch, _FakeHTTPResponse(202, b'{"job_id": "j_000000000001"}'))
    payload = b"V" * 5000
    VideoWorkerClient("http://10.0.0.5:8200").submit(io.BytesIO(payload), "clip.mp4", "Friday night")
    raw = _decode_chunked(b"".join(conn.sent))

    response = worker_client.post(
        "/jobs", content=raw, headers={"content-type": conn.headers["Content-Type"]})
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = worker_client.get("/jobs/" + job_id).json()
        if status["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "done", status
    assert status["name"] == "Friday night"
    assert seen["video"] == payload  # the video bytes survived the round trip
