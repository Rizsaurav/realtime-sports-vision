"""Extra adversarial tests for the video upload client side
(tasks/video_client.md: src/sideline/videos.py + src/sideline/library.py).

The acceptance suite (tests_integration/test_video_upload.py) covers the
happy path and the main error cases.  These tests attack the spec rules it
does not check, still without sockets (the worker client is a fake object;
the real ``videos.VideoWorkerClient`` gets a faked ``http.client``
connection, as the acceptance suite does):

- POST /api/videos: all five worker extensions accepted (the same five as
  the worker, case-insensitive), multi-dot and trailing-dot file names, a
  large (multi-megabyte) upload streamed byte-exact to the worker with the
  spool file cleaned up, the optional name forwarded (unicode), repeated
  uploads getting distinct jobs, and a worker error answer (500) passing
  through with its status and message;
- GET /api/videos/{job_id}: the done-for-the-first-time answer is exactly
  the worker's status dict plus ``match_id``; a done status without a name
  still stores the match; validation failures of the *normal* match path
  (repeated frame number, a result that is not UTF-8, an empty result)
  turn the job into ``failed`` with that message and are remembered
  without re-downloading; the store-size boundary (a result of exactly
  max_bytes is stored, one byte over is 413); a transient worker failure
  (status poll 502, download 404/502) keeps the job resolvable;
- the on-disk mapping after a restart: a job still running on the worker
  is not yet mapped, a restart serves both done and failed mappings, and
  DELETE forgets a mapped job even when no worker is configured;
- ordering/concurrency: two jobs resolve to their own matches, and a
  poll racing a delete never answers 5xx and always leaves a consistent
  final state (download at most once);
- /api/limits with an injected client carrying its own limit, and the real
  client built from ``worker_url``;
- the CLI: ``library --worker URL`` builds an app with video enabled;
- the real client (faked connection): it never reads the whole video into
  memory (chunked reads from a stream that only yields small pieces), a
  unicode name survives the multipart body, a base path in the worker URL
  is honoured (and https uses HTTPSConnection), unusable worker URLs are
  rejected with ValueError, and malformed worker answers (non-JSON error
  body, non-dict or status-less status answer, missing job id) raise
  WorkerClientError instead of being accepted.

Every assertion is required by tasks/video_client.md (and the worker
contract it references); no sockets and no real video are used, so the
suite is fast and deterministic.
"""

import io
import json
import re
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sideline import library, matches  # noqa: E402
from sideline import __main__ as cli  # noqa: E402
from sideline.videos import VideoWorkerClient, WorkerClientError, WORKER_EXTENSIONS  # noqa: E402

UNREACHABLE = ("the video processing machine is not reachable; "
               "check that the worker is running and try again")
EXTENSION_ERROR = ("file extension must be one of: %s; choose a video file"
                   % " ".join(WORKER_EXTENSIONS))
WORKER_LIMIT = 2 * 1024 ** 3
SMALL_LIMIT = 4096

VALID_3 = (b'{"frame": 0, "people": []}\n'
           b'{"frame": 1, "people": []}\n'
           b'{"frame": 2, "people": []}\n')
VALID_5 = VALID_3 + (b'{"frame": 3, "people": []}\n'
                     b'{"frame": 4, "people": []}\n')


# -- fakes -------------------------------------------------------------------


class FakeWorker:
    """A worker client with the same interface as videos.VideoWorkerClient,
    with hooks for transient failures and a configurable upload limit."""

    def __init__(self, max_bytes=WORKER_LIMIT):
        self.max_bytes = max_bytes
        self.jobs = {}
        self.submissions = []
        self.deletes = []
        self.result_fetches = 0
        self.submit_error = None
        self.status_error = None
        self.result_error = None
        self.result = VALID_3
        self._counter = 0

    def submit(self, stream, filename, name=None):
        if self.submit_error is not None:
            raise self.submit_error
        data = b""
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            data += chunk
        self._counter += 1
        job_id = "j_%012x" % self._counter
        self.submissions.append({"filename": filename, "name": name, "data": data})
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

    def set(self, job_id, **fields):
        self.jobs[job_id].update(fields)

    def finish(self, job_id, frames=None):
        job = self.jobs[job_id]
        job["status"] = "done"
        job["progress"] = 1.0
        if frames is not None:
            job["frames"] = frames

    def status(self, job_id):
        if self.status_error is not None:
            raise self.status_error
        job = self.jobs.get(job_id)
        if job is None:
            raise WorkerClientError("job not found", status=404)
        return dict(job)

    def fetch_result(self, job_id):
        if self.result_error is not None:
            raise self.result_error
        job = self.jobs.get(job_id)
        if job is None:
            raise WorkerClientError("job not found", status=404)
        if job["status"] != "done":
            raise WorkerClientError(
                "job is %s; the result is available once the job is done" % job["status"],
                status=409)
        self.result_fetches += 1
        data = self.result
        half = len(data) // 2
        return iter([data[:half], data[half:]])

    def delete(self, job_id):
        if job_id not in self.jobs:
            self.deletes.append(job_id)
            raise WorkerClientError("job not found", status=404)
        del self.jobs[job_id]
        self.deletes.append(job_id)


def make_app(root, fake):
    return TestClient(library.create_library_app(root, worker_client=fake))


def post_video(client, filename="clip.mp4", data=b"fake video bytes", name=None):
    kwargs = {"files": {"file": (filename, data, "video/mp4")}}
    if name is not None:
        kwargs["data"] = {"name": name}
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


def valid_jsonl_of(total):
    """A valid recording of exactly ``total`` bytes (the last line is
    padded with an ignored ``pad`` field)."""
    def plain(j):
        return json.dumps({"frame": j, "people": []}) + "\n"

    def base(j):
        return json.dumps({"frame": j, "people": [], "pad": ""}) + "\n"

    lines, offset, i = [], 0, 0
    while offset + len(plain(i + 1)) + len(base(i + 1)) <= total:
        lines.append(plain(i))
        offset += len(plain(i))
        i += 1
    pad = total - offset - len(base(i))
    assert pad >= 0, "cannot pad to %d bytes" % total
    lines.append(json.dumps({"frame": i, "people": [], "pad": "x" * pad}) + "\n")
    data = "".join(lines).encode("utf-8")
    assert len(data) == total
    return data


def small_store(monkeypatch):
    """The store of a freshly built app with a 4 KiB upload limit, so the
    size boundary can be probed without a 100 MiB result."""
    original = matches.MatchStore

    class SmallStore(original):
        def __init__(self, root, smooth=False):
            super().__init__(root, max_bytes=SMALL_LIMIT, smooth=smooth)

    monkeypatch.setattr(matches, "MatchStore", SmallStore)


# -- POST /api/videos ----------------------------------------------------------


@pytest.mark.parametrize("ext", sorted(WORKER_EXTENSIONS))
def test_all_five_worker_extensions_are_accepted(tmp_path, ext):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    response = post_video(client, filename="clip" + ext, data=b"V" * 512)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert re.fullmatch(r"j_[0-9a-f]{12}", body["job_id"])
    assert fake.submissions == [
        {"filename": "clip" + ext, "name": None, "data": b"V" * 512}]


def test_filename_edge_cases(tmp_path):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    # The extension is the last suffix, as on the worker.
    assert post_video(client, filename="my.clip.mp4").status_code == 202
    # No extension at all: a trailing dot is rejected with the extension
    # message, an empty file name is rejected with a 400 as well.
    response = post_video(client, filename="clip.")
    assert response.status_code == 400
    assert response.json()["error"] == EXTENSION_ERROR
    response = post_video(client, filename="")
    assert response.status_code == 400
    assert "error" in response.json()
    assert len(fake.submissions) == 1  # only the .mp4 file reached the worker


def test_large_upload_is_streamed_whole_to_the_worker(tmp_path, monkeypatch):
    spool = spool_under(monkeypatch, tmp_path)
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    data = b"0123456789abcdef" * 162500  # 2.6 MiB, larger than one chunk
    response = post_video(client, data=data)
    assert response.status_code == 202
    assert fake.submissions == [
        {"filename": "clip.mp4", "name": None, "data": data}]
    assert list(spool.iterdir()) == []  # the spool file is cleaned up


def test_the_name_field_is_forwarded_to_the_worker(tmp_path):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    assert post_video(client, name="F\u00fcssball \u2013 Nacht").status_code == 202
    assert fake.submissions[0]["name"] == "F\u00fcssball \u2013 Nacht"
    # No name field at all: the worker gets None (its own fallback applies).
    assert post_video(client).status_code == 202
    assert fake.submissions[1]["name"] is None


def test_repeated_uploads_get_distinct_queued_jobs(tmp_path):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    first = post_video(client, data=b"one").json()
    second = post_video(client, data=b"two").json()
    assert first["job_id"] != second["job_id"]
    for body in (first, second):
        assert body["status"] == "queued"
    assert [s["data"] for s in fake.submissions] == [b"one", b"two"]
    # Each job is independently visible with its own id.
    assert client.get("/api/videos/" + first["job_id"]).json()["job_id"] == first["job_id"]
    assert client.get("/api/videos/" + second["job_id"]).json()["job_id"] == second["job_id"]


def test_worker_submit_error_passes_through_status_and_message(tmp_path, monkeypatch):
    spool = spool_under(monkeypatch, tmp_path)
    fake = FakeWorker()
    fake.submit_error = WorkerClientError(
        "the processing machine crashed", status=500)
    client = make_app(tmp_path / "m", fake)
    response = post_video(client, data=b"z" * 1024)
    assert response.status_code == 500
    assert response.json() == {"error": "the processing machine crashed"}
    assert fake.submissions == []
    assert list(spool.iterdir()) == []  # the spool file is cleaned up


# -- GET /api/videos/{job_id}: the done answer ---------------------------------


def test_done_answer_is_the_workers_status_plus_match_id(tmp_path):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    job_id = post_video(client, name="Friday night").json()["job_id"]
    fake.finish(job_id, frames=7)
    fake.jobs[job_id]["seconds"] = 0.28
    expected = dict(fake.jobs[job_id])  # the worker's status at that moment
    body = client.get("/api/videos/" + job_id).json()
    expected["match_id"] = body["match_id"]
    assert body == expected  # the worker's own status, plus match_id
    assert re.fullmatch(r"m_[0-9a-f]{10}", body["match_id"])


def test_done_without_a_name_still_stores_the_match(tmp_path):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    job_id = post_video(client).json()["job_id"]
    del fake.jobs[job_id]["name"]  # the worker status carries no name
    fake.finish(job_id)
    body = client.get("/api/videos/" + job_id).json()
    assert body["status"] == "done" and body["match_id"]
    assert body.get("name") is None  # nothing is invented in the answer
    meta = client.get("/api/matches/%s" % body["match_id"]).json()
    assert isinstance(meta["name"], str) and meta["name"]
    assert meta["source"] == job_id


def test_two_jobs_resolve_to_their_own_matches(tmp_path):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    job_a = post_video(client, name="Match A").json()["job_id"]
    job_b = post_video(client, name="Match B").json()["job_id"]
    fake.finish(job_a, frames=3)
    fake.finish(job_b, frames=5)
    # Each job downloads its own result: set it right before the poll.
    fake.result = VALID_3
    body_a = client.get("/api/videos/" + job_a).json()
    fake.result = VALID_5
    body_b = client.get("/api/videos/" + job_b).json()
    assert body_a["match_id"] != body_b["match_id"]
    assert client.get("/api/matches/%s" % body_a["match_id"]).json()["name"] == "Match A"
    assert client.get("/api/matches/%s" % body_b["match_id"]).json()["name"] == "Match B"
    assert fake.result_fetches == 2
    assert sorted(fake.deletes) == sorted([job_a, job_b])
    root = tmp_path / "m"
    mapping = json.loads((root / "video_jobs.json").read_text(encoding="utf-8"))
    assert mapping == {job_a: {"match_id": body_a["match_id"]},
                       job_b: {"match_id": body_b["match_id"]}}
    ids = [m["id"] for m in client.get("/api/matches").json()["items"]]
    assert sorted(ids) == sorted([body_a["match_id"], body_b["match_id"]])
    # Repeated calls are served from the mapping, without re-downloading.
    assert client.get("/api/videos/" + job_a).json()["match_id"] == body_a["match_id"]
    assert client.get("/api/videos/" + job_b).json()["match_id"] == body_b["match_id"]
    assert fake.result_fetches == 2


# -- results that fail the normal match validation -----------------------------


def test_failed_result_is_remembered_without_re_downloading(tmp_path):
    root = tmp_path / "m"
    fake = FakeWorker()
    client = make_app(root, fake)
    job_id = post_video(client).json()["job_id"]
    fake.result = b""  # a result with no frames at all
    fake.finish(job_id)
    body = client.get("/api/videos/" + job_id).json()
    assert body == {"job_id": job_id, "status": "failed",
                    "error": "file contains no frames"}
    assert fake.deletes == [job_id]  # the worker's copy is deleted
    assert client.get("/api/matches").json()["items"] == []
    mapping = json.loads((root / "video_jobs.json").read_text(encoding="utf-8"))
    assert mapping == {job_id: {"error": "file contains no frames"}}
    # Later calls come from the mapping: same error, no new download.
    again = client.get("/api/videos/" + job_id).json()
    assert again == body
    assert fake.result_fetches == 1


def test_result_with_a_repeated_frame_fails_with_the_validation_message(tmp_path):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    job_id = post_video(client).json()["job_id"]
    fake.result = (b'{"frame": 0, "people": []}\n'
                   b'{"frame": 0, "people": []}\n')
    fake.finish(job_id)
    body = client.get("/api/videos/" + job_id).json()
    assert body["status"] == "failed"
    assert body["error"] == "line 2: frame must be strictly increasing"
    assert client.get("/api/matches").json()["items"] == []


def test_result_that_is_not_utf8_fails_the_job(tmp_path):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    job_id = post_video(client).json()["job_id"]
    fake.result = b"\xff\xfe{\"frame\": 0, \"people\": []}\n"
    fake.finish(job_id)
    body = client.get("/api/videos/" + job_id).json()
    assert body["status"] == "failed"
    assert body["error"] == "file is not valid UTF-8"


def test_result_exactly_at_the_store_limit_is_stored(tmp_path, monkeypatch):
    small_store(monkeypatch)
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    assert client.get("/api/limits").json()["max_upload_bytes"] == SMALL_LIMIT
    job_id = post_video(client).json()["job_id"]
    fake.result = valid_jsonl_of(SMALL_LIMIT)
    fake.finish(job_id)
    body = client.get("/api/videos/" + job_id).json()
    assert body["status"] == "done" and body["match_id"]
    meta = client.get("/api/matches/%s" % body["match_id"]).json()
    assert meta["size_bytes"] == SMALL_LIMIT


def test_result_one_byte_over_the_store_limit_fails(tmp_path, monkeypatch):
    small_store(monkeypatch)
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    job_id = post_video(client).json()["job_id"]
    fake.result = b"y" * (SMALL_LIMIT + 1)
    fake.finish(job_id)
    body = client.get("/api/videos/" + job_id).json()
    assert body["status"] == "failed"
    assert body["error"] == "file exceeds the maximum size of %d bytes" % SMALL_LIMIT
    assert client.get("/api/matches").json()["items"] == []


# -- transient worker failures keep the job resolvable --------------------------


def test_worker_down_while_polling_is_502_and_then_recovers(tmp_path):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    job_id = post_video(client).json()["job_id"]
    fake.finish(job_id)
    fake.status_error = WorkerClientError("connection refused", status=None)
    response = client.get("/api/videos/" + job_id)
    assert response.status_code == 502
    assert response.json() == {"error": UNREACHABLE}
    assert not (tmp_path / "m" / "video_jobs.json").exists()  # not resolved yet
    fake.status_error = None
    body = client.get("/api/videos/" + job_id).json()
    assert body["status"] == "done" and body["match_id"]
    assert fake.result_fetches == 1


def test_worker_down_while_downloading_is_502_and_then_recovers(tmp_path):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    job_id = post_video(client).json()["job_id"]
    fake.finish(job_id)
    fake.result_error = WorkerClientError("connection reset", status=None)
    response = client.get("/api/videos/" + job_id)
    assert response.status_code == 502
    assert response.json() == {"error": UNREACHABLE}
    assert not (tmp_path / "m" / "video_jobs.json").exists()
    fake.result_error = None
    body = client.get("/api/videos/" + job_id).json()
    assert body["status"] == "done" and body["match_id"]
    assert fake.result_fetches == 1


def test_download_404_keeps_the_job_resolvable(tmp_path):
    root = tmp_path / "m"
    fake = FakeWorker()
    client = make_app(root, fake)
    job_id = post_video(client).json()["job_id"]
    fake.finish(job_id)
    fake.result_error = WorkerClientError("job not found", status=404)
    response = client.get("/api/videos/" + job_id)
    assert response.status_code == 404
    assert response.json() == {"error": "job not found"}
    assert not (root / "video_jobs.json").exists()  # nothing was stored
    assert fake.deletes == []  # the worker's copy was not deleted
    assert job_id in fake.jobs  # the job is still there on the worker
    fake.result_error = None
    body = client.get("/api/videos/" + job_id).json()
    assert body["status"] == "done" and body["match_id"]
    assert fake.result_fetches == 1


# -- restarts and the on-disk mapping ------------------------------------------


def test_restart_with_the_job_still_running_on_the_worker(tmp_path):
    root = tmp_path / "m"
    fake = FakeWorker()
    client = make_app(root, fake)
    job_id = post_video(client, name="Friday night").json()["job_id"]
    fake.set(job_id, status="running", progress=0.5)
    assert client.get("/api/videos/" + job_id).json()["status"] == "running"
    assert not (root / "video_jobs.json").exists()  # not done: nothing to remember

    # A restart: a fresh app (same store root) whose worker still has the
    # job running. The status comes from the worker, not from a mapping.
    fresh = FakeWorker()
    fresh.jobs[job_id] = dict(fake.jobs[job_id])
    client2 = make_app(root, fresh)
    assert client2.get("/api/videos/" + job_id).json() == dict(fresh.jobs[job_id])
    assert not (root / "video_jobs.json").exists()

    fresh.finish(job_id, frames=3)
    body = client2.get("/api/videos/" + job_id).json()
    assert body["status"] == "done" and body["match_id"]
    mapping = json.loads((root / "video_jobs.json").read_text(encoding="utf-8"))
    assert mapping == {job_id: {"match_id": body["match_id"]}}
    # The match is a normal match of this store.
    assert client2.get("/api/matches/%s" % body["match_id"]).json()["name"] == "Friday night"


def test_restart_serves_both_done_and_failed_mappings(tmp_path):
    root = tmp_path / "m"
    fake = FakeWorker()
    client = make_app(root, fake)
    job_done = post_video(client).json()["job_id"]
    job_bad = post_video(client).json()["job_id"]
    fake.result = VALID_3
    fake.finish(job_done)
    match_id = client.get("/api/videos/" + job_done).json()["match_id"]
    fake.result = b"this is not json\n"
    fake.finish(job_bad)
    error = client.get("/api/videos/" + job_bad).json()["error"]

    # A restart with a worker that has never seen either job.
    client2 = make_app(root, FakeWorker())
    assert client2.get("/api/videos/" + job_done).json()["match_id"] == match_id
    assert client2.get("/api/videos/" + job_bad).json() == {
        "job_id": job_bad, "status": "failed", "error": error}


def test_delete_forgets_a_mapped_job_without_a_worker(tmp_path):
    root = tmp_path / "m"
    fake = FakeWorker()
    client = make_app(root, fake)
    job_id = post_video(client).json()["job_id"]
    fake.finish(job_id)
    match_id = client.get("/api/videos/" + job_id).json()["match_id"]

    # A restart without any worker at all: the mapping is still served, and
    # DELETE still forgets the job locally.
    bare = TestClient(library.create_library_app(root))
    assert bare.get("/api/videos/" + job_id).json()["match_id"] == match_id
    assert bare.delete("/api/videos/" + job_id).status_code == 204
    assert not (root / "video_jobs.json").exists()
    assert bare.get("/api/videos/" + job_id).status_code == 404
    # The stored match is untouched.
    assert bare.get("/api/matches/%s" % match_id).status_code == 200


# -- concurrency ----------------------------------------------------------------


def test_polling_and_deleting_race_stays_consistent(tmp_path):
    fake = FakeWorker()
    client = make_app(tmp_path / "m", fake)
    job_id = post_video(client).json()["job_id"]
    fake.finish(job_id)

    results = []
    lock = threading.Lock()

    def record(fn):
        response = fn()
        with lock:
            results.append(response.status_code)

    with ThreadPoolExecutor(6) as pool:
        futures = []
        for _ in range(3):
            futures.append(pool.submit(record,
                                       lambda: client.get("/api/videos/" + job_id)))
            futures.append(pool.submit(record,
                                       lambda: client.delete("/api/videos/" + job_id)))
        for future in futures:
            future.result()

    assert all(code < 500 for code in results), results
    assert fake.result_fetches <= 1  # the download happened at most once
    final = client.get("/api/videos/" + job_id)
    assert final.status_code in (200, 404)
    root = tmp_path / "m"
    path = root / "video_jobs.json"
    mapping = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if final.status_code == 200:
        body = final.json()
        assert body["status"] == "done" and body["match_id"]
        assert mapping.get(job_id) == {"match_id": body["match_id"]}
        assert client.get("/api/matches/%s" % body["match_id"]).status_code == 200
    else:
        assert job_id not in mapping


# -- /api/limits and the worker URL ---------------------------------------------


def test_limits_reports_the_injected_clients_limit(tmp_path):
    fake = FakeWorker(max_bytes=3 * 1024 ** 3)
    body = make_app(tmp_path / "m", fake).get("/api/limits").json()
    assert body["video_enabled"] is True
    assert body["max_video_bytes"] == 3 * 1024 ** 3


def test_worker_url_builds_the_real_client(tmp_path):
    client = TestClient(library.create_library_app(
        tmp_path / "m", worker_url="http://127.0.0.1:9999"))
    body = client.get("/api/limits").json()
    assert body["video_enabled"] is True
    assert body["max_video_bytes"] == WORKER_LIMIT


def test_cli_worker_flag_switches_on_video_uploads(monkeypatch, tmp_path):
    uvicorn = mock.MagicMock(name="uvicorn")
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setattr(sys, "argv", ["sideline", "library",
                                      "--root", str(tmp_path / "store"),
                                      "--worker", "http://127.0.0.1:9"])
    cli.main()
    app = uvicorn.run.call_args.args[0]
    with TestClient(app) as built:
        body = built.get("/api/limits").json()
    assert body["video_enabled"] is True
    assert body["max_video_bytes"] == WORKER_LIMIT


# -- the real worker client (faked connection, no sockets) -----------------------


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

    def endheaders(self, chunked=False):
        self.chunked = chunked

    def send(self, data):
        self.sent.append(bytes(data))

    def request(self, method, path):
        self.method, self.path = method, path

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


def _with_fake_connection(monkeypatch, response=None, kind="HTTPConnection"):
    conn = _FakeConnection(response or _FakeHTTPResponse(200))
    monkeypatch.setattr("http.client.%s" % kind, lambda *args, **kwargs: conn)
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


class _TrickleStream:
    """A file-like stream that only yields small pieces and records how much
    the client ever asked for in one read: a read of the whole video (None
    or the full size) is exactly what the specification forbids."""

    def __init__(self, data, step=64 * 1024):
        self._data = data
        self._step = step
        self._pos = 0
        self.max_requested = None
        self.reads = 0

    def read(self, size=-1):
        self.reads += 1
        if size is None or size < 0:
            raise AssertionError("the client asked to read the whole video at once")
        if self.max_requested is None or size > self.max_requested:
            self.max_requested = size
        end = min(self._pos + min(size, self._step), len(self._data))
        chunk = self._data[self._pos:end]
        self._pos = end
        return chunk


def test_client_submit_never_reads_the_whole_video_into_memory(monkeypatch):
    conn = _with_fake_connection(
        monkeypatch, _FakeHTTPResponse(202, b'{"job_id": "j_000000000001"}'))
    payload = b"0123456789abcdef" * 156250  # 2.5 MiB
    stream = _TrickleStream(payload)
    job_id = VideoWorkerClient("http://10.0.0.5:8200").submit(
        stream, "clip.mp4", "Big clip")
    assert job_id == "j_000000000001"
    # The video was read in several bounded reads, never the whole file at
    # once (the stream is 2.5 MiB; no single read may ask for all of it).
    assert stream.reads > 1
    assert stream.max_requested is not None
    assert stream.max_requested < len(payload)
    # And the complete video arrived, exactly once, in the chunked body.
    raw = _decode_chunked(b"".join(conn.sent))
    assert raw.count(payload) == 1
    assert b"Big clip" in raw


def test_client_submit_unicode_name_survives_the_multipart_body(monkeypatch):
    conn = _with_fake_connection(
        monkeypatch, _FakeHTTPResponse(202, b'{"job_id": "j_000000000001"}'))
    name = "F\u00fcssball \u2013 \u591c"
    VideoWorkerClient("http://10.0.0.5:8200").submit(io.BytesIO(b"abcd"), "clip.mp4", name)
    raw = _decode_chunked(b"".join(conn.sent)).decode("utf-8")
    assert ("name=\"name\"" in raw) and (name in raw)


def test_client_honours_a_base_path_and_the_scheme(monkeypatch):
    conn = _with_fake_connection(
        monkeypatch, _FakeHTTPResponse(202, b'{"job_id": "j_000000000001"}'))
    VideoWorkerClient("http://10.0.0.5:8200").submit(io.BytesIO(b"x"), "clip.mp4")
    assert conn.path == "/jobs"
    conn.response = _FakeHTTPResponse(200, b'{"job_id": "j_1", "status": "queued"}')
    client = VideoWorkerClient("127.0.0.1:8200/base")  # scheme optional
    assert client.status("j_1")["status"] == "queued"
    assert conn.path == "/base/jobs/j_1"
    # A https worker URL goes over an HTTPS connection.
    https_conn = _with_fake_connection(
        monkeypatch,
        _FakeHTTPResponse(200, b'{"job_id": "j_1", "status": "queued"}'),
        kind="HTTPSConnection")
    client = VideoWorkerClient("https://10.0.0.5:8200")
    assert client.status("j_1")["status"] == "queued"
    assert https_conn.path == "/jobs/j_1"


@pytest.mark.parametrize("url", ["", "http://", "http://:8200", "http://10.0.0.5:notaport"])
def test_client_rejects_unusable_worker_urls(url):
    with pytest.raises(ValueError):
        VideoWorkerClient(url)


def test_client_maps_a_non_json_error_body_to_the_fallback_message(monkeypatch):
    conn = _with_fake_connection(
        monkeypatch, _FakeHTTPResponse(500, b"<html>internal error</html>"))
    client = VideoWorkerClient("http://10.0.0.5:8200")
    with pytest.raises(WorkerClientError) as exc:
        client.status("j_1")
    assert exc.value.status == 500
    assert exc.value.message == "the video processing machine answered 500"


@pytest.mark.parametrize("body", [b"[1, 2, 3]", b'{"job_id": "j_1"}', b'"queued"'])
def test_client_status_rejects_malformed_answers(monkeypatch, body):
    _with_fake_connection(monkeypatch, _FakeHTTPResponse(200, body))
    client = VideoWorkerClient("http://10.0.0.5:8200")
    with pytest.raises(WorkerClientError):
        client.status("j_1")


@pytest.mark.parametrize("body", [b'{"job_id": null}', b'{"status": "queued"}', b'"j_1"'])
def test_client_submit_rejects_a_malformed_job_answer(monkeypatch, body):
    _with_fake_connection(monkeypatch, _FakeHTTPResponse(202, body))
    client = VideoWorkerClient("http://10.0.0.5:8200")
    with pytest.raises(WorkerClientError):
        client.submit(io.BytesIO(b"x"), "clip.mp4")


def test_client_unreachable_gives_status_none_on_every_call(monkeypatch, tmp_path):
    _with_refused_connection(monkeypatch)
    client = VideoWorkerClient("http://10.0.0.5:8200")
    with pytest.raises(WorkerClientError) as exc:
        client.fetch_result("j_1")
    assert exc.value.status is None
    assert "could not reach the video processing machine" in exc.value.message
    # And the library app maps that to the 502 reachability answer.
    fake = FakeWorker()
    fake.status_error = WorkerClientError(
        "could not reach the video processing machine at 10.0.0.5:8200: refused",
        status=None)
    client_app = make_app(tmp_path / "m", fake)
    response = client_app.get("/api/videos/j_000000000001")
    assert response.status_code == 502
    assert response.json() == {"error": UNREACHABLE}
