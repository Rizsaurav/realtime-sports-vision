"""Extra adversarial tests for the video worker (tasks/video_worker.md).

The acceptance suite (tests/test_worker.py) covers the happy paths; these
tests probe the specification's boundary values and the rules it does not
check there:

- the 409 on /result for a *queued* (not just running/failed) job;
- progress must stay within 0.0-1.0 even when the runner reports more
  frames than the estimated total (or a negative count);
- a runner that never calls on_progress must still leave a done job whose
  frame count matches the result file (0-frame and 3-frame results);
- the field types and ranges of GET /jobs/{id};
- FIFO order with three queued jobs and the health counters while they wait;
- many repeated uploads (distinct ids, all processed);
- deleting a job twice, deleting a queued job (it must never be processed);
- the disk-space boundary (free == max_bytes is *not* "below", so it is
  accepted) and the size boundary across a chunk boundary;
- restart: a previously failed job keeps its error, a done job's result is
  still streamed; a job finished 23h ago is *not* purged while one finished
  25h ago (done or failed) is purged on the next upload; a recovered queued
  job runs before newly uploaded ones;
- the ffprobe wiring: a stub ffprobe on PATH supplies the duration that
  drives progress, and a missing or failing ffprobe leaves progress at 0
  until done;
- ids that do not match ^j_[0-9a-f]{12}$ (including encoded path traversal)
  are 404 without touching the file system;
- the worker CLI: `python -m sideline worker` wiring (default host/port and
  the --max-gb to bytes conversion) and the default 2 GiB limit.

The runner is faked everywhere except the ffprobe PATH stubs; no real video
or models are used, so the suite is fast and deterministic.
"""

import json
import logging
import os
import re
import sys
import threading
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

from fastapi.testclient import TestClient  # noqa: E402

from sideline import __main__ as cli  # noqa: E402
from sideline import worker as worker_mod  # noqa: E402
from sideline.worker import create_worker_app  # noqa: E402

ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class FakeRunner:
    """Stands in for the ``sideline run`` subprocess.

    Writes ``lines`` result lines, reports progress 1..lines, and blocks on
    ``block`` once ``pause_after`` frames have been reported.
    """

    def __init__(self, lines=0, pause_after=None, block=None):
        self.lines = lines
        self.pause_after = pause_after
        self.block = block

    def __call__(self, video, out, on_progress):
        with open(out, "wb") as fh:
            for i in range(self.lines):
                fh.write((json.dumps({"frame": i}) + "\n").encode("utf-8"))
        for i in range(1, self.lines + 1):
            on_progress(i)
            if (self.pause_after is not None and i == self.pause_after
                    and self.block is not None):
                if not self.block.wait(15):
                    raise RuntimeError("fake runner: block timed out")


def post(client, filename="a.mp4", data=b"video-bytes", name=None):
    files = {"file": (filename, data)}
    fields = {"name": name} if name is not None else {}
    return client.post("/jobs", data=fields, files=files)


def wait_for(client, job_id, status, timeout=10.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = client.get("/jobs/%s" % job_id).json()
        if last["status"] == status:
            return last
        time.sleep(0.02)
    raise AssertionError("job %s did not reach %r, last state: %r"
                         % (job_id, status, last))


def wait_running_with_frames(client, job_id, frames, timeout=10.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = client.get("/jobs/%s" % job_id).json()
        if last["status"] == "running" and last["frames"] >= frames:
            return last
        if last["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)
    raise AssertionError("job %s never showed frames>=%d while running: %r"
                         % (job_id, frames, last))


def _install_ffprobe(bin_dir, body):
    """A fake ffprobe executable that prints (or exits) per ``body``."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "ffprobe"
    stub.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    stub.chmod(0o755)


def _write_state(root, job_id, **over):
    """A job folder on disk as left by an earlier worker process."""
    d = root / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / (job_id + ".mp4")).write_bytes(b"vid")
    state = {
        "job_id": job_id, "status": "done", "progress": 1.0, "frames": 2,
        "name": "n", "error": None, "created_at": "2025-01-01T00:00:00Z",
        "seconds": 0.08, "ext": ".mp4", "finished_at": "2025-01-01T00:00:00Z",
    }
    state.update(over)
    (d / "job.json").write_text(json.dumps(state), encoding="utf-8")


# -- result endpoint: every not-done status is a 409 -------------------------

def test_result_is_409_for_queued_job(tmp_path):
    hold = threading.Event()
    app = create_worker_app(tmp_path,
                            runner=FakeRunner(lines=1, pause_after=1, block=hold))
    client = TestClient(app)
    j1 = post(client).json()["job_id"]
    wait_for(client, j1, "running")
    j2 = post(client, filename="two.mp4").json()["job_id"]
    r = client.get("/jobs/%s/result" % j2)
    assert r.status_code == 409
    assert "error" in r.json()
    hold.set()
    wait_for(client, j1, "done")
    wait_for(client, j2, "done")
    app.state.worker.stop()


# -- progress must stay within 0.0-1.0 ---------------------------------------

def test_progress_clamped_to_one_when_frames_outrun_estimate(tmp_path, monkeypatch):
    # 1.0 s of video -> an estimated total of 25 frames, the runner writes 100.
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: 1.0)
    hold = threading.Event()
    app = create_worker_app(tmp_path,
                            runner=FakeRunner(lines=100, pause_after=50, block=hold))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    job = wait_running_with_frames(client, job_id, 50)
    assert job["progress"] == 1.0
    hold.set()
    done = wait_for(client, job_id, "done")
    assert done["progress"] == 1.0
    assert done["frames"] == 100
    app.state.worker.stop()


def test_negative_progress_report_stays_in_range(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: 1.0)
    hold = threading.Event()

    def runner(video, out, on_progress):
        with open(out, "wb") as fh:
            fh.write(b'{"frame": 0}\n')
        on_progress(-10)  # malformed report: must not push progress below 0
        assert hold.wait(15)
        on_progress(1)

    app = create_worker_app(tmp_path, runner=runner)
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    deadline = time.monotonic() + 10
    job = None
    while time.monotonic() < deadline:  # wait until the negative report landed
        job = client.get("/jobs/%s" % job_id).json()
        if job["status"] == "running":
            break
        time.sleep(0.02)
    assert job["status"] == "running"
    assert 0.0 <= job["progress"] <= 1.0
    assert job["progress"] == 0.0
    hold.set()
    assert wait_for(client, job_id, "done")["status"] == "done"
    app.state.worker.stop()


# -- frame counts when the runner is silent or writes nothing -----------------

def test_done_job_reports_frames_from_result_when_runner_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)

    def runner(video, out, on_progress):  # never calls on_progress
        with open(out, "wb") as fh:
            for i in range(3):
                fh.write((json.dumps({"frame": i}) + "\n").encode("utf-8"))

    app = create_worker_app(tmp_path, runner=runner)
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    done = wait_for(client, job_id, "done")
    assert done["frames"] == 3
    assert done["progress"] == 1.0
    assert done["seconds"] == 0.12  # 3 frames at 25 fps
    r = client.get("/jobs/%s/result" % job_id)
    assert r.status_code == 200
    assert [json.loads(line)["frame"] for line in r.text.splitlines()] == [0, 1, 2]
    app.state.worker.stop()


def test_done_job_with_empty_result(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=0))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    done = wait_for(client, job_id, "done")
    assert done["frames"] == 0
    assert done["progress"] == 1.0
    assert done["seconds"] == 0.0
    # the stored file name is generated, the client's file name is not used
    video = tmp_path / job_id / (job_id + ".mp4")
    assert video.read_bytes() == b"video-bytes"
    r = client.get("/jobs/%s/result" % job_id)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert r.text == ""
    app.state.worker.stop()


def test_failed_job_has_nonempty_error(tmp_path):
    def runner(video, out, on_progress):
        raise RuntimeError("")  # empty message: the job still needs a clear error

    app = create_worker_app(tmp_path, runner=runner)
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    job = wait_for(client, job_id, "failed")
    assert isinstance(job["error"], str) and job["error"].strip()
    assert client.get("/jobs/%s/result" % job_id).status_code == 409
    app.state.worker.stop()


def test_large_result_file_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)
    total = 5000

    def runner(video, out, on_progress):
        with open(out, "wb") as fh:
            for i in range(total):
                fh.write((json.dumps({"frame": i}) + "\n").encode("utf-8"))
        for k in range(1, 5):
            on_progress(k * total // 5)

    app = create_worker_app(tmp_path, runner=runner)
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    done = wait_for(client, job_id, "done")
    assert done["frames"] == total
    r = client.get("/jobs/%s/result" % job_id)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = r.text.splitlines()
    assert len(lines) == total
    assert json.loads(lines[0])["frame"] == 0
    assert json.loads(lines[-1])["frame"] == total - 1
    app.state.worker.stop()


# -- the job shape: types and ranges -------------------------------------------

def test_job_field_types_and_ranges(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=2))
    client = TestClient(app)
    job_id = post(client, filename="a.mp4").json()["job_id"]
    job = wait_for(client, job_id, "done")
    assert re.fullmatch(r"j_[0-9a-f]{12}", job["job_id"])
    assert job["status"] == "done"
    assert isinstance(job["progress"], float) and 0.0 <= job["progress"] <= 1.0
    assert job["progress"] == 1.0
    assert isinstance(job["frames"], int) and job["frames"] == 2
    assert isinstance(job["name"], str) and job["name"]
    assert "/" not in job["name"] and "\\" not in job["name"]
    assert job["error"] is None
    assert ISO_RE.fullmatch(job["created_at"])
    assert isinstance(job["seconds"], float) and job["seconds"] == 0.08
    app.state.worker.stop()


# -- queueing and repeated calls ------------------------------------------------

def test_fifo_order_with_three_jobs(tmp_path):
    hold = threading.Event()
    order = []
    started = {"count": 0}

    def runner(video, out, on_progress):
        vid = Path(str(video)).stem
        order.append(vid)
        started["count"] += 1
        if started["count"] == 1:  # hold the slot so two jobs pile up behind
            assert hold.wait(15)
        with open(out, "wb") as fh:
            fh.write(b'{"frame": 0}\n')
        on_progress(1)

    app = create_worker_app(tmp_path, runner=runner)
    client = TestClient(app)
    j1 = post(client, filename="one.mp4").json()["job_id"]
    wait_for(client, j1, "running")
    j2 = post(client, filename="two.mp4").json()["job_id"]
    j3 = post(client, filename="three.mp4").json()["job_id"]
    assert client.get("/health").json() == {"status": "ok", "busy": True, "queued": 2}
    assert client.get("/jobs/%s" % j2).json()["status"] == "queued"
    assert client.get("/jobs/%s" % j3).json()["status"] == "queued"
    hold.set()
    for job_id in (j1, j2, j3):
        assert wait_for(client, job_id, "done")["frames"] == 1
    assert order == [j1, j2, j3]
    assert client.get("/health").json() == {"status": "ok", "busy": False, "queued": 0}
    app.state.worker.stop()


def test_repeated_uploads_all_finish_with_distinct_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    ids = []
    for i in range(10):
        r = post(client, filename="v%02d.mp4" % i)
        assert r.status_code == 202
        assert r.json()["status"] == "queued"
        ids.append(r.json()["job_id"])
    assert len(set(ids)) == 10
    for job_id in ids:
        assert wait_for(client, job_id, "done")["frames"] == 1
    assert client.get("/health").json() == {"status": "ok", "busy": False, "queued": 0}
    app.state.worker.stop()


# -- deletion --------------------------------------------------------------------

def test_delete_twice_second_is_404(tmp_path):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    wait_for(client, job_id, "done")
    r = client.delete("/jobs/%s" % job_id)
    assert r.status_code == 204
    assert r.content == b""
    assert client.delete("/jobs/%s" % job_id).status_code == 404
    assert client.get("/jobs/%s" % job_id).status_code == 404
    app.state.worker.stop()


def test_deleted_queued_job_is_never_processed(tmp_path):
    hold = threading.Event()
    processed = []
    first = {"blocked": False}

    def runner(video, out, on_progress):
        vid = Path(str(video)).stem
        processed.append(vid)
        if not first["blocked"]:  # hold the slot for the second (deleted) job
            first["blocked"] = True
            assert hold.wait(15)
        with open(out, "wb") as fh:
            fh.write(b'{}\n')
        on_progress(1)

    app = create_worker_app(tmp_path, runner=runner)
    client = TestClient(app)
    j1 = post(client, filename="keep.mp4").json()["job_id"]
    wait_for(client, j1, "running")
    j2 = post(client, filename="drop.mp4").json()["job_id"]
    assert client.delete("/jobs/%s" % j2).status_code == 204
    hold.set()
    wait_for(client, j1, "done")
    assert processed == [j1]  # the deleted job never reached the runner
    assert client.get("/jobs/%s" % j2).status_code == 404
    assert client.get("/health").json()["queued"] == 0
    app.state.worker.stop()


def test_injected_runner_receives_cancel_event(tmp_path):
    """A runner that declares a ``cancel`` parameter is called with the
    job's cancel event, so a DELETE of a running job reaches the runner
    without breaking the runner signature, and the single worker thread
    (and the FIFO queue behind it) unblocks once the runner honours it."""
    events = []

    def runner(video, out, on_progress, cancel=None):
        events.append(cancel)
        with open(out, "wb") as fh:
            fh.write(b'{}\n')
        on_progress(1)
        assert cancel is not None and isinstance(cancel, threading.Event)
        assert cancel.wait(15)  # honour the cancellation, then return

    def wait_events(n):
        deadline = time.monotonic() + 5
        while len(events) < n:
            if time.monotonic() > deadline:
                raise AssertionError("runners did not start: %r" % events)
            time.sleep(0.01)

    app = create_worker_app(tmp_path, runner=runner)
    client = TestClient(app)
    for i, name in enumerate(("one.mp4", "two.mp4"), 1):
        job_id = post(client, filename=name).json()["job_id"]
        wait_for(client, job_id, "running")
        assert client.delete("/jobs/%s" % job_id).status_code == 204
        wait_events(i)
        assert events[i - 1].is_set()  # the DELETE propagated to the runner
        assert not (tmp_path / job_id).exists()
    time.sleep(0.3)
    # the worker thread is not stalled: idle and nothing left queued
    assert client.get("/health").json() == {"status": "ok", "busy": False, "queued": 0}
    app.state.worker.stop()


# -- upload boundaries -------------------------------------------------------------

def test_disk_space_boundary_free_equals_limit_accepted(tmp_path, monkeypatch):
    free = {"v": 100}
    monkeypatch.setattr(
        worker_mod.shutil, "disk_usage",
        lambda path: types.SimpleNamespace(total=1000, used=900, free=free["v"]))
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1), max_bytes=100)
    client = TestClient(app)
    # free space exactly at the limit is not "below max_bytes": accepted
    r = post(client, data=b"a" * 10)
    assert r.status_code == 202
    ok_id = r.json()["job_id"]
    wait_for(client, ok_id, "done")
    # one byte less free: rejected, and no job is created
    free["v"] = 99
    r = post(client, filename="b.mp4", data=b"a" * 10)
    assert r.status_code == 507
    assert "error" in r.json()
    assert [p.name for p in tmp_path.iterdir()] == [ok_id]
    app.state.worker.stop()


def test_413_at_chunk_boundary_and_accept_at_limit(tmp_path):
    limit = 1024 * 1024 + 4  # forces the check across a 1 MiB chunk boundary
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1), max_bytes=limit)
    client = TestClient(app)
    r = post(client, data=b"a" * (limit + 1))
    assert r.status_code == 413
    assert "error" in r.json()
    assert list(tmp_path.iterdir()) == []  # no partial job left behind
    r = post(client, filename="exact.mp4", data=b"a" * limit)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    wait_for(client, job_id, "done")
    assert (tmp_path / job_id / (job_id + ".mp4")).read_bytes() == b"a" * limit
    app.state.worker.stop()


def test_one_byte_limit_boundary(tmp_path):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=0), max_bytes=1)
    client = TestClient(app)
    r = post(client, data=b"a")  # exactly the limit: accepted
    assert r.status_code == 202
    ok_id = r.json()["job_id"]
    assert wait_for(client, ok_id, "done")["frames"] == 0
    r = post(client, filename="b.mp4", data=b"ab")
    assert r.status_code == 413
    assert "error" in r.json()
    assert client.get("/health").json()["queued"] == 0  # nothing was queued
    assert [p.name for p in tmp_path.iterdir()] == [ok_id]
    app.state.worker.stop()


def test_file_field_must_be_a_file_part(tmp_path):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    r = client.post("/jobs", data={"file": "a.mp4", "name": "x"})
    assert r.status_code == 400
    assert "error" in r.json()
    assert list(tmp_path.iterdir()) == []
    app.state.worker.stop()


def test_name_field_does_not_affect_stored_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    r = post(client, filename="a.mp4", name="report.MOV")
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    job = wait_for(client, job_id, "done")
    assert job["name"] == "report.MOV"
    videos = [p.name for p in (tmp_path / job_id).iterdir() if p.name.startswith(job_id)]
    # the stored video is <job_id>.<ext> of the uploaded file, not the name field
    assert len(videos) == 1 and videos[0].lower() == job_id + ".mp4"
    app.state.worker.stop()


@pytest.mark.parametrize("name, filename",
                         [("   ", "a.mp4"),
                          ("dir\\sub\\clip", "a.mp4"),
                          ("plain name", "a.b.mp4")])
def test_name_is_sanitised_and_extension_from_file(tmp_path, monkeypatch, name, filename):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    r = post(client, filename=filename, name=name)
    assert r.status_code == 202
    job = wait_for(client, r.json()["job_id"], "done")
    assert isinstance(job["name"], str) and job["name"].strip()
    assert "/" not in job["name"] and "\\" not in job["name"]
    app.state.worker.stop()


# -- restart and retention -----------------------------------------------------------

def test_corrupted_job_state_does_not_block_startup(tmp_path):
    """A state file with a non-numeric field must be skipped (and logged),
    not crash the service, and a valid job in the same root must still be
    recovered."""
    good = "j_0000000000e5"
    _write_state(tmp_path, good, status="queued", progress=0.0, frames=0,
                 error=None, seconds=None,
                 created_at="2025-06-01T00:00:00Z", finished_at=None)
    broken = "j_0000000000f6"
    d = tmp_path / broken
    d.mkdir()
    (d / (broken + ".mp4")).write_bytes(b"vid")
    (d / "job.json").write_text(json.dumps({
        "job_id": broken, "status": "queued", "progress": "garbage",
        "frames": 3, "name": "broken", "error": None,
        "created_at": "2025-01-01T00:00:00Z", "seconds": None,
        "ext": ".mp4", "finished_at": None}), encoding="utf-8")

    order = []

    def runner(video, out, on_progress):
        order.append(Path(str(video)).stem)
        with open(out, "wb") as fh:
            fh.write(b'{}\n')
        on_progress(1)

    app = create_worker_app(tmp_path, runner=runner)  # must not raise
    client = TestClient(app)
    assert client.get("/jobs/%s" % broken).status_code == 404  # skipped
    assert wait_for(client, good, "done")["status"] == "done"  # valid one recovered
    assert order == [good]
    app.state.worker.stop()


def test_disk_full_mid_upload_removes_partial_job(tmp_path, monkeypatch):
    """An OSError mid-stream (e.g. ENOSPC on write) must surface as a 507
    with an error and must not leave an orphaned partial job folder.
    """
    real_open = open

    class FlakyFile:
        def __init__(self, fh):
            self._fh = fh
            self._first = True

        def write(self, data):
            if self._first:  # one chunk lands, the next hits ENOSPC
                self._first = False
                return self._fh.write(data)
            raise OSError(28, "No space left on device")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._fh.__exit__(*exc)

    def fake_open(path, mode="r", *args, **kwargs):
        fh = real_open(path, mode, *args, **kwargs)
        if "w" in str(mode) and str(path).endswith(".mp4"):
            return FlakyFile(fh)
        return fh

    monkeypatch.setattr("builtins.open", fake_open)
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    r = post(client, data=b"a" * (2 * 1024 * 1024 + 1))  # spans two chunks
    assert r.status_code == 507
    assert "error" in r.json()
    assert list(tmp_path.iterdir()) == []  # no orphaned partial job
    assert client.get("/health").json()["queued"] == 0
    app.state.worker.stop()


def test_state_write_failure_fails_the_job(tmp_path, monkeypatch, caplog):
    """A failed state write is logged and turns the job failed, so the
    in-memory view cannot claim 'done' while the on-disk state is stale.
    """
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)

    def broken(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(worker_mod.Worker, "_atomic_write", staticmethod(broken))
    caplog.at_level(logging.ERROR, logger="sideline.worker")
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=2))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    job = wait_for(client, job_id, "failed", timeout=15.0)
    assert "state could not be persisted" in job["error"]
    assert any("could not persist" in rec.getMessage() for rec in caplog.records)
    app.state.worker.stop()


def test_restart_keeps_failed_job_and_serves_done_result(tmp_path):
    failed = "j_0000000000fa"
    _write_state(tmp_path, failed, status="failed", progress=0.5, frames=12,
                 error="disk full: cannot write result",
                 finished_at="2025-06-01T00:00:00Z")
    done = "j_0000000000db"
    _write_state(tmp_path, done)
    (tmp_path / done / "result.jsonl").write_text(
        '{"frame": 0}\n{"frame": 1}\n', encoding="utf-8")

    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    job = client.get("/jobs/%s" % failed).json()
    assert job["status"] == "failed"
    assert "disk full" in job["error"]
    assert client.get("/jobs/%s/result" % failed).status_code == 409
    done_state = client.get("/jobs/%s" % done).json()
    assert done_state["status"] == "done"  # not re-processed on restart
    assert done_state["frames"] == 2
    r = client.get("/jobs/%s/result" % done)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert [json.loads(line)["frame"] for line in r.text.splitlines()] == [0, 1]
    app.state.worker.stop()


def test_job_finished_within_24h_not_purged(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)
    old = "j_0000000000a1"
    stamp = (datetime.now(timezone.utc) - timedelta(hours=23)
             ).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_state(tmp_path, old, created_at=stamp, finished_at=stamp)

    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    new_id = post(client).json()["job_id"]  # a new upload triggers the lazy purge
    wait_for(client, new_id, "done")
    assert client.get("/jobs/%s" % old).json()["status"] == "done"
    assert (tmp_path / old).is_dir()
    app.state.worker.stop()


def test_finished_jobs_older_than_24h_purged_on_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)
    done_old = "j_0000000000b2"
    failed_old = "j_0000000000c3"
    done_stamp = (datetime.now(timezone.utc) - timedelta(hours=25)
                  ).strftime("%Y-%m-%dT%H:%M:%SZ")
    failed_stamp = (datetime.now(timezone.utc) - timedelta(hours=30)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_state(tmp_path, done_old, created_at=done_stamp, finished_at=done_stamp)
    _write_state(tmp_path, failed_old, status="failed", error="boom",
                 progress=0.0, frames=0, seconds=None,
                 created_at=failed_stamp, finished_at=failed_stamp)

    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    new_id = post(client).json()["job_id"]
    wait_for(client, new_id, "done")
    # both finished jobs (done and failed) are gone; the new one remains
    assert client.get("/jobs/%s" % done_old).status_code == 404
    assert client.get("/jobs/%s" % failed_old).status_code == 404
    assert not (tmp_path / done_old).exists()
    assert not (tmp_path / failed_old).exists()
    assert [p.name for p in tmp_path.iterdir()] == [new_id]
    app.state.worker.stop()


def test_recovered_queued_job_runs_before_new_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)
    pending = "j_0000000000d4"
    _write_state(tmp_path, pending, status="queued", progress=0.0, frames=0,
                 error=None, seconds=None,
                 created_at="2025-06-01T00:00:00Z", finished_at=None)

    order = []

    def runner(video, out, on_progress):
        order.append(Path(str(video)).stem)
        with open(out, "wb") as fh:
            fh.write(b'{}\n')
        on_progress(1)

    app = create_worker_app(tmp_path, runner=runner)
    client = TestClient(app)
    new_id = post(client, filename="fresh.mp4").json()["job_id"]
    wait_for(client, pending, "done")
    wait_for(client, new_id, "done")
    assert order == [pending, new_id]  # the recovered job kept its place first
    app.state.worker.stop()


# -- ffprobe wiring -------------------------------------------------------------------

def test_ffprobe_duration_drives_progress(tmp_path, monkeypatch):
    bin_dir = tmp_path / "ffprobe-bin"
    _install_ffprobe(bin_dir, 'printf "2.0\\n"\n')  # a 2 s video: 50 frames total
    monkeypatch.setenv("PATH", str(bin_dir))
    hold = threading.Event()
    app = create_worker_app(tmp_path / "root",
                            runner=FakeRunner(lines=50, pause_after=25, block=hold))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    job = wait_running_with_frames(client, job_id, 25)
    assert job["seconds"] == 2.0  # the probed duration
    assert job["progress"] == pytest.approx(0.5)
    hold.set()
    done = wait_for(client, job_id, "done")
    assert done["progress"] == 1.0
    assert done["seconds"] == 2.0
    app.state.worker.stop()


def test_ffprobe_missing_or_failing_falls_back_to_zero_progress(tmp_path, monkeypatch):
    # no ffprobe on PATH at all: progress stays 0 until done
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    hold = threading.Event()
    app = create_worker_app(tmp_path / "root1",
                            runner=FakeRunner(lines=5, pause_after=2, block=hold))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    job = wait_running_with_frames(client, job_id, 2)
    assert job["seconds"] is None
    assert job["progress"] == 0.0
    hold.set()
    done = wait_for(client, job_id, "done")
    assert done["seconds"] == 0.2  # 5 frames at 25 fps
    assert done["progress"] == 1.0
    app.state.worker.stop()

    # ffprobe present but failing: same fallback
    fail_bin = tmp_path / "fail-bin"
    _install_ffprobe(fail_bin, "exit 3\n")
    monkeypatch.setenv("PATH", str(fail_bin))
    hold2 = threading.Event()
    app2 = create_worker_app(tmp_path / "root2",
                             runner=FakeRunner(lines=5, pause_after=2, block=hold2))
    client2 = TestClient(app2)
    job_id2 = post(client2).json()["job_id"]
    job2 = wait_running_with_frames(client2, job_id2, 2)
    assert job2["seconds"] is None
    assert job2["progress"] == 0.0
    hold2.set()
    done2 = wait_for(client2, job_id2, "done")
    assert done2["seconds"] == 0.2
    app2.state.worker.stop()


# -- ids -------------------------------------------------------------------------------

def test_encoded_traversal_id_is_404(tmp_path):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    assert client.get("/jobs/j_0123456789ab%2F..%2F..%2Fx").status_code == 404
    assert client.get("/jobs/j_0123456789ab%2F..%2F..%2Fx/result").status_code == 404
    assert client.delete("/jobs/j_0123456789ab%2F..").status_code == 404
    assert list(tmp_path.iterdir()) == []  # the file system was not touched
    app.state.worker.stop()


# -- app construction and the worker CLI -------------------------------------------------

def test_create_worker_app_creates_root_and_default_limit(tmp_path):
    root = tmp_path / "nested" / "worker"
    app = create_worker_app(root, runner=FakeRunner(lines=0))
    assert root.is_dir()
    assert app.state.worker.max_bytes == 2 * 1024 ** 3  # the documented default
    app.state.worker.stop()


def _fake_app_factory(monkeypatch):
    created = {}
    sentinel = object()

    def factory(root, runner=None, max_bytes=2 * 1024 ** 3):
        created["root"] = root
        created["runner"] = runner
        created["max_bytes"] = max_bytes
        return sentinel

    monkeypatch.setattr(worker_mod, "create_worker_app", factory)
    return created, sentinel


def test_cli_worker_defaults_wire_uvicorn(tmp_path, monkeypatch):
    uvicorn = mock.MagicMock(name="uvicorn")
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    created, sentinel = _fake_app_factory(monkeypatch)
    root = tmp_path / "r"
    monkeypatch.setattr(sys, "argv", ["sideline", "worker", "--root", str(root)])
    cli.main()
    assert created["root"] == str(root)
    assert created["runner"] is None  # the service uses the default subprocess runner
    assert created["max_bytes"] == 2 * 1024 ** 3  # --max-gb defaults to 2
    args, kwargs = uvicorn.run.call_args
    assert args[0] is sentinel
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8200


def test_cli_worker_flags_wire_uvicorn(tmp_path, monkeypatch):
    uvicorn = mock.MagicMock(name="uvicorn")
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    created, sentinel = _fake_app_factory(monkeypatch)
    root = tmp_path / "r"
    monkeypatch.setattr(sys, "argv",
                        ["sideline", "worker", "--root", str(root),
                         "--host", "0.0.0.0", "--port", "9111", "--max-gb", "0.5"])
    cli.main()
    assert created["max_bytes"] == 536870912  # 0.5 GiB in bytes
    args, kwargs = uvicorn.run.call_args
    assert args[0] is sentinel
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9111
