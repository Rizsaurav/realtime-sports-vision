"""Tests for the video worker service (src/sideline/worker.py).

The runner is faked everywhere: no real video, no models. The worker
thread and the TestClient run in separate threads, so a job that blocks
its runner can be driven from the test with an event.
"""

import json
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sideline import worker as worker_mod
from sideline.__main__ import build_parser, _cmd_worker
from sideline.worker import create_worker_app

ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class FakeRunner:
    """Stands in for the ``sideline run`` subprocess.

    lines: number of result lines to write (and report via on_progress).
    error: exception to raise instead of processing.
    pause_after: report up to this many frames, then wait on block.
    delay: seconds to sleep before starting.
    """

    def __init__(self, lines=0, error=None, pause_after=None, block=None, delay=0.0):
        self.lines = lines
        self.error = error
        self.pause_after = pause_after
        self.block = block
        self.delay = delay
        self.started = []

    def __call__(self, video, out, on_progress):
        self.started.append(str(video))
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        with open(out, "wb") as fh:
            for i in range(self.lines):
                fh.write((json.dumps({"frame": i}) + "\n").encode("utf-8"))
        for i in range(1, self.lines + 1):
            on_progress(i)
            if self.pause_after is not None and i == self.pause_after and self.block is not None:
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
    raise AssertionError("job %s did not reach %r, last state: %r" % (job_id, status, last))


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


# -- health, upload, job shape ---------------------------------------------

def test_health_and_job_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=2))
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok", "busy": False, "queued": 0}
    r = post(client)
    assert r.status_code == 202
    body = r.json()
    assert set(body) == {"job_id", "status"}
    assert body["status"] == "queued"
    assert re.fullmatch(r"j_[0-9a-f]{12}", body["job_id"])
    job = wait_for(client, body["job_id"], "done")
    assert set(job) == {"job_id", "status", "progress", "frames", "name",
                        "error", "created_at", "seconds"}
    assert job["frames"] == 2
    assert job["progress"] == 1.0
    assert job["name"] == "a"
    assert job["error"] is None
    assert ISO_RE.fullmatch(job["created_at"])
    assert job["seconds"] == 0.08  # 2 frames at 25 fps, duration not probed
    assert client.get("/health").json() == {"status": "ok", "busy": False, "queued": 0}
    app.state.worker.stop()


def test_name_field_wins_and_is_stored(tmp_path):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    job_id = post(client, filename="cam1.mp4", name="Sunday league").json()["job_id"]
    job = wait_for(client, job_id, "done")
    assert job["name"] == "Sunday league"
    app.state.worker.stop()


# -- queueing ----------------------------------------------------------------

def test_queue_is_fifo_one_at_a_time(tmp_path):
    hold = threading.Event()
    order = []
    first = {}

    def runner(video, out, on_progress):
        vid = Path(str(video)).stem
        while "id" not in first:  # the test names job 1 right after the uploads
            time.sleep(0.01)
        if vid == first["id"]:
            order.append("start:1")
            with open(out, "wb") as fh:
                fh.write(b'{"frame": 0}\n')
            on_progress(1)
            assert hold.wait(15)
            order.append("end:1")
        else:
            order.append("start:2")
            with open(out, "wb") as fh:
                fh.write(b'{"frame": 0}\n{"frame": 1}\n')
            on_progress(1)
            on_progress(2)
            order.append("end:2")

    app = create_worker_app(tmp_path, runner=runner)
    client = TestClient(app)
    job1 = post(client, filename="one.mp4").json()["job_id"]
    job2 = post(client, filename="two.mp4").json()["job_id"]
    first["id"] = job1
    wait_for(client, job1, "running")
    assert client.get("/jobs/%s" % job2).json()["status"] == "queued"
    assert client.get("/health").json() == {"status": "ok", "busy": True, "queued": 1}
    hold.set()
    wait_for(client, job1, "done")
    wait_for(client, job2, "done")
    assert order == ["start:1", "end:1", "start:2", "end:2"]
    assert client.get("/health").json() == {"status": "ok", "busy": False, "queued": 0}
    app.state.worker.stop()


def test_failing_runner_marks_job_failed(tmp_path):
    app = create_worker_app(tmp_path,
                            runner=FakeRunner(error=RuntimeError("boom in fake runner")))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    job = wait_for(client, job_id, "failed")
    assert "boom in fake runner" in job["error"]
    assert job["progress"] == 0.0
    assert client.get("/jobs/%s/result" % job_id).status_code == 409
    app.state.worker.stop()


def test_progress_reporting(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: 2.0)  # 50 frames expected
    hold = threading.Event()
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=50, pause_after=10, block=hold))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    job = wait_running_with_frames(client, job_id, 10)
    assert job["seconds"] == 2.0
    assert job["progress"] == pytest.approx(0.2)
    hold.set()
    done = wait_for(client, job_id, "done")
    assert done["progress"] == 1.0
    assert done["frames"] == 50
    app.state.worker.stop()


def test_progress_stays_zero_until_done_without_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_probe_duration", lambda path: None)
    hold = threading.Event()
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=4, pause_after=2, block=hold))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    job = wait_running_with_frames(client, job_id, 2)
    assert job["progress"] == 0.0
    hold.set()
    assert wait_for(client, job_id, "done")["progress"] == 1.0
    app.state.worker.stop()


# -- deletion and cancellation -----------------------------------------------

def test_delete_cancels_running_job(tmp_path):
    release = threading.Event()
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=5, pause_after=1, block=release))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    wait_for(client, job_id, "running")
    assert client.delete("/jobs/%s" % job_id).status_code == 204
    assert client.get("/jobs/%s" % job_id).status_code == 404
    assert not (tmp_path / job_id).exists()
    release.set()  # the late runner must be dropped, not resurrect the job
    time.sleep(0.3)
    assert client.get("/jobs/%s" % job_id).status_code == 404
    # the worker thread is still alive and processes the next job
    next_id = post(client, filename="next.mp4").json()["job_id"]
    wait_for(client, next_id, "done")
    app.state.worker.stop()


def test_delete_queued_job(tmp_path):
    hold = threading.Event()
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1, pause_after=1, block=hold))
    client = TestClient(app)
    job1 = post(client, filename="one.mp4").json()["job_id"]
    job2 = post(client, filename="two.mp4").json()["job_id"]
    wait_for(client, job1, "running")
    assert client.delete("/jobs/%s" % job2).status_code == 204
    assert client.get("/jobs/%s" % job2).status_code == 404
    assert not (tmp_path / job2).exists()
    hold.set()
    wait_for(client, job1, "done")
    assert [p.name for p in tmp_path.iterdir()] == [job1]
    app.state.worker.stop()


def test_delete_done_job(tmp_path):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    wait_for(client, job_id, "done")
    assert client.delete("/jobs/%s" % job_id).status_code == 204
    assert client.get("/jobs/%s" % job_id).status_code == 404
    assert not (tmp_path / job_id).exists()
    app.state.worker.stop()


# -- names and paths ----------------------------------------------------------

def test_client_filename_never_used_in_paths(tmp_path):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    job_id = post(client, filename="../../x.mp4").json()["job_id"]
    job = wait_for(client, job_id, "done")
    assert job["name"] == "x"
    video = tmp_path / job_id / (job_id + ".mp4")
    assert video.is_file()
    evil = post(client, filename="y.mp4", name="../../../evil name").json()["job_id"]
    assert wait_for(client, evil, "done")["name"] == "evil name"
    assert {p.name for p in tmp_path.iterdir()} == {job_id, evil}
    app.state.worker.stop()


# -- upload validation ---------------------------------------------------------

def test_size_limit_with_tiny_max_bytes(tmp_path):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1), max_bytes=100)
    client = TestClient(app)
    r = post(client, data=b"a" * 101)
    assert r.status_code == 413
    assert "error" in r.json()
    assert list(tmp_path.iterdir()) == []
    r = post(client, data=b"a" * 100)
    assert r.status_code == 202
    app.state.worker.stop()


@pytest.mark.parametrize("filename", ["a.txt", "a", "a.mp4.bak", "movie.MP3"])
def test_wrong_extension_rejected(tmp_path, filename):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    r = post(client, filename=filename)
    assert r.status_code == 400
    assert "error" in r.json()
    assert list(tmp_path.iterdir()) == []
    app.state.worker.stop()


def test_uppercase_extension_accepted(tmp_path):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    r = post(client, filename="movie.MP4")
    assert r.status_code == 202
    assert (tmp_path / r.json()["job_id"] / (r.json()["job_id"] + ".MP4")).is_file()
    app.state.worker.stop()


def test_empty_file_rejected(tmp_path):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    r = post(client, data=b"")
    assert r.status_code == 400
    assert "error" in r.json()
    assert list(tmp_path.iterdir()) == []
    app.state.worker.stop()


def test_large_upload_round_trips(tmp_path):
    data = bytes(range(256)) * 12288  # 3 MiB
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    r = post(client, data=data)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    wait_for(client, job_id, "done")
    assert (tmp_path / job_id / (job_id + ".mp4")).read_bytes() == data
    app.state.worker.stop()


# -- result endpoint -------------------------------------------------------------

def test_result_streaming_and_409_until_done(tmp_path):
    hold = threading.Event()
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=3, pause_after=1, block=hold))
    client = TestClient(app)
    job_id = post(client).json()["job_id"]
    wait_for(client, job_id, "running")
    r = client.get("/jobs/%s/result" % job_id)
    assert r.status_code == 409
    assert "error" in r.json()
    hold.set()
    wait_for(client, job_id, "done")
    r = client.get("/jobs/%s/result" % job_id)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert [json.loads(line)["frame"] for line in r.text.splitlines()] == [0, 1, 2]
    app.state.worker.stop()


# -- ids and the 404 gate ----------------------------------------------------------

def test_missing_file_field_rejected(tmp_path):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    r = client.post("/jobs", data={"name": "no file"})
    assert r.status_code == 400
    assert "error" in r.json()
    assert list(tmp_path.iterdir()) == []
    app.state.worker.stop()


@pytest.mark.parametrize("job_id", ["nope", "j_short", "j_ZZZZZZZZZZZZ",
                                    "j_0123456789abX", "J_0123456789ab", "j_0123456789a",
                                    "j_ffffffffffff"])
def test_malformed_ids_404_without_side_effects(tmp_path, job_id):
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    assert client.get("/jobs/%s" % job_id).status_code == 404
    assert client.get("/jobs/%s/result" % job_id).status_code == 404
    assert client.delete("/jobs/%s" % job_id).status_code == 404
    assert list(tmp_path.iterdir()) == []
    app.state.worker.stop()


def test_id_gate_ignores_other_directories(tmp_path):
    evil = tmp_path / "evil"
    evil.mkdir()
    (evil / "job.json").write_text("{}", encoding="utf-8")
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    assert client.get("/jobs/evil").status_code == 404
    assert client.get("/health").json()["status"] == "ok"
    app.state.worker.stop()


# -- restart and retention -----------------------------------------------------------

def test_restart_recovery(tmp_path):
    app1 = create_worker_app(tmp_path, runner=FakeRunner(lines=2))
    client1 = TestClient(app1)
    done_id = post(client1).json()["job_id"]
    wait_for(client1, done_id, "done")
    app1.state.worker.stop()

    crashed = "j_deadbeef0001"
    d = tmp_path / crashed
    d.mkdir()
    (d / (crashed + ".mp4")).write_bytes(b"vid")
    (d / "job.json").write_text(json.dumps({
        "job_id": crashed, "status": "running", "progress": 0.4, "frames": 40,
        "name": "crashed", "error": None, "created_at": "2025-01-01T00:00:00Z",
        "seconds": None, "ext": ".mp4", "finished_at": None}), encoding="utf-8")
    pending = "j_cafebabe0002"
    d2 = tmp_path / pending
    d2.mkdir()
    (d2 / (pending + ".mp4")).write_bytes(b"vid")
    (d2 / "job.json").write_text(json.dumps({
        "job_id": pending, "status": "queued", "progress": 0.0, "frames": 0,
        "name": "pending", "error": None, "created_at": "2025-01-01T00:00:00Z",
        "seconds": None, "ext": ".mp4", "finished_at": None}), encoding="utf-8")

    app2 = create_worker_app(tmp_path, runner=FakeRunner(lines=3))
    client2 = TestClient(app2)
    job = client2.get("/jobs/%s" % crashed).json()
    assert job["status"] == "failed"
    assert "restart" in job["error"]
    # the job left queued is picked up again on the restarted worker
    assert wait_for(client2, pending, "done")["frames"] == 3
    # the finished job survives the restart
    assert client2.get("/jobs/%s" % done_id).json()["status"] == "done"
    app2.state.worker.stop()


def test_finished_jobs_purged_after_24_hours(tmp_path):
    old = "j_000000000001"
    d = tmp_path / old
    d.mkdir()
    (d / (old + ".mp4")).write_bytes(b"v")
    (d / "result.jsonl").write_text("{}\n", encoding="utf-8")
    (d / "job.json").write_text(json.dumps({
        "job_id": old, "status": "done", "progress": 1.0, "frames": 1,
        "name": "old", "error": None, "created_at": "2020-01-01T00:00:00Z",
        "seconds": 0.04, "ext": ".mp4", "finished_at": "2020-01-01T00:00:00Z"}),
        encoding="utf-8")
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    assert client.get("/jobs/%s" % old).json()["status"] == "done"
    post(client)  # a new upload triggers the lazy purge
    assert client.get("/jobs/%s" % old).status_code == 404
    assert not d.exists()
    app.state.worker.stop()


def test_disk_full_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod.shutil, "disk_usage",
                        lambda path: types.SimpleNamespace(total=1024, used=1014, free=10))
    app = create_worker_app(tmp_path, runner=FakeRunner(lines=1))
    client = TestClient(app)
    r = post(client)
    assert r.status_code == 507
    assert "error" in r.json()
    assert list(tmp_path.iterdir()) == []
    app.state.worker.stop()


# -- the default runner (subprocess boundary) ------------------------------------

class _FakeProc:
    """Stands in for the sideline run subprocess."""

    def __init__(self, returncode=None):
        self.returncode = returncode
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_default_runner_command_and_failure(tmp_path, monkeypatch):
    seen = {}

    def fake_popen(cmd, env=None, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = env
        return _FakeProc(returncode=1)

    monkeypatch.setattr(worker_mod.subprocess, "Popen", fake_popen)
    cancel = threading.Event()
    with pytest.raises(RuntimeError, match="exit 1"):
        worker_mod._default_runner(cancel)(tmp_path / "v.mp4", tmp_path / "o.jsonl", lambda n: None)
    cmd = seen["cmd"]
    assert cmd[0] == sys.executable
    assert cmd[1:7] == ["-m", "sideline", "run", str(tmp_path / "v.mp4"), "--delay", "5"]
    assert cmd[7:9] == ["--out", str(tmp_path / "o.jsonl")]
    src = Path(worker_mod.__file__).resolve().parent.parent
    assert seen["env"]["PYTHONPATH"].startswith(str(src))


def test_default_runner_kills_the_process_on_cancel(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_POLL_SECONDS", 0.05)
    proc = _FakeProc(returncode=None)  # runs forever
    monkeypatch.setattr(worker_mod.subprocess, "Popen", lambda *a, **k: proc)
    cancel = threading.Event()
    cancel.set()  # the job is deleted before the runner even polls
    worker_mod._default_runner(cancel)(tmp_path / "v.mp4", tmp_path / "o.jsonl", lambda n: None)
    assert proc.killed


def test_default_runner_enforces_the_time_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_POLL_SECONDS", 0.05)
    monkeypatch.setattr(worker_mod, "_TIME_LIMIT", 0.1)
    proc = _FakeProc(returncode=None)  # runs forever
    monkeypatch.setattr(worker_mod.subprocess, "Popen", lambda *a, **k: proc)
    cancel = threading.Event()
    with pytest.raises(RuntimeError, match="time limit"):
        worker_mod._default_runner(cancel)(tmp_path / "v.mp4", tmp_path / "o.jsonl", lambda n: None)
    assert proc.killed


def test_default_runner_reports_progress_from_output_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "_POLL_SECONDS", 0.02)
    reported = []

    class SlowProc(_FakeProc):
        def __init__(self):
            super().__init__(returncode=None)
            self._lines = 0

        def poll(self):
            if self._lines < 3:
                self._lines += 1
                with open(tmp_path / "o.jsonl", "ab") as fh:
                    fh.write(b"{}\n")
                return None
            self.returncode = 0
            return 0

    proc = SlowProc()
    monkeypatch.setattr(worker_mod.subprocess, "Popen", lambda *a, **k: proc)
    worker_mod._default_runner(threading.Event())(
        tmp_path / "v.mp4", tmp_path / "o.jsonl", reported.append)
    assert reported == [1, 2, 3]


# -- import hygiene and CLI -------------------------------------------------------------

def test_import_does_not_load_heavy_modules():
    src = Path(__file__).resolve().parent.parent / "src"
    code = ("import sys; sys.path.insert(0, %r)\n"
            "import sideline.worker\n"
            "heavy = {'torch', 'cv2', 'numpy'} & set(sys.modules)\n"
            "assert not heavy, heavy\n" % str(src))
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_worker_subcommand_parsing():
    args = build_parser().parse_args(["worker", "--root", "r"])
    assert (args.cmd, args.host, args.port, args.max_gb) == ("worker", "127.0.0.1", 8200, 2.0)
    args = build_parser().parse_args(["worker", "--root", "r", "--host", "0.0.0.0",
                                      "--port", "9999", "--max-gb", "0.5"])
    assert (args.host, args.port, args.max_gb) == ("0.0.0.0", 9999, 0.5)


@pytest.mark.parametrize("value", ["0", "-3", "nan", "inf"])
def test_worker_max_gb_rejects_bad_values(value):
    with pytest.raises(SystemExit) as exc:
        _cmd_worker(build_parser().parse_args(["worker", "--root", "r", "--max-gb", value]))
    assert "--max-gb" in str(exc.value)
