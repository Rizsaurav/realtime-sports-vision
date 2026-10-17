"""Extra tests for the library app's defensive error paths
(src/sideline/library.py).

The acceptance suites cover the documented behaviour; these probe the
failure branches they do not: the store walk and the upload spool
meeting I/O errors, the oversize check with a non-numeric
Content-Length, the video job log's corrupt and un-persistable state,
the worker result download cap, and the video routes' I/O-failure
answers.
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

from sideline import library  # noqa: E402
from sideline.matches import MatchStore  # noqa: E402


def recording(n=60, x0=-20.0):
    """A tiny recording, valid for MatchStore.add."""
    lines = []
    for f in range(n):
        people = [
            {"id": 1, "role": "player", "team": "left", "jersey": "9",
             "box": [10, 10, 20, 40], "xy": [x0 + 0.2 * f, 5.0]},
        ]
        ball = {"box": [5, 5, 6, 6], "xy": [-5.0 + 0.3 * f, 1.0]} if f % 2 == 0 else None
        lines.append(json.dumps({"frame": f, "delay": 0, "people": people, "ball": ball,
                                 "colors": {"left": "#d03030", "right": "#2050c0"}}))
    return ("\n".join(lines) + "\n").encode()


class _FakeRequest:
    """Just enough of a Starlette request for _oversize_check."""
    def __init__(self, method="POST", path="/api/matches", headers=None):
        self.method = method
        self.url = type("Url", (), {"path": path})()
        self.headers = headers or {}


# -- the store walk and the store gauges -------------------------------------

def test_store_bytes_is_zero_for_a_missing_root(tmp_path):
    assert library._store_bytes(tmp_path / "nope") == 0


def test_store_bytes_skips_files_it_cannot_stat(tmp_path, monkeypatch, caplog):
    root = tmp_path / "s"
    root.mkdir()
    (root / "a.json").write_bytes(b"x" * 10)
    (root / "b.json").write_bytes(b"y" * 7)
    real = os.path.getsize

    def bad(path):
        if path.endswith("a.json"):
            raise OSError("no permission")
        return real(path)

    monkeypatch.setattr(os.path, "getsize", bad)
    with caplog.at_level(logging.DEBUG, logger="sideline.library"):
        assert library._store_bytes(root) == 7    # the unreadable file is skipped


# -- the oversize check --------------------------------------------------------

def test_oversize_check_ignores_a_non_numeric_content_length(tmp_path, caplog):
    request = _FakeRequest(headers={"content-length": "abc"})
    with caplog.at_level(logging.DEBUG, logger="sideline.library"):
        assert library._oversize_check(request, 100, 10) is None


def test_oversize_check_answers_413_per_route_limit():
    video = library._oversize_check(_FakeRequest(path="/api/videos",
                                                  headers={"content-length": "200"}), 100, 10)
    assert video.status_code == 413
    assert "smaller recording" in video.body.decode()
    match = library._oversize_check(_FakeRequest(headers={"content-length": "20"}), 100, 10)
    assert match.status_code == 413
    under = library._oversize_check(_FakeRequest(headers={"content-length": "9"}), 100, 10)
    assert under is None
    get = library._oversize_check(_FakeRequest(method="GET", headers={"content-length": "999"}),
                                  100, 10)
    assert get is None
    none = library._oversize_check(_FakeRequest(), 100, 10)   # no Content-Length: no early answer
    assert none is None


def test_page_security_headers_are_empty_for_a_missing_page(tmp_path):
    assert library._page_security_headers(tmp_path / "nope.html") == {}


# -- the video job log ---------------------------------------------------------

def test_parse_job_mapping_keeps_the_valid_entries_and_drops_the_rest():
    class BadGet(dict):
        def get(self, key, default=None):
            raise TypeError("boom")

    data = {
        "j_0123456789ab": {"match_id": "m_0123456789"},
        "j_0123456789ac": {"error": "bad frames"},
        123: {"match_id": "m_0123456789"},               # non-str job id: dropped
        "not-a-job": {"match_id": "m_0123456789"},       # bad job id: dropped
        "j_0123456789ad": "not a dict",                  # non-dict entry: dropped
        "j_0123456789ae": {"match_id": "m_bad"},         # bad match id, no error: dropped
        "j_0123456789af": BadGet(),                       # raising entry: dropped
    }
    assert library._parse_job_mapping(data) == {
        "j_0123456789ab": {"match_id": "m_0123456789"},
        "j_0123456789ac": {"error": "bad frames"},
    }
    assert library._parse_job_mapping([1, 2]) == {}
    assert library._parse_job_mapping("payload") == {}


def test_job_log_starts_empty_from_a_corrupt_file_and_keeps_it(tmp_path, caplog):
    root = tmp_path / "m"
    root.mkdir()
    (root / "video_jobs.json").write_text("{corrupt", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="sideline.library"):
        log = library._VideoJobLog(root)
    assert log.get("j_0123456789ab") is None
    assert (root / "video_jobs.json").read_text(encoding="utf-8") == "{corrupt"


def test_job_log_ignores_malformed_job_ids(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    log = library._VideoJobLog(root)
    assert log.get("not-a-job") is None
    assert log.remove("not-a-job") is False
    log.set_done("not-a-job", "m_0123456789")       # ignored, nothing written
    assert not (root / "video_jobs.json").exists()


def test_job_log_write_failure_keeps_the_mapping_in_memory(tmp_path, monkeypatch, caplog):
    root = tmp_path / "m"
    root.mkdir()
    log = library._VideoJobLog(root)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("sideline.library.os.replace", boom)
    with caplog.at_level(logging.ERROR, logger="sideline.library"):
        log.set_done("j_0123456789ab", "m_0123456789")   # no exception escapes
    assert log.get("j_0123456789ab") == {"match_id": "m_0123456789"}


def test_job_log_unlink_failure_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    root = tmp_path / "m"
    root.mkdir()
    log = library._VideoJobLog(root)
    log.set_done("j_0123456789ab", "m_0123456789")

    def boom(self, missing_ok=False):
        raise OSError("busy")

    monkeypatch.setattr(Path, "unlink", boom)
    with caplog.at_level(logging.WARNING, logger="sideline.library"):
        assert log.remove("j_0123456789ab") is True    # the forget still counts


def test_job_log_temp_cleanup_failure_is_still_a_persistence_error(tmp_path, monkeypatch, caplog):
    root = tmp_path / "m"
    root.mkdir()
    log = library._VideoJobLog(root)

    def boom_replace(src, dst):
        raise OSError("disk full")

    def stuck(path):
        raise OSError("stuck")

    monkeypatch.setattr("sideline.library.os.replace", boom_replace)
    monkeypatch.setattr("sideline.library.os.unlink", stuck)
    with caplog.at_level(logging.ERROR, logger="sideline.library"):
        log.set_done("j_0123456789ab", "m_0123456789")   # no exception escapes
    assert log.get("j_0123456789ab") == {"match_id": "m_0123456789"}


# -- the worker result download --------------------------------------------------

class _FakeStream:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


class _FakeClient:
    def __init__(self, chunks):
        self.stream = _FakeStream(chunks)

    def fetch_result(self, job_id):
        return self.stream


def test_download_result_truncates_at_the_cap_and_closes_the_stream():
    client = _FakeClient([b"a" * 10, b"b" * 10, b"c" * 10])
    data = library._download_result(client, "j_0123456789ab", 15)
    assert data == b"a" * 10 + b"b" * 5
    assert client.stream.closed is True


def test_download_result_closes_the_stream_when_exhausted():
    client = _FakeClient([b"a", b"b"])
    assert library._download_result(client, "j_0123456789ab", 100) == b"ab"
    assert client.stream.closed is True


def test_download_result_works_without_a_close_on_the_stream():
    class NoClose(list):
        pass
    client = type("C", (), {"fetch_result": lambda self, job_id: NoClose([b"z"])})()
    assert library._download_result(client, "j_0123456789ab", 100) == b"z"


def test_download_result_with_a_zero_cap_reads_nothing_and_closes():
    client = _FakeClient([b"a", b"b"])
    assert library._download_result(client, "j_0123456789ab", 0) == b""
    assert client.stream.closed is True


# -- the upload spool --------------------------------------------------------------

class _FakeFile:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    async def read(self, n=-1):
        return self.chunks.pop(0) if self.chunks else b""


def test_spool_upload_fdopen_failure_cleans_up_and_raises(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no fd")

    monkeypatch.setattr(os, "fdopen", boom)
    with pytest.raises(OSError):
        asyncio.run(library._spool_upload(_FakeFile([b"x"]), ".mp4", b"x", 100))
    assert not [p for p in os.listdir(os.getcwd())
                if p.startswith(".video-upload-")]


def test_remove_spool_unlink_failure_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    spool = tmp_path / "spool"
    spool.write_bytes(b"x")

    def boom(path):
        raise OSError("busy")

    monkeypatch.setattr(os, "unlink", boom)
    with caplog.at_level(logging.WARNING, logger="sideline.library"):
        library._remove_spool(str(spool))


# -- the video routes' I/O-failure answers ------------------------------------------

class _DoneWorker:
    max_bytes = 0

    def __init__(self):
        self.deleted = []

    def status(self, job_id):
        return {"status": "done", "name": "video.mp4"}

    def fetch_result(self, job_id):
        return iter([recording()])

    def delete(self, job_id):
        self.deleted.append(job_id)


def test_video_job_status_store_failure_answers_500(tmp_path, monkeypatch):
    app = FastAPI()
    store = MatchStore(tmp_path / "m")

    def bad_add(raw, name, source=None):
        raise OSError("disk full")

    monkeypatch.setattr(store, "add", bad_add)
    library._video_routes(app, store, _DoneWorker(), library._VideoJobLog(tmp_path / "j"),
                          100, None)
    client = TestClient(app)
    resp = client.get("/api/videos/j_0123456789ab")
    assert resp.status_code == 500
    assert "could not store the processed match" in resp.json()["error"]


def test_video_upload_worker_submit_oserror_answers_500(tmp_path):
    class _BoomWorker:
        max_bytes = 10 ** 9

        def submit(self, stream, filename, name):
            raise OSError("worker disk full")

        def status(self, job_id):
            return {"status": "queued", "progress": 0.0}

        def fetch_result(self, job_id):
            raise AssertionError("not reached")

        def delete(self, job_id):
            pass

    app = library.create_library_app(tmp_path / "m", worker_client=_BoomWorker())
    client = TestClient(app)
    resp = client.post("/api/videos",
                       files={"file": ("v.mp4", b"x" * 100, "video/mp4")},
                       data={"name": "v"})
    assert resp.status_code == 500
    assert "could not be staged" in resp.json()["error"]
