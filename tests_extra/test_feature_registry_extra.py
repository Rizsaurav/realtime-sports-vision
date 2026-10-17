"""Extra tests for the feature registry (tasks/feature_registry.md).

The acceptance suite (tests/test_feature_registry.py) covers the spec'd
behaviour; these probe the error paths it does not:

- FeatureContext.frames with malformed and non-string ids, and with the
  frames file missing while the meta still exists;
- cache_json refusing ids and names that would escape the store root
  (path traversal), treating a corrupt or unreadable cache file as
  stale, and surviving a failed rename (the old file stays, the temp
  file goes);
- people_xy skipping people entries that are not dicts;
- a feature module that sys.exit()s on import, or inside register:
  skipped and rolled back like any other broken feature, the app still
  starts, the failure is logged.
"""
import importlib
import json
import logging
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

from sideline import library  # noqa: E402
from sideline.features import FeatureContext  # noqa: E402
from sideline.features._common import people_xy  # noqa: E402
from sideline.matches import MatchStore  # noqa: E402

FEATURES_DIR = Path(sys.modules["sideline.features"].__file__).resolve().parent


def recording(n=60, x0=-20.0):
    """A tiny recording: players, a goalkeeper, a referee, a ball."""
    lines = []
    for f in range(n):
        people = [
            {"id": 1, "role": "player", "team": "left", "jersey": "9",
             "box": [10, 10, 20, 40], "xy": [x0 + 0.2 * f, 5.0]},
            {"id": 2, "role": "goalkeeper", "team": "left", "jersey": "1",
             "box": [10, 10, 20, 40], "xy": [-48.0, 0.0]},
        ]
        ball = {"box": [5, 5, 6, 6], "xy": [-5.0 + 0.3 * f, 1.0]} if f % 2 == 0 else None
        lines.append(json.dumps({"frame": f, "delay": 0, "people": people, "ball": ball,
                                 "colors": {"left": "#d03030", "right": "#2050c0"}}))
    return ("\n".join(lines) + "\n").encode()


def add_match(store, data, name="game"):
    meta, _created = store.add(data, name)
    return meta["id"]


def upload(client, data=None, name="game"):
    return client.post("/api/matches",
                       files={"file": (name + ".jsonl", data or recording(), "application/json")},
                       data={"name": name})


@pytest.fixture()
def scratch():
    """Temporary feature modules in the real package, removed afterwards."""
    created = []

    def add(name, source):
        path = FEATURES_DIR / (name + ".py")
        path.write_text(source, encoding="utf-8")
        created.append(path)

    yield add
    importlib.invalidate_caches()
    stems = {path.stem for path in created}
    for module in [m for m in list(sys.modules) if m.startswith("sideline.features.")]:
        if module.rsplit(".", 1)[1] in stems:
            del sys.modules[module]
    for path in created:
        path.unlink()


SYS_EXIT_IMPORT = "import sys\nsys.exit(42)\n"
SYS_EXIT_REGISTER = """\
def register(router, ctx):
    @router.get('/zz-exit')
    def zz_exit(match_id: str):
        return {'partial': 'exit'}
    import sys
    sys.exit(3)
"""


# -- FeatureContext error paths -------------------------------------------

def test_frames_rejects_malformed_and_non_string_ids(tmp_path):
    store = MatchStore(tmp_path / "m")
    mid = add_match(store, recording())
    ctx = FeatureContext(store)
    for bad in ("nope", "m_", "m_012345678", "m_012345678g", "m_01234567890",
                "m_0123456789/../x", 123, None, 4.5):
        assert ctx.frames(bad) is None
    assert len(ctx.frames(mid)) == 60  # still works after the rejections


def test_frames_is_none_when_the_frames_file_is_missing(tmp_path):
    store = MatchStore(tmp_path / "m")
    mid = add_match(store, recording())
    ctx = FeatureContext(store)
    assert len(ctx.frames(mid)) == 60
    (store.root / mid / "frames.jsonl").unlink()
    assert store.get(mid) is not None            # the meta still exists
    assert ctx.frames(mid) is None               # but the frames are gone


def test_people_xy_skips_people_entries_that_are_not_dicts():
    frame = {"people": [42, None, "x",
                        {"id": 1, "role": "player", "team": "left", "xy": [1.0, 2.0]}]}
    assert people_xy(frame) == [(1, 1.0, 2.0)]


# -- cache_json error paths --------------------------------------------------

def test_cache_json_refuses_ids_and_names_that_escape_the_store_root(tmp_path):
    store = MatchStore(tmp_path / "m")
    mid = add_match(store, recording())
    ctx = FeatureContext(store)
    for bad_id in ("..", "m_012345678g", "m_0123456789/../..", 123, None, "m_0123456789/"):
        with pytest.raises(ValueError):
            ctx.cache_json(bad_id, "thing", lambda: {"v": 1})
    for bad_name in ("../../x", "a/b", "../evil", "..", ".", ".hidden", "",
                     "has space", 7, None):
        with pytest.raises(ValueError):
            ctx.cache_json(mid, bad_name, lambda: {"v": 1})
    # nothing was written outside (or inside) the store
    assert not (tmp_path / "features").exists()
    assert not (store.root / "x.json").exists()
    assert not (store.root / mid / "features").exists()
    # a plain dotted/dashed name is still fine
    assert ctx.cache_json(mid, "a.b-c", lambda: {"v": 1}) == {"v": 1}
    assert (store.root / mid / "features" / "a.b-c.json").is_file()


def test_cache_json_corrupt_cache_file_is_recomputed(tmp_path):
    store = MatchStore(tmp_path / "m")
    mid = add_match(store, recording())
    ctx = FeatureContext(store)
    assert ctx.cache_json(mid, "thing", lambda: {"value": 1}) == {"value": 1}
    cache_path = store.root / mid / "features" / "thing.json"
    frames_path = store.root / mid / "frames.jsonl"
    st = frames_path.stat()
    os.utime(frames_path, (st.st_atime, st.st_mtime + 10))
    cache_path.write_text("{not json", encoding="utf-8")
    stc = cache_path.stat()
    os.utime(cache_path, (stc.st_atime, stc.st_mtime + 10))  # corrupt but fresh
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"value": calls["n"]}

    assert ctx.cache_json(mid, "thing", compute) == {"value": 1}
    assert calls["n"] == 1                        # not served, recomputed
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"value": 1}


def test_cache_json_unreadable_cache_file_is_recomputed(tmp_path, monkeypatch):
    store = MatchStore(tmp_path / "m")
    mid = add_match(store, recording())
    ctx = FeatureContext(store)
    assert ctx.cache_json(mid, "thing", lambda: {"value": 1}) == {"value": 1}
    cache_path = store.root / mid / "features" / "thing.json"
    frames_path = store.root / mid / "frames.jsonl"
    st = frames_path.stat()
    os.utime(frames_path, (st.st_atime, st.st_mtime + 10))
    stc = cache_path.stat()
    os.utime(cache_path, (stc.st_atime, stc.st_mtime + 20))  # fresh, so it would be served

    def broken(self, encoding=None):
        raise OSError("read denied")

    monkeypatch.setattr(Path, "read_text", broken)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"value": calls["n"]}

    assert ctx.cache_json(mid, "thing", compute) == {"value": 1}
    assert calls["n"] == 1                        # the unreadable file was not served


def test_cache_json_temp_cleanup_failure_still_raises_the_original(tmp_path, monkeypatch):
    store = MatchStore(tmp_path / "m")
    mid = add_match(store, recording())
    ctx = FeatureContext(store)
    ctx.cache_json(mid, "thing", lambda: {"value": 1})
    cache_dir = store.root / mid / "features"
    cache_path = cache_dir / "thing.json"
    frames_path = store.root / mid / "frames.jsonl"
    st = frames_path.stat()
    os.utime(frames_path, (st.st_atime, st.st_mtime + 10))

    def boom(src, dst):
        raise OSError("disk full")

    def stuck(path):
        raise OSError("stuck")

    monkeypatch.setattr("sideline.features._common.os.replace", boom)
    monkeypatch.setattr("sideline.features._common.os.unlink", stuck)
    with pytest.raises(OSError, match="disk full"):   # the original failure propagates
        ctx.cache_json(mid, "thing", lambda: {"value": 2})
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"value": 1}


def test_cache_json_failed_rename_keeps_the_old_file_and_leaves_no_temp(tmp_path, monkeypatch):
    store = MatchStore(tmp_path / "m")
    mid = add_match(store, recording())
    ctx = FeatureContext(store)
    ctx.cache_json(mid, "thing", lambda: {"value": 1})
    cache_dir = store.root / mid / "features"
    cache_path = cache_dir / "thing.json"
    frames_path = store.root / mid / "frames.jsonl"
    st = frames_path.stat()
    os.utime(frames_path, (st.st_atime, st.st_mtime + 10))

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("sideline.features._common.os.replace", boom)
    with pytest.raises(OSError):
        ctx.cache_json(mid, "thing", lambda: {"value": 2})
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"value": 1}
    assert sorted(p.name for p in cache_dir.iterdir()) == ["thing.json"]


def test_cache_json_mid_write_failure_keeps_the_old_file(tmp_path):
    store = MatchStore(tmp_path / "m")
    mid = add_match(store, recording())
    ctx = FeatureContext(store)
    ctx.cache_json(mid, "thing", lambda: {"value": 1})
    cache_dir = store.root / mid / "features"
    cache_path = cache_dir / "thing.json"
    frames_path = store.root / mid / "frames.jsonl"
    st = frames_path.stat()
    os.utime(frames_path, (st.st_atime, st.st_mtime + 10))
    with pytest.raises(TypeError):                # json.dump of a non-serialisable value
        ctx.cache_json(mid, "thing", lambda: object())
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"value": 1}
    assert sorted(p.name for p in cache_dir.iterdir()) == ["thing.json"]


# -- a feature that sys.exit()s is skipped, never fatal ----------------------

def test_sys_exit_on_import_skips_the_feature_and_the_app_starts(scratch, tmp_path):
    scratch("zz_sys_exit_import", SYS_EXIT_IMPORT)
    client = TestClient(library.create_library_app(tmp_path / "m"))
    by_name = {f["name"]: f for f in client.get("/api/features").json()["features"]}
    assert by_name["zz_sys_exit_import"] == {
        "name": "zz_sys_exit_import", "ok": False, "error": "SystemExit: 42", "routes": []}
    assert by_name["ping"]["ok"] is True
    mid = upload(client).json()["id"]
    assert client.get(f"/api/matches/{mid}/ping").status_code == 200


def test_sys_exit_in_register_is_skipped_and_rolled_back(scratch, tmp_path):
    scratch("zz_sys_exit_register", SYS_EXIT_REGISTER)
    client = TestClient(library.create_library_app(tmp_path / "m"))
    by_name = {f["name"]: f for f in client.get("/api/features").json()["features"]}
    assert by_name["zz_sys_exit_register"]["ok"] is False
    assert by_name["zz_sys_exit_register"]["error"].startswith("SystemExit")
    assert by_name["zz_sys_exit_register"]["routes"] == []
    mid = upload(client).json()["id"]
    assert client.get(f"/api/matches/{mid}/zz-exit").status_code == 404  # rolled back
    assert client.get(f"/api/matches/{mid}/ping").status_code == 200     # others unaffected


def test_sys_exit_feature_is_logged_like_any_other_failure(scratch, tmp_path, caplog):
    scratch("zz_sys_exit_import", SYS_EXIT_IMPORT)
    with caplog.at_level(logging.ERROR, logger="sideline.features"):
        library.create_library_app(tmp_path / "m")
    lines = [record.message for record in caplog.records
             if record.levelno == logging.ERROR and record.name == "sideline.features"]
    parsed = [json.loads(line) for line in lines]
    assert {"event": "feature_load_failed", "feature": "zz_sys_exit_import",
            "error": "SystemExit: 42"} in parsed
