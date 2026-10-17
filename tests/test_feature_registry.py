"""Acceptance tests for the feature registry (spec: tasks/feature_registry.md)."""
import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

from sideline import library  # noqa: E402
from sideline.features import FeatureContext, register_all  # noqa: E402
from sideline.features._common import (FPS, PITCH_X, PITCH_Y, ball_xy,  # noqa: E402
                                       by_frame, frame_seconds,
                                       people_xy, rounded, window_ranges)
from sideline.matches import MatchStore  # noqa: E402

SRC = Path(os.environ.get("SIDELINE_SRC", "src")).resolve()
FEATURES_DIR = Path(sys.modules["sideline.features"].__file__).resolve().parent


def recording(n=60, x0=-20.0):
    """A tiny recording: players, a goalkeeper, a referee, a ball, colours."""
    lines = []
    for f in range(n):
        people = [
            {"id": 1, "role": "player", "team": "left", "jersey": "9", "box": [10, 10, 20, 40],
             "xy": [x0 + 0.2 * f, 5.0]},
            {"id": 2, "role": "goalkeeper", "team": "left", "jersey": "1", "box": [10, 10, 20, 40],
             "xy": [-48.0, 0.0]},
            {"id": 3, "role": "player", "team": "right", "jersey": "10", "box": [10, 10, 20, 40],
             "xy": [20.0, -5.0]},
            {"id": 4, "role": "referee", "team": None, "jersey": None, "box": [10, 10, 20, 40],
             "xy": [0.0, 0.0]},
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


def counting_store(store):
    """The store with load_frames wrapped in a call counter."""
    loads = {"n": 0}
    original = store.load_frames

    def counting(match_id):
        loads["n"] += 1
        return original(match_id)

    store.load_frames = counting
    return loads


@pytest.fixture()
def client(tmp_path):
    return TestClient(library.create_library_app(tmp_path / "matches"))


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


GOOD = """\
def register(router, ctx):
    @router.get("/zz")
    def zz(match_id: str):
        ctx.require(match_id)
        return {"feature": "zz"}


def register_app(app, ctx):
    @app.get("/api/zz-app")
    def zz_app():
        return {"feature": "zz-app"}
"""

BROKEN_IMPORT = 'raise RuntimeError("import boom")\n'
BROKEN_REGISTER = ('def register(router, ctx):\n'
                   '    raise ValueError("register boom")\n')
BROKEN_APP = """\
def register(router, ctx):
    @router.get("/zz-before")
    def zz_before(match_id: str):
        return {"partial": "match"}


def register_app(app, ctx):
    @app.get("/api/zz-app-partial")
    def zz_app_partial():
        return {"partial": "app"}
    raise RuntimeError("app boom")
"""
NO_REGISTER = "# a module with no register(): not a feature\n"


# -- the example feature ------------------------------------------------------

def test_ping_feature_reports_the_frame_count(client):
    mid = upload(client).json()["id"]
    assert client.get(f"/api/matches/{mid}/ping").json() == {"match_id": mid, "frames": 60}


def test_ping_404_body_carries_an_error(client):
    body = client.get("/api/matches/nope/ping").json()
    assert "error" in body["detail"]


@pytest.mark.parametrize("bad_id", ["", "nope", "m_", "m_123", "m_012345678",
                                    "m_01234567890", "m_012345678g", "m_0123456789",
                                    "../etc/passwd"])
def test_ping_is_404_for_unknown_and_malformed_ids(client, bad_id):
    assert client.get(f"/api/matches/{bad_id}/ping").status_code == 404


def test_require_returns_meta_or_raises_404(tmp_path):
    store = MatchStore(tmp_path / "m")
    ctx = FeatureContext(store)
    mid = add_match(store, recording())
    assert ctx.require(mid) == store.get(mid)
    for bad in ("nope", "m_123", "../etc/passwd", "m_0123456789"):
        with pytest.raises(HTTPException) as excinfo:
            ctx.require(bad)
        assert excinfo.value.status_code == 404
        assert isinstance(excinfo.value.detail, dict)
        assert "error" in excinfo.value.detail


# -- the registry -------------------------------------------------------------

def test_features_index_lists_the_loaded_feature(client):
    listed = client.get("/api/features").json()["features"]
    assert {"name": "ping", "ok": True, "error": None,
            "routes": ["/api/matches/{match_id}/ping"]} in listed


def test_register_all_mounts_the_router_and_the_index(tmp_path):
    app = FastAPI()
    results = register_all(app, FeatureContext(MatchStore(tmp_path / "m")))
    names = [entry["name"] for entry in results]
    assert "ping" in names and names == sorted(names)         # other features may exist next to the example
    client = TestClient(app)
    assert client.get("/api/features").json()["features"] == results
    assert any(route.path == "/api/matches/{match_id}/ping" for route in app.routes)


def test_features_index_reports_every_feature_sorted_by_name(scratch, tmp_path):
    scratch("zz_good", GOOD)
    scratch("zz_broken_import", BROKEN_IMPORT)
    scratch("zz_broken_register", BROKEN_REGISTER)
    scratch("zz_broken_app", BROKEN_APP)
    scratch("zz_broken_noreg", NO_REGISTER)
    client = TestClient(library.create_library_app(tmp_path / "m"))
    everything = client.get("/api/features").json()["features"]
    assert [f["name"] for f in everything] == sorted(f["name"] for f in everything)
    features = [f for f in everything if f["name"] == "ping" or f["name"].startswith("zz_")]
    assert [f["name"] for f in features] == [
        "ping", "zz_broken_app", "zz_broken_import", "zz_broken_noreg",
        "zz_broken_register", "zz_good"]
    by_name = {f["name"]: f for f in features}
    assert by_name["ping"] == {"name": "ping", "ok": True, "error": None,
                               "routes": ["/api/matches/{match_id}/ping"]}
    assert by_name["zz_good"] == {"name": "zz_good", "ok": True, "error": None,
                                  "routes": ["/api/matches/{match_id}/zz", "/api/zz-app"]}
    assert by_name["zz_broken_import"] == {
        "name": "zz_broken_import", "ok": False,
        "error": "RuntimeError: import boom", "routes": []}
    assert by_name["zz_broken_register"] == {
        "name": "zz_broken_register", "ok": False,
        "error": "ValueError: register boom", "routes": []}
    assert by_name["zz_broken_app"] == {
        "name": "zz_broken_app", "ok": False,
        "error": "RuntimeError: app boom", "routes": []}
    assert by_name["zz_broken_noreg"]["ok"] is False
    assert by_name["zz_broken_noreg"]["error"].startswith("AttributeError")
    assert by_name["zz_broken_noreg"]["routes"] == []


def test_broken_features_are_skipped_without_blocking_the_others(scratch, tmp_path):
    scratch("zz_good", GOOD)
    scratch("zz_broken_import", BROKEN_IMPORT)
    scratch("zz_broken_register", BROKEN_REGISTER)
    client = TestClient(library.create_library_app(tmp_path / "m"))
    mid = upload(client).json()["id"]
    assert client.get(f"/api/matches/{mid}/ping").status_code == 200
    assert client.get(f"/api/matches/{mid}/zz").json() == {"feature": "zz"}
    assert client.get("/api/zz-app").json() == {"feature": "zz-app"}


def test_failing_register_app_rolls_back_the_partial_routes(scratch, tmp_path):
    scratch("zz_broken_app", BROKEN_APP)
    client = TestClient(library.create_library_app(tmp_path / "m"))
    mid = upload(client).json()["id"]
    assert client.get(f"/api/matches/{mid}/zz-before").status_code == 404
    assert client.get("/api/zz-app-partial").status_code == 404
    assert client.get(f"/api/matches/{mid}/ping").status_code == 200  # others unaffected


def test_failed_features_are_logged_as_json_lines(scratch, tmp_path, caplog):
    scratch("zz_broken_import", BROKEN_IMPORT)
    scratch("zz_broken_register", BROKEN_REGISTER)
    with caplog.at_level(logging.ERROR, logger="sideline.features"):
        library.create_library_app(tmp_path / "m")
    lines = [record.message for record in caplog.records
             if record.levelno == logging.ERROR and record.name == "sideline.features"]
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert all(entry["event"] == "feature_load_failed" for entry in parsed)
    by_feature = {entry["feature"]: entry for entry in parsed}
    assert by_feature["zz_broken_import"]["error"] == "RuntimeError: import boom"
    assert by_feature["zz_broken_register"]["error"] == "ValueError: register boom"


def test_create_library_app_registers_the_features_once(tmp_path, monkeypatch):
    real = library.register_all
    calls = {"n": 0}

    def spy(app, ctx):
        calls["n"] += 1
        return real(app, ctx)

    monkeypatch.setattr(library, "register_all", spy)
    app = library.create_library_app(tmp_path / "m")
    assert calls["n"] == 1
    with TestClient(app) as client:
        assert client.get("/api/features").status_code == 200
        assert client.get("/healthz").json() == {"status": "ok"}


# -- FeatureContext ------------------------------------------------------------

def test_meta_and_analytics_accessors(tmp_path):
    store = MatchStore(tmp_path / "m")
    mid = add_match(store, recording())
    ctx = FeatureContext(store)
    assert ctx.meta(mid) == store.get(mid)
    assert ctx.meta("m_0123456789") is None
    summary = ctx.analytics(mid)
    assert summary["fps"] == 25 and summary["frame_count"] == 60
    assert ctx.analytics(mid) == summary


def test_frames_cache_is_an_lru_of_four(tmp_path):
    store = MatchStore(tmp_path / "m")
    loads = counting_store(store)
    ctx = FeatureContext(store)
    mids = [add_match(store, recording(x0=-20.0 - 5.0 * i), "game%d" % i) for i in range(5)]
    for mid in mids[:4]:
        assert len(ctx.frames(mid)) == 60
    assert loads["n"] == 4
    assert ctx.frames(mids[0]) is not None
    assert loads["n"] == 4                      # served from the cache
    assert len(ctx.frames(mids[4])) == 60
    assert loads["n"] == 5                      # a miss evicts the LRU match (mids[1])
    assert ctx.frames(mids[1]) is not None
    assert loads["n"] == 6                      # evicted: read again


def test_frames_cache_invalidates_after_delete_and_replace(tmp_path):
    store = MatchStore(tmp_path / "m")
    loads = counting_store(store)
    ctx = FeatureContext(store)
    data = recording()
    mid = add_match(store, data)
    first = ctx.frames(mid)
    assert loads["n"] == 1
    assert ctx.frames(mid) == first
    assert loads["n"] == 1                      # served from the cache
    assert store.delete(mid) is True
    assert ctx.frames(mid) is None
    assert loads["n"] == 1                      # a deleted match is not read
    add_match(store, data)                      # the same bytes: the same id, a new file
    assert ctx.frames(mid) == first
    assert loads["n"] == 2                      # the cache did not survive the delete
    frames_path = store.root / mid / "frames.jsonl"
    frames_path.write_text(
        json.dumps({"frame": 0, "delay": 0, "people": [], "ball": None, "colors": None}) + "\n",
        encoding="utf-8")
    replaced = ctx.frames(mid)
    assert loads["n"] == 3                      # a replaced file is read again
    assert [f["frame"] for f in replaced] == [0]


def test_cache_json_serves_the_store_until_frames_are_newer(tmp_path):
    store = MatchStore(tmp_path / "m")
    mid = add_match(store, recording())
    ctx = FeatureContext(store)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"value": calls["n"]}

    assert ctx.cache_json(mid, "thing", compute) == {"value": 1}
    assert (store.root / mid / "features" / "thing.json").is_file()
    assert ctx.cache_json(mid, "thing", compute) == {"value": 1}
    assert calls["n"] == 1                      # served from the file
    frames_path = store.root / mid / "frames.jsonl"
    st = frames_path.stat()
    os.utime(frames_path, (st.st_atime, st.st_mtime + 10))
    assert ctx.cache_json(mid, "thing", compute) == {"value": 2}
    assert calls["n"] == 2                      # recomputed after the frames changed
    os.utime(frames_path, (st.st_atime, st.st_mtime))  # frames back to their old time
    assert ctx.cache_json(mid, "thing", compute) == {"value": 2}
    assert calls["n"] == 2                      # and cached again


def test_cache_json_writes_atomically_and_never_clobbers_on_failure(tmp_path):
    store = MatchStore(tmp_path / "m")
    mid = add_match(store, recording())
    ctx = FeatureContext(store)
    ctx.cache_json(mid, "thing", lambda: {"value": 1})
    ctx.cache_json(mid, "other", lambda: {"value": [1, 2, 3]})
    cache_dir = store.root / mid / "features"
    assert sorted(p.name for p in cache_dir.iterdir()) == ["other.json", "thing.json"]
    assert json.loads((cache_dir / "other.json").read_text(encoding="utf-8")) == {"value": [1, 2, 3]}
    # make the stored "thing" stale, then let the recompute fail
    frames_path = store.root / mid / "frames.jsonl"
    st = frames_path.stat()
    os.utime(frames_path, (st.st_atime, st.st_mtime + 10))

    def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        ctx.cache_json(mid, "thing", boom)
    assert json.loads((cache_dir / "thing.json").read_text(encoding="utf-8")) == {"value": 1}
    assert sorted(p.name for p in cache_dir.iterdir()) == ["other.json", "thing.json"]


# -- the pure helpers -----------------------------------------------------------

def test_module_constants():
    assert FPS == 25
    assert PITCH_X == 52.5
    assert PITCH_Y == 34.0


def test_by_frame_maps_numbers_to_frames():
    frames = [{"frame": 100, "people": []}, {"frame": 102, "people": []}]
    mapping = by_frame(frames)
    assert set(mapping) == {100, 102}
    assert mapping[100] is frames[0] and mapping[102] is frames[1]


def test_people_xy_filters_by_role_team_and_position():
    frame = {"people": [
        {"id": 7, "role": "player", "team": "left", "xy": [-12.34, 5.6]},
        {"id": 8, "role": "goalkeeper", "team": "right", "xy": [50.0, 0.0]},
        {"id": 9, "role": "referee", "team": None, "xy": [0.0, 0.0]},
        {"id": 10, "role": "player", "team": "left", "xy": None},
        {"id": 11, "role": "player", "team": "right"},
    ]}
    assert people_xy(frame) == [(7, -12.34, 5.6), (8, 50.0, 0.0)]
    assert people_xy(frame, team="left") == [(7, -12.34, 5.6)]
    assert people_xy(frame, team="right") == [(8, 50.0, 0.0)]
    assert people_xy(frame, roles=("referee",)) == [(9, 0.0, 0.0)]
    assert people_xy(frame, team="left", roles=("goalkeeper",)) == []
    assert people_xy({"people": []}) == []
    assert people_xy({}) == []


def test_ball_xy_returns_the_position_or_none():
    assert ball_xy({"ball": {"box": [1, 2, 3, 4], "xy": [3.1, -2.0]}}) == (3.1, -2.0)
    assert ball_xy({"ball": {"box": [1, 2, 3, 4], "xy": None}}) is None
    assert ball_xy({"ball": None}) is None
    assert ball_xy({}) is None


def test_frame_seconds_counts_from_the_first_frame():
    assert frame_seconds(0, 0) == 0.0
    assert frame_seconds(25, 0) == 1.0
    assert frame_seconds(125, 0) == 5.0
    assert frame_seconds(125, 25) == 4.0


def test_rounded_rounds_floats_in_depth_and_keeps_other_types():
    source = {"a": 1.23456, "b": [2.5678, {"c": 3.999, "d": 0.004}],
              "s": "x", "n": None, "i": 7, "t": True}
    assert rounded(source) == {"a": 1.23, "b": [2.57, {"c": 4.0, "d": 0.0}],
                               "s": "x", "n": None, "i": 7, "t": True}
    assert rounded(source, ndigits=0) == {"a": 1.0, "b": [3.0, {"c": 4.0, "d": 0.0}],
                                          "s": "x", "n": None, "i": 7, "t": True}
    assert rounded(1.5678, 2) == 1.57
    assert rounded(7, 2) == 7 and isinstance(rounded(7, 2), int)


def test_window_ranges_covers_inclusive_ranges():
    assert window_ranges(0, 9, 5) == [(0, 4), (5, 9)]
    assert window_ranges(100, 124, 10) == [(100, 109), (110, 119), (120, 124)]
    assert window_ranges(0, 24, 25) == [(0, 24)]
    assert window_ranges(7, 7, 25) == [(7, 7)]
    assert window_ranges(10, 9, 5) == []
    assert window_ranges(0, 9, 0) == []


# -- import hygiene ---------------------------------------------------------------

def test_importing_the_package_does_not_import_heavy_packages():
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import sideline.features, sideline.features._common, sideline.features.ping; "
        "bad = [m for m in ('torch', 'cv2', 'numpy', 'sideline.pipeline') "
        "if m in sys.modules]; print(','.join(bad)); sys.exit(1 if bad else 0)"
    ) % str(SRC)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
