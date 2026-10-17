"""Acceptance tests for src/sideline/matches.py (match library API)."""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sideline import analytics, matches

COLORS = {"left": "#d03030", "right": "#2050c0"}

FAKE_SUMMARY = {
    "fps": 25, "frame_count": 3, "first_frame": 10, "last_frame": 14, "duration_s": 0.2,
    "players": [
        {"id": 7, "role": "player", "team": "left", "jersey": "10", "frames": 2, "seconds": 0.08,
         "distance_m": 0.55, "avg_speed_ms": 6.9, "max_speed_ms": 7.5, "sprints": 0},
        {"id": 9, "role": "referee", "team": None, "jersey": None, "frames": 1, "seconds": 0.04,
         "distance_m": 0.0, "avg_speed_ms": 0.0, "max_speed_ms": 0.0, "sprints": 0},
    ],
    "teams": {}, "heatmaps": {}, "ball": {"frames_with_ball": 0, "distance_m": 0.0},
}


def frames_basic():
    return [
        {"frame": 10, "delay": 25, "colors": None, "ball": None, "people": [
            {"id": 7, "role": "player", "team": "left", "jersey": "10", "box": [1, 2, 3, 4], "xy": [1.5, -2.25]},
            {"id": 9, "role": "referee", "team": None, "jersey": None, "box": [1, 2, 3, 4], "xy": None}]},
        {"frame": 11, "delay": 25, "colors": COLORS, "ball": None, "people": [
            {"id": 7, "role": "player", "team": "left", "jersey": "10", "box": [1, 2, 3, 4], "xy": [2.0, -2.5]},
            {"id": 12, "role": "goalkeeper", "team": "right", "jersey": None, "box": [1, 2, 3, 4], "xy": [-3, 4]}]},
        {"frame": 14, "delay": 25, "colors": None, "ball": None, "people": []},
    ]


def to_bytes(frames, sep="\n"):
    return (sep.join(json.dumps(f) for f in frames) + "\n").encode()


def seq(n, start=0):
    return [{"frame": start + i, "delay": 0, "people": [], "colors": None, "ball": None} for i in range(n)]


@pytest.fixture
def root(tmp_path):
    return tmp_path / "library"


@pytest.fixture
def client(root):
    app = FastAPI()
    app.include_router(matches.create_router(matches.MatchStore(root)))
    return TestClient(app)


@pytest.fixture
def fake(monkeypatch):
    calls = []

    def summarize(frames, fps=25):
        calls.append({"frames": frames, "fps": fps})
        return FAKE_SUMMARY

    monkeypatch.setattr(analytics, "summarize", summarize)
    return calls


def upload(c, data, name=None, filename="game.jsonl", **form):
    if name is not None:
        form["name"] = name
    return c.post("/api/matches", files={"file": (filename, data, "application/json")}, data=form)


def make(c, frames=None, **kw):
    r = upload(c, to_bytes(frames if frames is not None else frames_basic()), **kw)
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------- upload and meta

def test_upload_returns_201_and_meta(client):
    data = to_bytes(frames_basic())
    r = upload(client, data)
    assert r.status_code == 201
    m = r.json()
    assert m["id"] == "m_" + hashlib.sha256(data).hexdigest()[:10]
    assert m["size_bytes"] == len(data)
    assert m["frame_count"] == 3
    assert m["first_frame"] == 10 and m["last_frame"] == 14
    assert m["delay"] == 25


def test_meta_duration_rounded(client):
    m = make(client, seq(7))
    assert m["duration_s"] == 0.28
    m = make(client, seq(100, start=3))
    assert m["duration_s"] == 4.0
    m = make(client, seq(1, start=5))
    assert m["duration_s"] == 0.04


def test_meta_duration_uses_first_and_last_frame(client):
    m = make(client)
    assert m["duration_s"] == 0.2


def test_meta_colors_from_last_frame_with_colors(client):
    assert make(client)["colors"] == COLORS


def test_meta_colors_null_when_never_known(client):
    assert make(client, seq(3))["colors"] is None


def test_meta_colors_last_non_null_wins(client):
    fr = seq(3)
    fr[0]["colors"] = {"left": "#111111", "right": "#222222"}
    fr[1]["colors"] = COLORS
    assert make(client, fr)["colors"] == COLORS


def test_meta_delay_from_first_frame(client):
    fr = seq(3)
    fr[0]["delay"] = 125
    fr[1]["delay"] = 0
    assert make(client, fr)["delay"] == 125


def test_meta_created_at_iso_utc(client):
    m = make(client)
    assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", m["created_at"])


def test_meta_has_no_source_by_default(client):
    assert "source" not in make(client)


def test_meta_source_when_given(client):
    m = make(client, source="clip_01.mp4")
    assert m["source"] == "clip_01.mp4"
    assert client.get("/api/matches/" + m["id"]).json()["source"] == "clip_01.mp4"


def test_default_name_is_file_stem(client):
    assert make(client, filename="derby night.final.jsonl")["name"] == "derby night.final"


def test_explicit_name_is_used(client):
    assert make(client, name="  Derby  ")["name"] == "Derby"


def test_blank_name_falls_back_to_file_stem(client):
    assert make(client, name="   ", filename="abc.jsonl")["name"] == "abc"


def test_duplicate_upload_returns_200_same_match(client):
    data = to_bytes(frames_basic())
    first = upload(client, data, name="First")
    second = upload(client, data, name="Second", filename="other.jsonl")
    assert first.status_code == 201 and second.status_code == 200
    assert second.json() == first.json()
    assert second.json()["name"] == "First"
    assert len(client.get("/api/matches").json()["items"]) == 1


def test_different_bytes_give_different_ids(client):
    a = make(client, seq(3))
    b = make(client, seq(4))
    assert a["id"] != b["id"]


def test_blank_lines_and_crlf_are_accepted(client):
    data = (json.dumps(seq(1)[0]) + "\r\n\r\n   \n" + json.dumps(seq(2)[1]) + "\r\n").encode()
    m = upload(client, data).json()
    assert m["frame_count"] == 2
    assert m["size_bytes"] == len(data)


def test_storage_layout(client, root):
    m = make(client)
    d = root / m["id"]
    assert json.loads((d / "meta.json").read_text()) == m
    assert (d / "frames.jsonl").read_bytes() == to_bytes(frames_basic())
    assert not (d / "analytics.json").exists()


def test_root_directory_is_created(root):
    assert not root.exists()
    matches.MatchStore(root)
    assert root.is_dir()


# ---------------------------------------------------------------- validation

def test_empty_file_is_400(client):
    r = upload(client, b"")
    assert r.status_code == 400 and r.json()["error"]


def test_only_blank_lines_is_400(client):
    r = upload(client, b"\n  \n\n")
    assert r.status_code == 400 and r.json()["error"]


def test_missing_file_field_is_400(client):
    r = client.post("/api/matches", data={"name": "x"})
    assert r.status_code == 400 and r.json()["error"]


def test_invalid_json_names_line_number(client):
    good = [json.dumps(f) for f in seq(3)]
    data = "\n".join(good[:2] + ["{not json"] + good[2:]).encode()
    r = upload(client, data)
    assert r.status_code == 400
    assert re.search(r"\b3\b", r.json()["error"])


def test_line_numbers_count_blank_lines(client):
    good = [json.dumps(f) for f in seq(2)]
    data = ("\n".join([good[0], "", "", "oops", good[1]])).encode()
    r = upload(client, data)
    assert r.status_code == 400
    assert re.search(r"\b4\b", r.json()["error"])


@pytest.mark.parametrize("line", ["[1, 2]", "42", '"text"', "null"])
def test_non_object_line_is_400_with_line_number(client, line):
    data = (json.dumps(seq(1)[0]) + "\n" + line + "\n").encode()
    r = upload(client, data)
    assert r.status_code == 400
    assert re.search(r"\b2\b", r.json()["error"])


@pytest.mark.parametrize("bad", [
    {"people": []},
    {"frame": "3", "people": []},
    {"frame": 3.5, "people": []},
    {"frame": True, "people": []},
    {"frame": None, "people": []},
])
def test_bad_frame_field_is_400_with_line_number(client, bad):
    data = (json.dumps({"frame": 0, "people": []}) + "\n" + json.dumps(bad) + "\n").encode()
    r = upload(client, data)
    assert r.status_code == 400
    assert re.search(r"\b2\b", r.json()["error"])


@pytest.mark.parametrize("bad", [
    {"frame": 5},
    {"frame": 5, "people": None},
    {"frame": 5, "people": {"id": 1}},
    {"frame": 5, "people": "x"},
])
def test_bad_people_field_is_400_with_line_number(client, bad):
    data = (json.dumps({"frame": 0, "people": []}) + "\n" + json.dumps(bad) + "\n").encode()
    r = upload(client, data)
    assert r.status_code == 400
    assert re.search(r"\b2\b", r.json()["error"])


@pytest.mark.parametrize("numbers", [[5, 5], [5, 4], [1, 2, 2], [1, 3, 2]])
def test_frames_must_strictly_increase(client, numbers):
    fr = [{"frame": n, "people": []} for n in numbers]
    r = upload(client, to_bytes(fr))
    assert r.status_code == 400
    bad_line = next(i for i in range(1, len(numbers)) if numbers[i] <= numbers[i - 1]) + 1
    assert re.search(r"\b%d\b" % bad_line, r.json()["error"])


def test_non_utf8_is_400(client):
    r = upload(client, b'{"frame": 1, "people": []}\n\xff\xfe\n')
    assert r.status_code == 400 and r.json()["error"]


def test_invalid_upload_stores_nothing(client, root):
    upload(client, b"nope")
    assert client.get("/api/matches").json() == {"items": []}
    assert [p for p in root.iterdir()] == []


# ---------------------------------------------------------------- size limit

def test_default_max_size_is_100_mib(root):
    assert matches.MatchStore(root).max_bytes == 100 * 1024 * 1024


def test_oversize_upload_is_413(root):
    data = to_bytes(seq(5))
    app = FastAPI()
    app.include_router(matches.create_router(matches.MatchStore(root, max_bytes=len(data) - 1)))
    c = TestClient(app)
    r = upload(c, data)
    assert r.status_code == 413 and r.json()["error"]
    assert c.get("/api/matches").json() == {"items": []}


def test_upload_of_exactly_max_size_is_accepted(root):
    data = to_bytes(seq(5))
    app = FastAPI()
    app.include_router(matches.create_router(matches.MatchStore(root, max_bytes=len(data))))
    assert upload(TestClient(app), data).status_code == 201


def test_oversize_checked_before_content(root):
    app = FastAPI()
    app.include_router(matches.create_router(matches.MatchStore(root, max_bytes=10)))
    assert upload(TestClient(app), b"x" * 50).status_code == 413


# ---------------------------------------------------------------- get / list / delete

def test_get_match(client):
    m = make(client)
    r = client.get("/api/matches/" + m["id"])
    assert r.status_code == 200 and r.json() == m


def test_get_unknown_match_404(client):
    r = client.get("/api/matches/m_0123456789")
    assert r.status_code == 404 and r.json()["error"]


def test_list_empty(client):
    assert client.get("/api/matches").json() == {"items": []}


def test_list_contains_all_meta(client):
    a = make(client, seq(3))
    b = make(client, seq(4))
    items = client.get("/api/matches").json()["items"]
    assert sorted(items, key=lambda m: m["id"]) == sorted([a, b], key=lambda m: m["id"])


def test_list_newest_first_then_id(client, root):
    ms = [make(client, seq(n)) for n in (2, 3, 4, 5)]
    stamps = ["2025-01-01T10:00:00Z", "2025-03-01T10:00:00Z", "2025-02-01T10:00:00Z", "2025-02-01T10:00:00Z"]
    for m, s in zip(ms, stamps):
        m["created_at"] = s
        (root / m["id"] / "meta.json").write_text(json.dumps(m))
    got = [m["id"] for m in client.get("/api/matches").json()["items"]]
    tie = sorted([ms[2]["id"], ms[3]["id"]], reverse=True)
    assert got == [ms[1]["id"]] + tie + [ms[0]["id"]]


def test_delete_then_404(client, root):
    m = make(client)
    r = client.delete("/api/matches/" + m["id"])
    assert r.status_code == 204 and r.content == b""
    assert client.get("/api/matches/" + m["id"]).status_code == 404
    assert client.get("/api/matches").json() == {"items": []}
    assert not (root / m["id"]).exists()
    assert client.delete("/api/matches/" + m["id"]).status_code == 404


def test_delete_unknown_404(client):
    r = client.delete("/api/matches/m_aaaaaaaaaa")
    assert r.status_code == 404 and r.json()["error"]


def test_reupload_after_delete_is_201(client):
    data = to_bytes(frames_basic())
    assert upload(client, data).status_code == 201
    mid = upload(client, data).json()["id"]
    client.delete("/api/matches/" + mid)
    assert upload(client, data).status_code == 201


def test_persistence_across_store_instances(client, root):
    m = make(client, name="Keep")
    app = FastAPI()
    app.include_router(matches.create_router(matches.MatchStore(root)))
    c2 = TestClient(app)
    assert c2.get("/api/matches/" + m["id"]).json() == m
    assert c2.get("/api/matches").json()["items"] == [m]
    assert upload(c2, to_bytes(frames_basic())).status_code == 200
    assert len(c2.get("/api/matches/" + m["id"] + "/frames").json()["frames"]) == 3


# ---------------------------------------------------------------- id safety

@pytest.mark.parametrize("bad", [
    "..", "m_..", "m_0123456789a", "m_012345678", "m_ABCDEF0123", "m_zzzzzzzzzz",
    "x_0123456789", "..%2F..%2Fetc", "m_0123456789%2F..%2F..", "%2e%2e",
])
@pytest.mark.parametrize("suffix", ["", "/frames", "/analytics", "/export.csv"])
def test_malformed_ids_are_404(client, bad, suffix):
    r = client.get("/api/matches/" + bad + suffix)
    assert r.status_code == 404


@pytest.mark.parametrize("bad", ["..", "..%2F..", "m_ABCDEF0123", "m_0123456789%2F.."])
def test_malformed_id_delete_is_404(client, root, bad):
    sentinel = root.parent / "keep.txt"
    sentinel.write_text("x")
    make(client)
    assert client.delete("/api/matches/" + bad).status_code == 404
    assert sentinel.exists() and root.exists()
    assert len(client.get("/api/matches").json()["items"]) == 1


def test_store_get_rejects_malformed_ids(root):
    store = matches.MatchStore(root)
    assert store.get("../x") is None
    assert store.get("m_ABCDEF0123") is None


def test_malformed_id_does_not_read_outside_root(client, root):
    outside = root.parent / "m_0123456789"
    outside.mkdir()
    (outside / "meta.json").write_text("{}")
    (outside / "frames.jsonl").write_text("{}")
    assert client.get("/api/matches/..%2Fm_0123456789").status_code == 404


# ---------------------------------------------------------------- frames

@pytest.fixture
def ten(client):
    m = make(client, seq(10, start=100))
    return m["id"]


def get_frames(c, mid, **q):
    return c.get("/api/matches/%s/frames" % mid, params=q)


def test_frames_default_returns_all(client, ten):
    r = get_frames(client, ten)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 10
    assert [f["frame"] for f in body["frames"]] == list(range(100, 110))
    assert set(body) == {"frames", "total"}


def test_frames_are_the_uploaded_objects(client):
    m = make(client)
    assert get_frames(client, m["id"]).json()["frames"] == frames_basic()


def test_frames_start_end_inclusive(client, ten):
    body = get_frames(client, ten, start=102, end=105).json()
    assert [f["frame"] for f in body["frames"]] == [102, 103, 104, 105]
    assert body["total"] == 4


def test_frames_start_only(client, ten):
    body = get_frames(client, ten, start=107).json()
    assert [f["frame"] for f in body["frames"]] == [107, 108, 109]


def test_frames_end_only(client, ten):
    body = get_frames(client, ten, end=101).json()
    assert [f["frame"] for f in body["frames"]] == [100, 101]


def test_frames_step(client, ten):
    body = get_frames(client, ten, step=3).json()
    assert [f["frame"] for f in body["frames"]] == [100, 103, 106, 109]
    assert body["total"] == 10


def test_frames_step_starts_at_first_selected(client, ten):
    body = get_frames(client, ten, start=101, end=108, step=4).json()
    assert [f["frame"] for f in body["frames"]] == [101, 105]
    assert body["total"] == 8


def test_frames_step_larger_than_selection(client, ten):
    body = get_frames(client, ten, start=104, end=106, step=50).json()
    assert [f["frame"] for f in body["frames"]] == [104]
    assert body["total"] == 3


def test_frames_with_gaps_select_by_frame_number(client):
    fr = [{"frame": n, "people": []} for n in (0, 5, 6, 20, 21)]
    mid = make(client, fr)["id"]
    body = get_frames(client, mid, start=5, end=20).json()
    assert [f["frame"] for f in body["frames"]] == [5, 6, 20]
    body = get_frames(client, mid, start=7, end=19).json()
    assert body == {"frames": [], "total": 0}


def test_frames_range_outside_recording_is_empty(client, ten):
    assert get_frames(client, ten, start=500, end=600).json() == {"frames": [], "total": 0}


def test_frames_start_equal_end(client, ten):
    body = get_frames(client, ten, start=103, end=103).json()
    assert [f["frame"] for f in body["frames"]] == [103]


@pytest.mark.parametrize("q", [
    {"start": "abc"}, {"end": "x"}, {"step": "two"}, {"start": "1.5"}, {"start": ""},
    {"start": -1}, {"end": -5}, {"step": -1}, {"step": 0},
    {"start": 105, "end": 104},
])
def test_frames_invalid_params_are_400(client, ten, q):
    r = get_frames(client, ten, **q)
    assert r.status_code == 400 and r.json()["error"]


def test_frames_unknown_match_404(client):
    r = get_frames(client, "m_0123456789")
    assert r.status_code == 404 and r.json()["error"]


# ---------------------------------------------------------------- analytics

def test_analytics_computed_once_and_cached(client, root, fake):
    m = make(client)
    r1 = client.get("/api/matches/%s/analytics" % m["id"])
    r2 = client.get("/api/matches/%s/analytics" % m["id"])
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == FAKE_SUMMARY and r2.json() == FAKE_SUMMARY
    assert len(fake) == 1
    assert json.loads((root / m["id"] / "analytics.json").read_text()) == FAKE_SUMMARY


def test_analytics_receives_parsed_frames_and_fps(client, fake):
    m = make(client)
    client.get("/api/matches/%s/analytics" % m["id"])
    assert fake[0]["frames"] == frames_basic()
    assert fake[0]["fps"] == 25


def test_analytics_not_computed_on_upload(client, fake):
    make(client)
    assert fake == []


def test_analytics_cache_is_shared_across_stores(client, root, fake):
    m = make(client)
    client.get("/api/matches/%s/analytics" % m["id"])
    app = FastAPI()
    app.include_router(matches.create_router(matches.MatchStore(root)))
    c2 = TestClient(app)
    assert c2.get("/api/matches/%s/analytics" % m["id"]).json() == FAKE_SUMMARY
    assert len(fake) == 1


def test_analytics_per_match(client, fake):
    a = make(client, seq(3))
    b = make(client, seq(4))
    client.get("/api/matches/%s/analytics" % a["id"])
    client.get("/api/matches/%s/analytics" % b["id"])
    assert [len(c["frames"]) for c in fake] == [3, 4]


def test_analytics_unknown_match_404(client, fake):
    r = client.get("/api/matches/m_0123456789/analytics")
    assert r.status_code == 404 and r.json()["error"]
    assert fake == []


def test_analytics_removed_with_match(client, root, fake):
    m = make(client)
    client.get("/api/matches/%s/analytics" % m["id"])
    client.delete("/api/matches/" + m["id"])
    assert not (root / m["id"]).exists()


def test_matches_module_has_no_heavy_imports():
    with open(matches.__file__, encoding="utf-8") as fh:
        text = fh.read()
    for pat in (r"^\s*(import|from)\s+(torch|cv2|numpy)\b", r"^\s*from\s+\.pipeline\b", r"^\s*from\s+sideline\.pipeline\b",
                r"^\s*import\s+sideline\.pipeline\b"):
        assert not re.search(pat, text, re.M)
    assert not re.search(r"^(import analytics|from \. import analytics|from \.analytics import)", text, re.M), \
        "analytics must be imported lazily inside the function that needs it"


# ---------------------------------------------------------------- csv

POSITIONS_CSV = (
    "frame,id,team,jersey,role,x,y\n"
    "10,7,left,10,player,1.5,-2.25\n"
    "11,7,left,10,player,2.0,-2.5\n"
    "11,12,right,,goalkeeper,-3,4\n"
)

PLAYERS_CSV = (
    "id,team,jersey,role,frames,seconds,distance_m,avg_speed_ms,max_speed_ms,sprints\n"
    "7,left,10,player,2,0.08,0.55,6.9,7.5,0\n"
    "9,,,referee,1,0.04,0.0,0.0,0.0,0\n"
)


def test_positions_csv_exact(client):
    m = make(client)
    r = client.get("/api/matches/%s/export.csv?kind=positions" % m["id"])
    assert r.status_code == 200
    assert r.text == POSITIONS_CSV


def test_csv_default_kind_is_positions(client):
    m = make(client)
    assert client.get("/api/matches/%s/export.csv" % m["id"]).text == POSITIONS_CSV


def test_csv_headers(client):
    m = make(client)
    r = client.get("/api/matches/%s/export.csv?kind=positions" % m["id"])
    assert r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == 'attachment; filename="%s-positions.csv"' % m["id"]


def test_positions_csv_skips_people_without_xy(client):
    fr = [{"frame": 0, "people": [{"id": 1, "role": "player", "team": None, "jersey": None, "xy": None},
                                  {"id": 2, "role": "player", "team": None, "jersey": None}]}]
    m = make(client, fr)
    r = client.get("/api/matches/%s/export.csv?kind=positions" % m["id"])
    assert r.text == "frame,id,team,jersey,role,x,y\n"


def test_positions_csv_does_not_use_analytics(client, fake):
    m = make(client)
    client.get("/api/matches/%s/export.csv?kind=positions" % m["id"])
    assert fake == []


def test_players_csv_exact(client, fake):
    m = make(client)
    r = client.get("/api/matches/%s/export.csv?kind=players" % m["id"])
    assert r.status_code == 200
    assert r.text == PLAYERS_CSV
    assert r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == 'attachment; filename="%s-players.csv"' % m["id"]


def test_players_csv_uses_cached_analytics(client, fake):
    m = make(client)
    client.get("/api/matches/%s/analytics" % m["id"])
    client.get("/api/matches/%s/export.csv?kind=players" % m["id"])
    client.get("/api/matches/%s/export.csv?kind=players" % m["id"])
    assert len(fake) == 1


@pytest.mark.parametrize("kind", ["", "teams", "Positions", "players.csv"])
def test_csv_unknown_kind_is_400(client, fake, kind):
    m = make(client)
    r = client.get("/api/matches/%s/export.csv" % m["id"], params={"kind": kind})
    assert r.status_code == 400 and r.json()["error"]


def test_csv_unknown_match_404(client, fake):
    for kind in ("positions", "players"):
        r = client.get("/api/matches/m_0123456789/export.csv?kind=" + kind)
        assert r.status_code == 404 and r.json()["error"]
