"""Acceptance tests for the formation feature (spec: features/formation/spec.md).

Recordings are built in code with the helpers below; every expected value is
hand-computable from the spec's rules (fps 25, window 60.0 s, line gap
10.0 m, pitch half-length 52.5 m).
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

import pytest
from fastapi.testclient import TestClient

from sideline import library

COLORS = {"left": "#d03030", "right": "#2050c0"}

# Worked recording of spec rule 14: static 4-4-2 (left, ids 1..10) and
# 4-3-3 (right, ids 21..30), one goalkeeper per side, one referee.
LEFT_442 = [(-50.0, 15.0), (-50.0, 5.0), (-50.0, -5.0), (-50.0, -15.0),
            (-30.0, 15.0), (-30.0, 5.0), (-30.0, -5.0), (-30.0, -15.0),
            (-10.0, 6.0), (-10.0, -6.0)]
RIGHT_433 = [(50.0, -15.0), (50.0, -5.0), (50.0, 5.0), (50.0, 15.0),
             (30.0, -15.0), (30.0, 0.0), (30.0, 15.0),
             (10.0, -10.0), (10.0, 0.0), (10.0, 10.0)]

# Window-average slots of the 1500-frame classic recording (spec rule 14):
# line 0 is the back line, seat 0 the team's own left side.
LEFT_SLOTS = [
    {"line": 0, "seat": 0, "x": -50.0, "y": 15.0, "n": 1500},
    {"line": 0, "seat": 1, "x": -50.0, "y": 5.0, "n": 1500},
    {"line": 0, "seat": 2, "x": -50.0, "y": -5.0, "n": 1500},
    {"line": 0, "seat": 3, "x": -50.0, "y": -15.0, "n": 1500},
    {"line": 1, "seat": 0, "x": -30.0, "y": 15.0, "n": 1500},
    {"line": 1, "seat": 1, "x": -30.0, "y": 5.0, "n": 1500},
    {"line": 1, "seat": 2, "x": -30.0, "y": -5.0, "n": 1500},
    {"line": 1, "seat": 3, "x": -30.0, "y": -15.0, "n": 1500},
    {"line": 2, "seat": 0, "x": -10.0, "y": 6.0, "n": 1500},
    {"line": 2, "seat": 1, "x": -10.0, "y": -6.0, "n": 1500},
]
RIGHT_SLOTS = [
    {"line": 0, "seat": 0, "x": 50.0, "y": -15.0, "n": 1500},
    {"line": 0, "seat": 1, "x": 50.0, "y": -5.0, "n": 1500},
    {"line": 0, "seat": 2, "x": 50.0, "y": 5.0, "n": 1500},
    {"line": 0, "seat": 3, "x": 50.0, "y": 15.0, "n": 1500},
    {"line": 1, "seat": 0, "x": 30.0, "y": -15.0, "n": 1500},
    {"line": 1, "seat": 1, "x": 30.0, "y": 0.0, "n": 1500},
    {"line": 1, "seat": 2, "x": 30.0, "y": 15.0, "n": 1500},
    {"line": 2, "seat": 0, "x": 10.0, "y": -10.0, "n": 1500},
    {"line": 2, "seat": 1, "x": 10.0, "y": 0.0, "n": 1500},
    {"line": 2, "seat": 2, "x": 10.0, "y": 10.0, "n": 1500},
]


def person(pid, role, team, xy, jersey=None):
    return {"id": pid, "role": role, "team": team,
            "jersey": str(jersey) if jersey is not None else None,
            "box": [0, 0, 20, 40], "xy": xy}


def frame_line(n, people, ball=None):
    return json.dumps({"frame": n, "delay": 0, "people": people,
                       "ball": ball, "colors": COLORS})


def recording(people_fn, frames, ball=None):
    """frames.jsonl bytes: one line per frame number in frames."""
    lines = [frame_line(f, people_fn(f), ball(f) if callable(ball) else ball)
             for f in frames]
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse(data):
    return [json.loads(line) for line in data.decode("utf-8").split("\n")
            if line.strip()]


def classic_people(frame):
    people = [person(pid, "player", "left", [x, y], pid)
              for pid, (x, y) in enumerate(LEFT_442, start=1)]
    people += [person(pid, "player", "right", [x, y], pid)
               for pid, (x, y) in enumerate(RIGHT_433, start=21)]
    people.append(person(11, "goalkeeper", "left", [-49.0, 0.0], 1))
    people.append(person(12, "goalkeeper", "right", [49.0, 0.0], 1))
    people.append(person(40, "referee", None, [0.0, 0.0]))
    return people


def make_classic(n=1500, ball=True):
    ball = (lambda f: {"box": [0, 0, 12, 12], "xy": [0.05 * f - 12.5, 0.0]}) if ball else None
    return recording(classic_people, range(n), ball=ball)


@pytest.fixture()
def client(tmp_path):
    return TestClient(library.create_library_app(tmp_path / "matches"))


def upload(c, data, name="game"):
    r = c.post("/api/matches",
               files={"file": (name + ".jsonl", data, "application/json")},
               data={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --- the classic recording --------------------------------------------------

def test_classic_route_shape(client):
    mid = upload(client, make_classic())
    r = client.get("/api/matches/%s/formation" % mid)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"match_id", "window_s", "windows"}
    assert body["match_id"] == mid
    assert body["window_s"] == 60.0
    assert len(body["windows"]) == 1
    w = body["windows"][0]
    assert set(w) == {"index", "start_frame", "end_frame", "duration_s", "left", "right"}
    assert (w["index"], w["start_frame"], w["end_frame"], w["duration_s"]) == (0, 0, 1499, 60.0)


def test_classic_left_team(client):
    mid = upload(client, make_classic())
    left = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]["left"]
    assert set(left) == {"formation", "confidence", "players", "slots"}
    assert left["formation"] == "4-4-2"
    assert left["confidence"] == 1.0
    assert left["players"] == 10
    assert left["slots"] == LEFT_SLOTS


def test_classic_right_team(client):
    mid = upload(client, make_classic())
    right = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]["right"]
    assert right["formation"] == "4-3-3"
    assert right["confidence"] == 1.0
    assert right["players"] == 10
    assert right["slots"] == RIGHT_SLOTS


def test_window_s_30_two_windows(client):
    mid = upload(client, make_classic())
    body = client.get("/api/matches/%s/formation?window_s=30" % mid).json()
    assert body["window_s"] == 30.0
    assert [(w["index"], w["start_frame"], w["end_frame"], w["duration_s"])
            for w in body["windows"]] == [(0, 0, 749, 30.0), (1, 750, 1499, 30.0)]
    assert [w["left"]["formation"] for w in body["windows"]] == ["4-4-2", "4-4-2"]


def test_window_s_5_twelve_windows(client):
    mid = upload(client, make_classic())
    body = client.get("/api/matches/%s/formation?window_s=5" % mid).json()
    assert [(w["start_frame"], w["end_frame"]) for w in body["windows"]] == [
        (0, 124), (125, 249), (250, 374), (375, 499), (500, 624), (625, 749),
        (750, 874), (875, 999), (1000, 1124), (1125, 1249), (1250, 1374), (1375, 1499)]


def test_window_s_600_one_window(client):
    mid = upload(client, make_classic())
    body = client.get("/api/matches/%s/formation?window_s=600" % mid).json()
    assert len(body["windows"]) == 1
    assert (body["windows"][0]["start_frame"], body["windows"][0]["end_frame"],
            body["windows"][0]["duration_s"]) == (0, 1499, 60.0)


def test_partial_last_window(client):
    mid = upload(client, make_classic(1750))
    body = client.get("/api/matches/%s/formation" % mid).json()
    assert [(w["start_frame"], w["end_frame"], w["duration_s"])
            for w in body["windows"]] == [(0, 1499, 60.0), (1500, 1749, 10.0)]
    assert body["windows"][1]["left"]["formation"] == "4-4-2"
    assert body["windows"][1]["left"]["slots"][0]["n"] == 250


# --- edge cases -------------------------------------------------------------

def test_single_frame(client):
    mid = upload(client, recording(classic_people, [100]))
    w = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]
    assert (w["start_frame"], w["end_frame"], w["duration_s"]) == (100, 100, 0.04)
    assert w["left"]["formation"] == "4-4-2" and w["left"]["slots"][0]["n"] == 1
    assert w["right"]["formation"] == "4-3-3" and w["right"]["slots"][0]["n"] == 1


def test_team_absent_is_null(client):
    left_only = [person(pid, "player", "left", [x, y])
                 for pid, (x, y) in enumerate(LEFT_442, start=1)]
    mid = upload(client, recording(lambda f: left_only, [7]))
    w = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]
    assert w["left"]["formation"] == "4-4-2"
    assert w["right"] == {"formation": None, "confidence": 0.0, "players": 0, "slots": []}


def test_team_positions_all_null_is_null(client):
    def people(frame):
        p = classic_people(frame)
        for q in p:
            if q["team"] == "right":
                q["xy"] = None
        return p
    mid = upload(client, recording(people, range(1500)))
    w = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]
    assert w["right"] == {"formation": None, "confidence": 0.0, "players": 0, "slots": []}
    assert w["left"]["formation"] == "4-4-2" and w["left"]["confidence"] == 1.0


def test_two_players_null_three_players_split(client):
    pts = [(-10.0, 0.0), (10.0, 0.0), (20.0, 0.0)]

    def people(count):
        return lambda frame: [person(i + 1, "player", "left", list(pts[i]))
                              for i in range(count)]
    m2 = upload(client, recording(people(2), range(5)), name="two")
    w = client.get("/api/matches/%s/formation" % m2).json()["windows"][0]
    assert w["left"] == {"formation": None, "confidence": 0.0, "players": 2, "slots": []}
    m3 = upload(client, recording(people(3), range(5)), name="three")
    w = client.get("/api/matches/%s/formation" % m3).json()["windows"][0]
    # d = 42.5, 62.5, 72.5: gaps 20.0 and 10.0 (>= 10.0) -> three single lines
    assert w["left"]["formation"] == "1-1-1"
    assert w["left"]["confidence"] == 1.0
    assert w["left"]["slots"] == [
        {"line": 0, "seat": 0, "x": -10.0, "y": 0.0, "n": 5},
        {"line": 1, "seat": 0, "x": 10.0, "y": 0.0, "n": 5},
        {"line": 2, "seat": 0, "x": 20.0, "y": 0.0, "n": 5},
    ]


def test_frame_gaps_count_only_recorded_frames(client):
    frames = list(range(1000)) + list(range(2000, 2500))
    mid = upload(client, recording(classic_people, frames))
    ws = client.get("/api/matches/%s/formation" % mid).json()["windows"]
    assert [(w["start_frame"], w["end_frame"], w["duration_s"]) for w in ws] == \
        [(0, 1499, 60.0), (1500, 2499, 40.0)]
    for w in ws:
        assert w["left"]["confidence"] == 1.0
    assert ws[0]["left"]["slots"][0]["n"] == 1000
    assert ws[1]["left"]["slots"][0]["n"] == 500


def test_missing_xy_drops_player(client):
    def people(frame):
        p = classic_people(frame)
        for q in p:
            if q["id"] == 9:
                q["xy"] = None
        return p
    mid = upload(client, recording(people, range(1500)))
    w = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]
    assert w["left"]["formation"] == "4-4-1"
    assert w["left"]["players"] == 9
    assert w["left"]["confidence"] == 1.0
    assert len(w["left"]["slots"]) == 9


def test_duplicate_id_first_wins(client):
    def people(frame):
        p = classic_people(frame)
        p.append(person(9, "player", "left", [99.0, 99.0], 9))  # bogus, after the real one
        return p
    mid = upload(client, recording(people, range(1500)))
    w = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]
    assert w["left"]["formation"] == "4-4-2"
    assert w["left"]["players"] == 10
    assert w["left"]["slots"] == LEFT_SLOTS


def test_referee_with_team_excluded(client):
    def people(frame):
        p = classic_people(frame)
        p.append(person(41, "referee", "left", [-40.0, 0.0]))
        return p
    mid = upload(client, recording(people, range(5)))
    w = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]
    assert w["left"]["formation"] == "4-4-2"
    assert w["left"]["players"] == 10


def test_ball_absent_no_effect(client):
    mid = upload(client, make_classic(ball=False))
    w = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]
    assert w["left"]["formation"] == "4-4-2"
    assert w["right"]["formation"] == "4-3-3"


# --- the line/seat/confidence rules ------------------------------------------

def test_gap_boundary_starts_new_line(client):
    # d = 10.0, 20.0, 30.5, 32.0 (x = d - 52.5): gaps 10.0 (new line,
    # exactly the threshold), 10.5 (new), 1.5 (same line).
    xy = [(-42.5, 0.0), (-32.5, 0.0), (-22.0, 5.0), (-20.5, -5.0)]

    def people(frame):
        return [person(i + 1, "player", "left", list(xy[i])) for i in range(4)]
    mid = upload(client, recording(people, range(5)))
    w = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]
    assert w["left"]["formation"] == "1-1-2"
    assert w["left"]["players"] == 4
    assert w["left"]["confidence"] == 0.85  # last line spans 1.5 m: 1 - 1.5/10
    assert w["left"]["slots"] == [
        {"line": 0, "seat": 0, "x": -42.5, "y": 0.0, "n": 5},
        {"line": 1, "seat": 0, "x": -32.5, "y": 0.0, "n": 5},
        {"line": 2, "seat": 0, "x": -22.0, "y": 5.0, "n": 5},
        {"line": 2, "seat": 1, "x": -20.5, "y": -5.0, "n": 5},
    ]


def test_confidence_coverage(client):
    def people(frame):
        p = classic_people(frame)
        if frame >= 1200:
            for q in p:
                if q["team"] == "left":
                    q["xy"] = None
        return p
    mid = upload(client, recording(people, range(1500)))
    w = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]
    assert w["left"]["formation"] == "4-4-2"
    assert w["left"]["confidence"] == 0.8  # 1200 of 1500 recorded frames
    assert w["left"]["players"] == 10
    assert w["left"]["slots"][0]["n"] == 1200
    assert w["right"]["confidence"] == 1.0


def test_confidence_tightness(client):
    # d = 10.0, 15.0 (same line, spread 5.0 -> factor 0.5), 30.0 (gap 15.0 -> new line)
    xy = [(-42.5, 0.0), (-37.5, 0.0), (-22.5, 0.0)]

    def people(frame):
        return [person(i + 1, "player", "left", list(xy[i])) for i in range(3)]
    mid = upload(client, recording(people, range(5)))
    w = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]
    assert w["left"]["formation"] == "2-1"
    assert w["left"]["confidence"] == 0.5
    assert w["left"]["slots"] == [
        {"line": 0, "seat": 0, "x": -42.5, "y": 0.0, "n": 5},
        {"line": 0, "seat": 1, "x": -37.5, "y": 0.0, "n": 5},
        {"line": 1, "seat": 0, "x": -22.5, "y": 0.0, "n": 5},
    ]


def test_positions_rounded_to_two_decimals(client):
    def people(frame):
        p = classic_people(frame)
        for q in p:
            if q["id"] == 10:
                q["xy"] = [-10.006, -6.126]
        return p
    mid = upload(client, recording(people, range(1500)))
    w = client.get("/api/matches/%s/formation" % mid).json()["windows"][0]
    assert w["left"]["formation"] == "4-4-2"
    assert w["left"]["confidence"] == 1.0
    assert [s for s in w["left"]["slots"] if s["line"] == 2] == [
        {"line": 2, "seat": 0, "x": -10.0, "y": 6.0, "n": 1500},
        {"line": 2, "seat": 1, "x": -10.01, "y": -6.13, "n": 1500},
    ]


def test_window_s_fractional_12_5(client):
    # W = int(round(12.5 * 25)) = int(round(312.5)) = 312 (halves to even)
    mid = upload(client, make_classic())
    body = client.get("/api/matches/%s/formation?window_s=12.5" % mid).json()
    assert body["window_s"] == 12.5
    assert [(w["start_frame"], w["end_frame"], w["duration_s"])
            for w in body["windows"]] == \
        [(0, 311, 12.48), (312, 623, 12.48), (624, 935, 12.48),
         (936, 1247, 12.48), (1248, 1499, 10.08)]


# --- input validation ---------------------------------------------------------

@pytest.mark.parametrize("bad", ["abc", "0", "4.9", "601", "-3", "inf"])
def test_window_s_invalid_is_400(client, bad):
    mid = upload(client, make_classic(50))
    r = client.get("/api/matches/%s/formation?window_s=%s" % (mid, bad))
    assert r.status_code == 400


def test_window_s_error_message_exact(client):
    mid = upload(client, make_classic(50))
    r = client.get("/api/matches/%s/formation?window_s=abc" % mid)
    assert r.status_code == 400
    assert r.json() == {"error": "window_s must be a number between 5 and 600"}


@pytest.mark.parametrize("bad", [
    "m_" + "0" * 10, "m_x", "m_0123456789a", "m_012345678",
    "m_ABCDEF0123", "x_0123456789", "M_0123456789",
    "m_0123456789%2F..%2F..", "%2e%2e",
])
def test_unknown_and_malformed_ids_404(client, bad):
    r = client.get("/api/matches/%s/formation" % bad)
    assert r.status_code == 404
    assert "error" in r.json()


def test_404_takes_precedence_over_400(client):
    r = client.get("/api/matches/m_" + "0" * 10 + "/formation?window_s=abc")
    assert r.status_code == 404
    assert "error" in r.json()


def test_empty_upload_rejected_400(client):
    r = client.post("/api/matches",
                    files={"file": ("empty.jsonl", b"", "application/json")},
                    data={"name": "empty"})
    assert r.status_code == 400
    assert "error" in r.json()


# --- idempotence and caching ---------------------------------------------------

def test_idempotent_repeated_calls(client):
    mid = upload(client, make_classic())
    path = "/api/matches/%s/formation?window_s=30" % mid
    assert client.get(path).status_code == 200
    assert client.get(path).json() == client.get(path).json()


def test_cached_json_written_next_to_match(tmp_path):
    root = tmp_path / "matches"
    c = TestClient(library.create_library_app(root))
    mid = upload(c, make_classic())
    r = c.get("/api/matches/%s/formation" % mid)
    assert r.status_code == 200
    cached = root / mid / "features" / "formation-w60.0.json"
    assert cached.is_file()
    assert json.loads(cached.read_text(encoding="utf-8")) == \
        {k: v for k, v in r.json().items() if k != "match_id"}
    c.get("/api/matches/%s/formation?window_s=30" % mid)
    assert (root / mid / "features" / "formation-w30.0.json").is_file()


# --- the pure function -----------------------------------------------------------

def test_pure_function_matches_the_route(client):
    from sideline.features.formation import build_formation
    mid = upload(client, make_classic())
    api = client.get("/api/matches/%s/formation?window_s=30" % mid).json()
    pure = build_formation(parse(make_classic()), window_s=30)
    assert pure == {k: v for k, v in api.items() if k != "match_id"}


def test_pure_function_does_not_mutate_input():
    from sideline.features.formation import build_formation
    frames = parse(make_classic(100))
    snapshot = copy.deepcopy(frames)
    build_formation(frames, window_s=30)
    assert frames == snapshot


def test_pure_function_empty_frames():
    from sideline.features.formation import build_formation
    assert build_formation([]) == {"window_s": 60.0, "windows": []}


# --- registry integration and performance -----------------------------------------

def test_api_features_lists_formation(client):
    r = client.get("/api/features")
    assert r.status_code == 200
    entry = next(e for e in r.json()["features"] if e["name"] == "formation")
    assert entry["ok"] is True
    assert entry["error"] is None
    assert "/formation" in entry["routes"]


def test_performance_25000_frames_under_budget(client):
    people = classic_people(0)
    tail = json.dumps({"delay": 0, "people": people, "ball": None, "colors": COLORS})
    inner = tail[1:-1]
    big = ("\n".join('{"frame": %d, %s}' % (f, inner) for f in range(25000)) + "\n").encode()
    assert len(big) > 50 * 1024 * 1024  # a genuine "huge input"
    mid = upload(client, big, name="big")
    start = time.time()
    r = client.get("/api/matches/%s/formation" % mid)
    elapsed = time.time() - start
    assert r.status_code == 200
    assert elapsed < 10.0, "cold GET took %.2fs" % elapsed
    body = r.json()
    assert len(body["windows"]) == 17
    assert (body["windows"][-1]["start_frame"], body["windows"][-1]["end_frame"],
            body["windows"][-1]["duration_s"]) == (24000, 24999, 40.0)
    assert body["windows"][0]["left"]["formation"] == "4-4-2"
    assert body["windows"][0]["right"]["formation"] == "4-3-3"
    start = time.time()
    again = client.get("/api/matches/%s/formation" % mid)
    assert time.time() - start < 3.0  # the cached payload
    assert again.json() == body
