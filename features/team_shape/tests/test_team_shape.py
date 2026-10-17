"""Acceptance tests for the team_shape feature (spec: features/team_shape/spec.md).

Recordings are built in code by the small helpers below; every expected value
is derived by hand from the spec.
"""
import copy
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

import pytest
from fastapi.testclient import TestClient

from sideline import library
from sideline.features import team_shape

# Hand-computed metric sets (spec worked examples), rounded to 2 decimals as
# the route must return them.
SQUARE_M = {"centroid": [5.0, 0.0], "width": 10.0, "length": 10.0,
            "hull_area": 100.0, "defensive_line": 57.5, "compactness": 7.07}
DIAMOND_M = {"centroid": [25.0, 0.0], "width": 16.0, "length": 10.0,
             "hull_area": 80.0, "defensive_line": 27.5, "compactness": 6.5}

SQUARE_XY = [(0, -5), (0, 5), (10, -5), (10, 5)]
DIAMOND_XY = [(20, 0), (30, 0), (25, 8), (25, -8)]


def person(i, x, y, team="left", role="player", id_=None, xy="__auto__"):
    return {"id": i if id_ is None else id_, "role": role, "team": team,
            "jersey": str(i), "box": [0, 0, 10, 20],
            "xy": [x, y] if xy == "__auto__" else xy}


def frames_of(numbers, people, ball=None):
    return [{"frame": f, "delay": 0, "people": [dict(p) for p in people],
             "ball": ball, "colors": None} for f in numbers]


def to_bytes(frames):
    return ("\n".join(json.dumps(f) for f in frames) + "\n").encode()


def make_client(tmp_path):
    return TestClient(library.create_library_app(tmp_path / "store"))


def upload(client, data, name="game"):
    r = client.post("/api/matches",
                    files={"file": (name + ".jsonl", data, "application/json")},
                    data={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def get_shape(client, mid, team=None):
    url = "/api/matches/%s/team-shape" % mid
    return client.get(url, params={"team": team} if team is not None else None)


def basic_people():
    people = [person(i, x, y) for i, (x, y) in
              enumerate(SQUARE_XY, start=1)]
    people += [person(i, x, y, "right") for i, (x, y) in
               enumerate(DIAMOND_XY, start=11)]
    return people


def basic_recording(ball=False):
    ball_obj = {"box": [5, 5, 6, 6], "xy": [0.0, 0.0]} if ball else None
    return frames_of(range(50), basic_people(), ball=ball_obj)


def moving_recording():
    # player 1 at x=0 in frames 0-19 (20 frames), x=10 in frames 20-24 (5)
    frames = []
    for f in range(25):
        x = 0.0 if f < 20 else 10.0
        frames.append({"frame": f, "delay": 0, "ball": None, "colors": None,
                       "people": [person(1, x, 0), person(2, 0, 0),
                                  person(3, 0, 10), person(4, 30, 10)]})
    return frames


def five_left_recording():
    left = [person(1, 0, 0), person(2, 10, 0), person(3, 12, 4),
            person(4, 14, -4), person(5, 16, 0)]
    right = [person(11, 20, 0, "right"), person(12, 30, 0, "right"),
             person(13, 25, 8, "right")]
    return frames_of(range(25), left + right)


def gap_recording():
    return frames_of(list(range(25)) + list(range(50, 75)),
                     [person(i, x, y) for i, (x, y) in enumerate(SQUARE_XY, start=1)])


def sparse_recording():
    people = [person(1, 0, 0), person(1, 100, 100, id_=1), person(2, 10, 0),
              person(3, 0, 10), person(4, 10, 10),
              person(5, 50, 50, xy=None), person(6, 60, 60, xy=None)]
    return frames_of([0], people)


def one_frame_recording():
    return frames_of([7], basic_people())


def keeper_ref_recording():
    people = [person(i, x, y) for i, (x, y) in enumerate(SQUARE_XY, start=1)]
    people += [person(12, -50, 0, role="goalkeeper"),
               person(99, 40, 20, team=None, role="referee")]
    return frames_of([0], people)


def collinear_recording():
    return frames_of([0], [person(1, 0, 0), person(2, 1, 0),
                           person(3, 2, 0), person(4, 3, 0)])


def interior_recording():
    return frames_of([0], [person(i, x, y) for i, (x, y) in enumerate(SQUARE_XY, start=1)]
                     + [person(9, 5, 5)])


def right_five_recording():
    people = [person(21, 10, 0, "right"), person(22, 12, 3, "right"),
              person(23, 14, -3, "right"), person(24, 16, 0, "right"),
              person(25, 40, 2, "right")]
    return frames_of([0], people)


def avg_recording():
    # second 0: the square at x 0..10; second 1: shifted +10
    sq = [person(1, 0, -5), person(2, 0, 5), person(3, 10, -5), person(4, 10, 5)]
    sq2 = [person(1, 10, -5), person(2, 10, 5), person(3, 20, -5), person(4, 20, 5)]
    frames = []
    for f in range(50):
        frames.append({"frame": f, "delay": 0, "ball": None, "colors": None,
                       "people": [dict(p) for p in (sq if f < 25 else sq2)]})
    return frames


def big_recording(n):
    frames = []
    for f in range(n):
        people = []
        for i in range(10):
            people.append(person(
                i + 1,
                round(-30 + 3 * i + 0.01 * math.sin(f / 50.0 + i), 2),
                round(-20 + 4 * i + 0.01 * math.cos(f / 40.0 + i), 2)))
        people.append(person(12, -50.0, 0.0, role="goalkeeper"))
        for i in range(10):
            people.append(person(
                23 + i,
                round(10 + 3 * i + 0.01 * math.sin(f / 60.0 + i), 2),
                round(-20 + 4 * i, 2), team="right"))
        people.append(person(34, 50.0, 0.0, team="right", role="goalkeeper"))
        frames.append({"frame": f, "delay": 0, "people": people,
                       "ball": None, "colors": None})
    return frames


# --- route: the basic recording -------------------------------------------

def test_basic_route_document(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    r = get_shape(client, mid)
    assert r.status_code == 200
    doc = r.json()
    assert doc["match_id"] == mid
    assert doc["fps"] == 25
    assert doc["first_frame"] == 0 and doc["last_frame"] == 49
    assert set(doc["teams"]) == {"left", "right"}
    left, right = doc["teams"]["left"], doc["teams"]["right"]
    assert [s["t"] for s in left["seconds"]] == [0, 1]
    assert [s["t"] for s in right["seconds"]] == [0, 1]
    assert left["seconds"][0] == {"t": 0, **SQUARE_M}
    assert left["seconds"][1] == {"t": 1, **SQUARE_M}
    assert right["seconds"][0] == {"t": 0, **DIAMOND_M}
    assert right["seconds"][1] == {"t": 1, **DIAMOND_M}


def test_basic_route_match_averages(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    doc = get_shape(client, mid).json()
    # both seconds are identical, so the average equals one sample's metrics
    assert doc["teams"]["left"]["average"] == SQUARE_M
    assert doc["teams"]["right"]["average"] == DIAMOND_M


def test_team_query_filters_to_one_team(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    r = get_shape(client, mid, team="left")
    assert r.status_code == 200
    doc = r.json()
    assert list(doc["teams"]) == ["left"]
    assert doc["teams"]["left"]["seconds"][0] == {"t": 0, **SQUARE_M}
    r = get_shape(client, mid, team="right")
    assert list(r.json()["teams"]) == ["right"]
    assert r.json()["teams"]["right"]["seconds"][0] == {"t": 0, **DIAMOND_M}


def test_team_query_invalid_is_400(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    for bad in ("middle", "LEFT", "Left", ""):
        r = get_shape(client, mid, team=bad)
        assert r.status_code == 400, bad
        assert r.json() == {"detail": {"error": "team must be 'left' or 'right'"}}


def test_unknown_match_is_404(tmp_path):
    client = make_client(tmp_path)
    r = client.get("/api/matches/m_0000000000/team-shape")
    assert r.status_code == 404
    assert r.json() == {"detail": {"error": "match not found"}}


@pytest.mark.parametrize("bad_id", ["m_short", "m_123456789a1", "nope", "m_"])
def test_malformed_match_id_is_404(tmp_path, bad_id):
    client = make_client(tmp_path)
    r = client.get("/api/matches/%s/team-shape" % bad_id)
    assert r.status_code == 404
    assert r.json() == {"detail": {"error": "match not found"}}


def test_bad_team_on_unknown_match_is_404(tmp_path):
    client = make_client(tmp_path)
    r = get_shape(client, "m_0000000000", team="bogus")
    assert r.status_code == 404  # the match is resolved before the parameter
    assert r.json() == {"detail": {"error": "match not found"}}


def test_repeated_gets_are_idempotent(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    r1, r2 = get_shape(client, mid), get_shape(client, mid)
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    # pinned to the spec's numbers, not just to the first response
    assert r1.json()["teams"]["left"]["seconds"][0] == {"t": 0, **SQUARE_M}
    assert r2.json()["teams"]["right"]["average"] == DIAMOND_M


def test_cache_file_written_and_matches_response(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    doc = get_shape(client, mid).json()
    cache = tmp_path / "store" / mid / "features" / "team-shape.json"
    assert cache.is_file()
    assert json.loads(cache.read_text(encoding="utf-8")) == doc
    # a filtered read must not replace the cache with a filtered document
    assert get_shape(client, mid, team="left").status_code == 200
    full = json.loads(cache.read_text(encoding="utf-8"))
    assert set(full["teams"]) == {"left", "right"}


def test_cache_recomputed_when_frames_replaced(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    assert get_shape(client, mid).json()["last_frame"] == 49
    # the match is "replaced": 100 frames, same 8 people
    path = tmp_path / "store" / mid / "frames.jsonl"
    time.sleep(0.02)
    path.write_bytes(to_bytes(frames_of(range(100), basic_people())))
    time.sleep(0.02)
    again = make_client(tmp_path)  # fresh app: no in-memory frame LRU
    doc = get_shape(again, mid).json()
    assert doc["last_frame"] == 99
    assert [s["t"] for s in doc["teams"]["left"]["seconds"]] == [0, 1, 2, 3]
    assert doc["teams"]["left"]["seconds"][0] == {"t": 0, **SQUARE_M}


def test_route_does_not_mutate_the_frames_file(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    path = tmp_path / "store" / mid / "frames.jsonl"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    get_shape(client, mid)
    get_shape(client, mid, team="left")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_feature_is_listed_in_api_features(tmp_path):
    client = make_client(tmp_path)
    r = client.get("/api/features")
    assert r.status_code == 200
    feats = r.json()["features"]
    assert "team_shape" in [f["name"] for f in feats]
    entry = [f for f in feats if f["name"] == "team_shape"][0]
    assert entry["ok"] is True and entry["error"] is None
    assert any("team-shape" in str(route) for route in entry["routes"])


# --- route: the edge-case recordings ---------------------------------------

def test_moving_player_uses_second_average(tmp_path):
    rec = moving_recording()
    xs = {f["frame"]: f["people"][0]["xy"][0] for f in rec}
    assert xs[0] == 0.0 and xs[19] == 0.0 and xs[20] == 10.0 and xs[24] == 10.0
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    doc = get_shape(client, mid).json()
    # player 1's second position is (20*0 + 5*10)/25 = 2.0; with (0,0), (0,10),
    # (30,10) that gives centroid [8, 5], width 10, length 30, hull 160
    # (shoelace over (0,0), (2,0), (30,10), (0,10)), defensive_line
    # |mean(0,0,2,30) + 52.5| = 60.5, compactness (sqrt(61) + 2*sqrt(89) +
    # sqrt(509))/4 = 12.31
    assert doc["teams"]["left"]["seconds"] == [
        {"t": 0, "centroid": [8.0, 5.0], "width": 10.0, "length": 30.0,
         "hull_area": 160.0, "defensive_line": 60.5, "compactness": 12.31}]
    assert doc["teams"]["left"]["average"]["defensive_line"] == 60.5
    assert doc["teams"]["right"]["seconds"] == []
    assert doc["teams"]["right"]["average"] is None


def test_five_left_right_below_minimum(tmp_path):
    rec = five_left_recording()
    assert len(rec[0]["people"]) == 8
    assert sum(1 for p in rec[0]["people"] if p["team"] == "right") == 3
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    doc = get_shape(client, mid).json()
    # 5 left players: the forward at x=16 is excluded from the 4 deepest
    # (mean 9 -> 61.5); hull of (0,0), (14,-4), (16,0), (12,4) is 64
    assert doc["teams"]["left"]["seconds"] == [
        {"t": 0, "centroid": [10.4, 0.0], "width": 8.0, "length": 16.0,
         "hull_area": 64.0, "defensive_line": 61.5, "compactness": 5.22}]
    assert doc["teams"]["right"]["seconds"] == []
    assert doc["teams"]["right"]["average"] is None


def test_frame_gaps_skip_missing_seconds(tmp_path):
    rec = gap_recording()
    numbers = [f["frame"] for f in rec]
    assert 25 not in numbers and 49 not in numbers and 74 in numbers
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    doc = get_shape(client, mid).json()
    assert doc["last_frame"] == 74
    left = doc["teams"]["left"]["seconds"]
    assert [s["t"] for s in left] == [0, 2]
    assert left[0] == {"t": 0, **SQUARE_M}
    assert left[1] == {"t": 2, **SQUARE_M}


def test_sparse_frame_duplicate_id_and_missing_positions(tmp_path):
    rec = sparse_recording()
    ids = [p["id"] for p in rec[0]["people"]]
    assert ids.count(1) == 2  # id 1 really is duplicated
    assert sum(1 for p in rec[0]["people"] if p["xy"] is None) == 2
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    doc = get_shape(client, mid).json()
    # first entry of id 1 wins: (0,0), not (100,100); the two null xy people
    # are dropped, leaving exactly the 10x10 square
    assert doc["teams"]["left"]["seconds"] == [
        {"t": 0, "centroid": [5.0, 5.0], "width": 10.0, "length": 10.0,
         "hull_area": 100.0, "defensive_line": 57.5, "compactness": 7.07}]
    assert doc["teams"]["right"]["seconds"] == []


def test_single_frame_recording(tmp_path):
    rec = one_frame_recording()
    assert len(rec) == 1 and rec[0]["frame"] == 7
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    doc = get_shape(client, mid).json()
    assert doc["first_frame"] == 7 and doc["last_frame"] == 7
    assert doc["teams"]["left"]["seconds"] == [{"t": 0, **SQUARE_M}]
    assert doc["teams"]["right"]["seconds"] == [{"t": 0, **DIAMOND_M}]


def test_goalkeeper_and_referee_are_excluded(tmp_path):
    rec = keeper_ref_recording()
    keepers = [p for p in rec[0]["people"] if p["role"] == "goalkeeper"]
    assert keepers and keepers[0]["xy"] == [-50, 0] and keepers[0]["team"] == "left"
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    doc = get_shape(client, mid).json()
    # the keeper at x=-50 would pull defensive_line to 45.0 and the centroid
    # to [-6, 0] if it (or the referee) were counted
    assert doc["teams"]["left"]["seconds"][0]["defensive_line"] == 57.5
    assert doc["teams"]["left"]["seconds"][0]["centroid"] == [5.0, 0.0]
    assert doc["teams"]["right"]["seconds"] == []


def test_collinear_hull_is_zero(tmp_path):
    rec = collinear_recording()
    assert all(p["xy"][1] == 0 for p in rec[0]["people"])
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    doc = get_shape(client, mid).json()
    # x = 0,1,2,3: centroid [1.5, 0], width 0, length 3, hull 0.0,
    # defensive_line |1.5 + 52.5| = 54.0, compactness (1.5+0.5+0.5+1.5)/4 = 1.0
    assert doc["teams"]["left"]["seconds"] == [
        {"t": 0, "centroid": [1.5, 0.0], "width": 0.0, "length": 3.0,
         "hull_area": 0.0, "defensive_line": 54.0, "compactness": 1.0}]


def test_interior_point_does_not_change_hull(tmp_path):
    rec = interior_recording()
    assert len(rec[0]["people"]) == 5
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    doc = get_shape(client, mid).json()
    # 5th player at (5,5), inside the square: hull stays 100.0, centroid
    # moves to [5, 1], deepest 4 have mean x (0+0+5+10)/4 = 3.75 -> 56.25,
    # compactness (2*sqrt(61) + 2*sqrt(41) + 4)/5 = 6.49
    assert doc["teams"]["left"]["seconds"] == [
        {"t": 0, "centroid": [5.0, 1.0], "width": 10.0, "length": 10.0,
         "hull_area": 100.0, "defensive_line": 56.25, "compactness": 6.49}]


def test_right_team_deepest_four(tmp_path):
    rec = right_five_recording()
    assert sum(1 for p in rec[0]["people"] if p["team"] == "right") == 5
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    doc = get_shape(client, mid).json()
    # 4 deepest right players are x = 40,16,14,12 -> mean 20.5 -> 32.0
    assert doc["teams"]["right"]["seconds"] == [
        {"t": 0, "centroid": [18.4, 0.4], "width": 6.0, "length": 30.0,
         "hull_area": 92.0, "defensive_line": 32.0, "compactness": 8.99}]
    assert doc["teams"]["left"]["seconds"] == []
    assert doc["teams"]["left"]["average"] is None


def test_ball_is_ignored(tmp_path):
    with_ball = basic_recording(ball=True)
    without_ball = basic_recording(ball=False)
    assert all(f["ball"] is not None for f in with_ball)
    assert all(f["ball"] is None for f in without_ball)
    client = make_client(tmp_path)
    mid1 = upload(client, to_bytes(with_ball))
    mid2 = upload(client, to_bytes(without_ball))
    d1, d2 = get_shape(client, mid1).json(), get_shape(client, mid2).json()
    d1.pop("match_id")
    d2.pop("match_id")
    assert d1 == d2


def test_average_across_differing_seconds(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(avg_recording()))
    doc = get_shape(client, mid).json()
    left = doc["teams"]["left"]
    assert left["seconds"][0] == {"t": 0, **SQUARE_M}
    assert left["seconds"][1] == {"t": 1, "centroid": [15.0, 0.0], "width": 10.0,
                                  "length": 10.0, "hull_area": 100.0,
                                  "defensive_line": 67.5, "compactness": 7.07}
    assert left["average"] == {"centroid": [10.0, 0.0], "width": 10.0,
                               "length": 10.0, "hull_area": 100.0,
                               "defensive_line": 62.5, "compactness": 7.07}


def test_performance_25000_frames_within_budget(tmp_path):
    rec = big_recording(25000)
    assert len(rec) == 25000
    assert all(len(f["people"]) == 22 for f in rec)
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    t0 = time.monotonic()
    r = get_shape(client, mid)
    elapsed = time.monotonic() - t0
    assert r.status_code == 200
    assert elapsed < 30.0, "team-shape took %.1fs, budget is 30s" % elapsed
    left = r.json()["teams"]["left"]["seconds"]
    right = r.json()["teams"]["right"]["seconds"]
    assert len(left) == 1000 and len(right) == 1000
    assert left[0]["t"] == 0 and left[-1]["t"] == 999


# --- pure functions ---------------------------------------------------------

def test_shape_match_empty_frames():
    assert team_shape.shape_match([]) == {
        "fps": 25, "first_frame": None, "last_frame": None,
        "teams": {"left": {"seconds": [], "average": None},
                  "right": {"seconds": [], "average": None}}}


def test_shape_series_empty():
    assert team_shape.shape_series([], "left") == []
    assert team_shape.shape_series([], "right") == []


def test_team_second_below_minimum_is_none():
    assert team_shape.team_second([(0, 0), (1, 1)], "left") is None
    assert team_shape.team_second([(0, 0), (1, 1), (2, 2)], "left") is None
    assert team_shape.team_second([(0, 0), (1, 1), (2, 2)], "right") is None


def test_team_second_exact_values():
    m = team_shape.team_second([(0, -5), (0, 5), (10, -5), (10, 5)], "left")
    assert m["centroid"] == [5.0, 0.0]
    assert m["width"] == 10.0 and m["length"] == 10.0
    assert m["hull_area"] == 100.0
    assert m["defensive_line"] == 57.5
    # unrounded: every corner sits 5*sqrt(2) from the centroid
    assert m["compactness"] == pytest.approx(5 * math.sqrt(2), abs=1e-12)


def test_team_second_unknown_team_raises():
    with pytest.raises(ValueError):
        team_shape.team_second([(0, 0), (1, 0), (0, 1), (1, 1)], "middle")


def test_shape_series_gaps_direct():
    ts = [s["t"] for s in team_shape.shape_series(gap_recording(), "left")]
    assert ts == [0, 2]


def test_seconds_relative_to_first_frame():
    frames = frames_of(range(25, 75),
                       [person(i, x, y) for i, (x, y) in enumerate(SQUARE_XY, start=1)])
    ts = [s["t"] for s in team_shape.shape_series(frames, "left")]
    assert ts == [0, 1]  # not [1, 2]: t is relative to first_frame = 25, not frame // 25


def test_duplicate_frames_last_wins():
    a = [person(1, 0, 0), person(2, 10, 0), person(3, 0, 10), person(4, 10, 10)]
    b = [person(1, 100, 100), person(2, 10, 0), person(3, 0, 10), person(4, 10, 10)]
    frames = [{"frame": 0, "delay": 0, "people": [dict(p) for p in a],
               "ball": None, "colors": None},
              {"frame": 0, "delay": 0, "people": [dict(p) for p in b],
               "ball": None, "colors": None},
              {"frame": 1, "delay": 0, "people": [dict(p) for p in b],
               "ball": None, "colors": None}]
    assert sum(1 for f in frames if f["frame"] == 0) == 2
    s = team_shape.shape_series(frames, "left")[0]
    # last frame-0 wins: player 1 at (100,100) -> centroid [30, 30]
    assert s["centroid"] == [30.0, 30.0]


def test_shape_match_does_not_mutate_input():
    frames = basic_recording()
    snapshot = copy.deepcopy(frames)
    team_shape.shape_match(frames)
    team_shape.shape_series(frames, "left")
    team_shape.shape_series(frames, "right")
    assert frames == snapshot


def test_missing_positions_ignored_direct():
    people = [person(1, 0, 0), person(2, 10, 0), person(3, 0, 10),
              person(4, 10, 10), person(5, 50, 50, xy=None),
              person(6, 60, 60, xy=[1]), person(7, 70, 70, xy=[float("inf"), 0])]
    # the fixture really holds the three kinds of invalid positions
    assert any(p["xy"] is None for p in people)
    assert any(p["xy"] == [1] for p in people)
    assert any(p["xy"] == [float("inf"), 0] for p in people)
    frames = frames_of([0], people)
    s = team_shape.shape_series(frames, "left")
    assert len(s) == 1
    assert s[0]["centroid"] == [5.0, 5.0]  # the 50/60/70 points did not leak in
    few = [person(1, 0, 0), person(2, 10, 0, xy=None),
           person(3, 0, 10, xy=[1]), person(4, 10, 10)]
    assert team_shape.shape_series(frames_of([0], few), "left") == []
