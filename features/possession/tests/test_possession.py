"""Acceptance tests for the possession feature (spec.md in the feature folder).

Every expected value is computed by hand from the spec: fps 25, one frame is
0.04 s, reach is 3 m (boundary included), a possession run needs at least
14 consecutive frames (13 x 0.04 = 0.52 s >= 0.5 s), windows are 300 s =
7500 frames wide. The feature is not built yet, so the route answers 404 and
the module import fails: the whole suite must fail right now and pass only
once src/sideline/features/possession.py implements the spec.
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------- helpers

def P(pid, x, y, team="left", role="player", xy=None):
    """One people[] entry; xy=None forces "xy": null (position unknown)."""
    return {"id": pid, "role": role, "team": team, "jersey": str(pid),
            "box": [0, 0, 10, 10], "xy": [x, y] if xy is None else xy}


def F(n, ball=None, *people):
    """One frame. ball: None (missing), a 2-tuple, or "xy_null" (found in
    the image but not placed on the pitch)."""
    if ball == "xy_null":
        ball_obj = {"box": [0, 0, 12, 12], "xy": None}
    elif ball is None:
        ball_obj = None
    else:
        ball_obj = {"box": [0, 0, 12, 12], "xy": [ball[0], ball[1]]}
    return {"frame": n, "delay": 0, "colors": None, "ball": ball_obj,
            "people": list(people)}


def block(start, count, ball, people):
    """`count` consecutive frames from `start`, all with the same ball
    position and people."""
    return [F(start + i, ball, *people) for i in range(count)]


def seg(a, b, owner, seconds):
    return {"start_frame": a, "end_frame": b, "owner": owner, "seconds": seconds}


def win(w, frames, left, right, none):
    return {"window": w, "from_s": 300 * w, "to_s": 300 * (w + 1),
            "frames": frames, "left": left, "right": right, "none": none}


def pct(left, right, none):
    return {"left": left, "right": right, "none": none}


def expected(frame_count, duration_s, timeline, percentages, windows):
    return {"fps": 25, "frame_count": frame_count, "duration_s": duration_s,
            "timeline": timeline, "percentages": percentages, "windows": windows}


def compute(frames):
    from sideline.features import possession
    return possession.compute_possession(frames)


def to_bytes(frames):
    return ("\n".join(json.dumps(f) for f in frames) + "\n").encode()


@pytest.fixture()
def client(tmp_path):
    from sideline import library
    return TestClient(library.create_library_app(tmp_path / "store")), tmp_path / "store"


def upload(c, frames):
    r = c.post("/api/matches",
               files={"file": ("game.jsonl", to_bytes(frames), "application/json")})
    assert r.status_code == 201, r.text
    return r.json()


def possession(c, mid):
    return c.get("/api/matches/%s/possession" % mid)


# Standard 20-frame situations, all around ball at (0, 0):
# L_NEAR is 1 m away (in reach), R_FAR is 40 m away (out of reach).
NEAR = [P(1, 1.0, 0.0), P(2, 40.0, 0.0, team="right")]


# --------------------------------------------------------------------- pure function

def test_hold_30_frames_left():
    frames = block(0, 30, (0, 0), NEAR)
    assert compute(frames) == expected(
        30, 1.2,
        [seg(0, 29, "left", 1.2)],
        pct(100.0, 0.0, 0.0),
        [win(0, 30, 100.0, 0.0, 0.0)])


def test_13_frames_below_half_second_not_counted():
    frames = block(0, 13, (0, 0), NEAR)
    assert compute(frames) == expected(
        13, 0.52,
        [seg(0, 12, "none", 0.52)],
        pct(0.0, 0.0, 100.0),
        [win(0, 13, 0.0, 0.0, 100.0)])


def test_14_frames_meets_half_second():
    frames = block(0, 14, (0, 0), NEAR)
    assert compute(frames) == expected(
        14, 0.56,
        [seg(0, 13, "left", 0.56)],
        pct(100.0, 0.0, 0.0),
        [win(0, 14, 100.0, 0.0, 0.0)])


def test_distance_exactly_3_m_is_in_reach():
    frames = block(0, 20, (0, 0), [P(1, 3.0, 0.0), P(2, 40.0, 0.0, team="right")])
    assert compute(frames) == expected(
        20, 0.8,
        [seg(0, 19, "left", 0.8)],
        pct(100.0, 0.0, 0.0),
        [win(0, 20, 100.0, 0.0, 0.0)])


def test_distance_3_point_1_m_is_out_of_reach():
    frames = block(0, 20, (0, 0), [P(1, 3.1, 0.0), P(2, 40.0, 0.0, team="right")])
    assert compute(frames) == expected(
        20, 0.8,
        [seg(0, 19, "none", 0.8)],
        pct(0.0, 0.0, 100.0),
        [win(0, 20, 0.0, 0.0, 100.0)])


def test_equal_distances_is_none():
    frames = block(0, 20, (0, 0), [P(1, -1.0, 0.0), P(2, 1.0, 0.0, team="right")])
    assert compute(frames) == expected(
        20, 0.8,
        [seg(0, 19, "none", 0.8)],
        pct(0.0, 0.0, 100.0),
        [win(0, 20, 0.0, 0.0, 100.0)])


def test_possession_switch_gives_two_segments():
    frames = block(0, 20, (0, 0), [P(1, 1.0, 0.0), P(2, 50.0, 0.0, team="right")])
    frames += block(20, 20, (50, 0), [P(1, 1.0, 0.0), P(2, 49.0, 0.0, team="right")])
    assert compute(frames) == expected(
        40, 1.6,
        [seg(0, 19, "left", 0.8), seg(20, 39, "right", 0.8)],
        pct(50.0, 50.0, 0.0),
        [win(0, 40, 50.0, 50.0, 0.0)])


def test_frame_gap_breaks_the_run():
    frames = block(0, 7, (0, 0), NEAR)
    frames += block(10, 7, (0, 0), NEAR)
    assert compute(frames) == expected(
        14, 0.68,
        [seg(0, 6, "none", 0.28), seg(10, 16, "none", 0.28)],
        pct(0.0, 0.0, 100.0),
        [win(0, 14, 0.0, 0.0, 100.0)])


def test_missing_ball_stops_the_run():
    frames = block(0, 15, (0, 0), NEAR)
    frames += block(15, 15, None, NEAR)
    assert compute(frames) == expected(
        30, 1.2,
        [seg(0, 14, "left", 0.6), seg(15, 29, "none", 0.6)],
        pct(50.0, 0.0, 50.0),
        [win(0, 30, 50.0, 0.0, 50.0)])


def test_ball_found_but_unplaced_is_missing():
    frames = block(0, 3, "xy_null", [P(1, 1.0, 0.0)])
    assert compute(frames) == expected(
        3, 0.12,
        [seg(0, 2, "none", 0.12)],
        pct(0.0, 0.0, 100.0),
        [win(0, 3, 0.0, 0.0, 100.0)])


def test_referee_nearest_is_ignored():
    frames = block(0, 20, (0, 0), [
        P(9, 0.5, 0.0, team=None, role="referee"),
        P(1, 10.0, 0.0),
        P(2, 30.0, 0.0, team="right")])
    assert compute(frames) == expected(
        20, 0.8,
        [seg(0, 19, "none", 0.8)],
        pct(0.0, 0.0, 100.0),
        [win(0, 20, 0.0, 0.0, 100.0)])


def test_goalkeeper_counts_as_player():
    frames = block(0, 20, (0, 0), [
        P(1, 1.0, 0.0, role="goalkeeper"), P(2, 40.0, 0.0, team="right")])
    assert compute(frames) == expected(
        20, 0.8,
        [seg(0, 19, "left", 0.8)],
        pct(100.0, 0.0, 0.0),
        [win(0, 20, 100.0, 0.0, 0.0)])


def test_nearest_player_of_a_team_wins():
    # The 5 m left player is listed first; the 1 m one must win.
    frames = block(0, 20, (0, 0), [
        P(3, 5.0, 0.0), P(4, 1.0, 0.0), P(2, 40.0, 0.0, team="right")])
    assert compute(frames) == expected(
        20, 0.8,
        [seg(0, 19, "left", 0.8)],
        pct(100.0, 0.0, 0.0),
        [win(0, 20, 100.0, 0.0, 0.0)])


def test_no_people_is_none():
    frames = block(0, 5, (0, 0), [])
    assert compute(frames) == expected(
        5, 0.2,
        [seg(0, 4, "none", 0.2)],
        pct(0.0, 0.0, 100.0),
        [win(0, 5, 0.0, 0.0, 100.0)])


def test_single_frame_recording():
    frames = [F(0, None, P(1, 1.0, 0.0))]
    assert compute(frames) == expected(
        1, 0.04,
        [seg(0, 0, "none", 0.04)],
        pct(0.0, 0.0, 100.0),
        [win(0, 1, 0.0, 0.0, 100.0)])


def test_first_frame_not_zero():
    frames = block(100, 14, (0, 0), NEAR)
    assert compute(frames) == expected(
        14, 0.56,
        [seg(100, 113, "left", 0.56)],
        pct(100.0, 0.0, 0.0),
        [win(0, 14, 100.0, 0.0, 0.0)])


def test_windows_skip_empty_spans():
    frames = block(0, 25, (0, 0), NEAR)
    frames += block(15000, 25, (50, 0),
                    [P(1, 1.0, 0.0), P(2, 49.0, 0.0, team="right")])
    assert compute(frames) == expected(
        50, 601.0,
        [seg(0, 24, "left", 1.0), seg(15000, 15024, "right", 1.0)],
        pct(50.0, 50.0, 0.0),
        [win(0, 25, 100.0, 0.0, 0.0), win(2, 25, 0.0, 100.0, 0.0)])


def test_short_runs_do_not_accumulate():
    # left 6 frames, ball away 6 frames, left 6 frames: 6 < 14, no possession.
    away = [P(1, 1.0, 0.0), P(2, 40.0, 0.0, team="right")]
    frames = block(0, 6, (0, 0), NEAR)
    frames += block(6, 6, (20, 0), away)
    frames += block(12, 6, (0, 0), NEAR)
    assert compute(frames) == expected(
        18, 0.72,
        [seg(0, 17, "none", 0.72)],
        pct(0.0, 0.0, 100.0),
        [win(0, 18, 0.0, 0.0, 100.0)])


def test_percentages_rounded_to_one_decimal():
    # 48 frames: 14 left, 3 none, 14 right, 17 none.
    away = [P(1, 1.0, 0.0), P(2, 50.0, 0.0, team="right")]
    frames = block(0, 14, (0, 0), NEAR)
    frames += block(14, 3, (20, 0), away)
    frames += block(17, 14, (50, 0),
                    [P(1, 1.0, 0.0), P(2, 49.0, 0.0, team="right")])
    frames += block(31, 17, (20, 0), away)
    assert compute(frames) == expected(
        48, 1.92,
        [seg(0, 13, "left", 0.56), seg(14, 16, "none", 0.12),
         seg(17, 30, "right", 0.56), seg(31, 47, "none", 0.68)],
        pct(29.2, 29.2, 41.7),
        [win(0, 48, 29.2, 29.2, 41.7)])


def test_input_frames_not_mutated():
    frames = block(0, 20, (0, 0), [P(1, 1.0, 0.0), P(2, 50.0, 0.0, team="right")])
    frames += block(20, 20, (50, 0), [P(1, 1.0, 0.0), P(2, 49.0, 0.0, team="right")])
    before = copy.deepcopy(frames)
    compute(frames)
    assert frames == before


def test_empty_frame_list():
    assert compute([]) == expected(0, 0.0, [], pct(0.0, 0.0, 0.0), [])


def test_only_one_team_present():
    frames = block(0, 20, (0, 0), [P(1, 1.0, 0.0)])
    assert compute(frames) == expected(
        20, 0.8,
        [seg(0, 19, "left", 0.8)],
        pct(100.0, 0.0, 0.0),
        [win(0, 20, 100.0, 0.0, 0.0)])


def test_duplicate_track_ids_counted_per_entry():
    frames = block(0, 20, (0, 0), [
        P(5, 50.0, 0.0), P(5, 1.0, 0.0), P(2, 40.0, 0.0, team="right")])
    assert compute(frames) == expected(
        20, 0.8,
        [seg(0, 19, "left", 0.8)],
        pct(100.0, 0.0, 0.0),
        [win(0, 20, 100.0, 0.0, 0.0)])


def test_non_dict_people_entries_ignored():
    frames = block(0, 20, (0, 0), [P(1, 1.0, 0.0), "junk", 42])
    assert compute(frames) == expected(
        20, 0.8,
        [seg(0, 19, "left", 0.8)],
        pct(100.0, 0.0, 0.0),
        [win(0, 20, 100.0, 0.0, 0.0)])


def test_null_xy_and_unknown_team_ignored():
    frames = block(0, 20, (0, 0), [
        P(2, 99.0, 99.0, xy=None), P(3, 0.5, 0.0, team=None)])
    assert compute(frames) == expected(
        20, 0.8,
        [seg(0, 19, "none", 0.8)],
        pct(0.0, 0.0, 100.0),
        [win(0, 20, 0.0, 0.0, 100.0)])


def test_frame_without_people_key():
    frames = [{"frame": i, "delay": 0, "colors": None,
               "ball": {"box": [0, 0, 12, 12], "xy": [0.0, 0.0]}} for i in range(5)]
    assert compute(frames) == expected(
        5, 0.2,
        [seg(0, 4, "none", 0.2)],
        pct(0.0, 0.0, 100.0),
        [win(0, 5, 0.0, 0.0, 100.0)])


def test_frame_at_exactly_300_s_is_in_window_1():
    # 12-frame runs are below the 14-frame minimum, so both stretches are
    # "none"; the point of the test is the window split: frame 7500 is
    # exactly 300 s after frame 0, so it belongs to window 1.
    frames = block(0, 12, (0, 0), NEAR)
    frames += block(7500, 12, (0, 0), NEAR)
    assert compute(frames) == expected(
        24, 300.48,
        [seg(0, 11, "none", 0.48), seg(7500, 7511, "none", 0.48)],
        pct(0.0, 0.0, 100.0),
        [win(0, 12, 0.0, 0.0, 100.0), win(1, 12, 0.0, 0.0, 100.0)])


def test_seconds_rounded_to_two_decimals():
    frames = block(0, 47, (0, 0), NEAR)
    assert compute(frames) == expected(
        47, 1.88,
        [seg(0, 46, "left", 1.88)],
        pct(100.0, 0.0, 0.0),
        [win(0, 47, 100.0, 0.0, 0.0)])


# --------------------------------------------------------------------- HTTP route

def switch_frames():
    return (block(0, 20, (0, 0), [P(1, 1.0, 0.0), P(2, 50.0, 0.0, team="right")])
            + block(20, 20, (50, 0), [P(1, 1.0, 0.0), P(2, 49.0, 0.0, team="right")]))


def test_http_possession_body(client):
    c, _ = client
    mid = upload(c, switch_frames())["id"]
    r = possession(c, mid)
    assert r.status_code == 200
    assert r.json() == {
        "match_id": mid, "fps": 25, "frame_count": 40, "duration_s": 1.6,
        "timeline": [seg(0, 19, "left", 0.8), seg(20, 39, "right", 0.8)],
        "percentages": pct(50.0, 50.0, 0.0),
        "windows": [win(0, 40, 50.0, 50.0, 0.0)]}


def test_http_unknown_match_404(client):
    c, _ = client
    assert possession(c, "m_0000000000").status_code == 404


def test_http_malformed_ids_404(client):
    c, _ = client
    for bad in ("m_12345678901", "m_short", "m_NOTHEX1234", "no_id_at_all"):
        assert possession(c, bad).status_code == 404, bad


def test_http_idempotent(client):
    c, _ = client
    mid = upload(c, switch_frames())["id"]
    r1, r2 = possession(c, mid), possession(c, mid)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json()


def test_http_cache_file_written(client):
    c, store = client
    mid = upload(c, switch_frames())["id"]
    assert possession(c, mid).status_code == 200
    cache = store / mid / "features" / "possession.json"
    assert cache.is_file()
    json.loads(cache.read_text(encoding="utf-8"))  # a valid JSON document


def test_http_frames_and_meta_not_mutated(client):
    c, _ = client
    mid = upload(c, switch_frames())["id"]
    frames_before = c.get("/api/matches/%s/frames" % mid).json()
    meta_before = c.get("/api/matches/%s" % mid).json()
    assert possession(c, mid).status_code == 200
    assert c.get("/api/matches/%s/frames" % mid).json() == frames_before
    assert c.get("/api/matches/%s" % mid).json() == meta_before


def test_http_invalid_upload_400(client):
    c, _ = client
    data = to_bytes([F(5, (0, 0)), F(5, (0, 0))])  # frame not strictly increasing
    r = c.post("/api/matches",
               files={"file": ("game.jsonl", data, "application/json")})
    assert r.status_code == 400


def test_http_out_of_reach_is_all_none(client):
    c, _ = client
    mid = upload(c, block(0, 20, (0, 0),
                          [P(1, 3.1, 0.0), P(2, 40.0, 0.0, team="right")]))["id"]
    r = possession(c, mid)
    assert r.status_code == 200
    body = r.json()
    assert body["timeline"] == [seg(0, 19, "none", 0.8)]
    assert body["percentages"] == pct(0.0, 0.0, 100.0)


def test_http_multi_window(client):
    c, _ = client
    frames = block(0, 25, (0, 0), NEAR)
    frames += block(15000, 25, (50, 0),
                    [P(1, 1.0, 0.0), P(2, 49.0, 0.0, team="right")])
    mid = upload(c, frames)["id"]
    r = possession(c, mid)
    assert r.status_code == 200
    body = r.json()
    assert body["frame_count"] == 50
    assert body["duration_s"] == 601.0
    assert body["percentages"] == pct(50.0, 50.0, 0.0)
    assert body["windows"] == [win(0, 25, 100.0, 0.0, 0.0),
                               win(2, 25, 0.0, 100.0, 0.0)]


# --------------------------------------------------------------------- performance

_PERF = {}


def perf_data():
    """25,000 consecutive frames, 22 players (11 per team) and a ball each:
    the left player 0 sits at (1, 0), the ball at (0, 0), so left owns every
    frame (d_left = 1.0 <= 3.0, d_right >= 60.0)."""
    if not _PERF:
        frames = []
        for i in range(25000):
            people = [P(k, 1.0 + k * 0.1, k * 0.05) for k in range(11)]
            people += [P(100 + k, 60.0 + k * 0.1, k * 0.05, team="right")
                       for k in range(11)]
            frames.append(F(i, (0.0, 0.0), *people))
        _PERF["frames"] = frames
        _PERF["data"] = to_bytes(frames)
    return _PERF


def test_perf_pure_25000_frames_under_5_s():
    frames = perf_data()["frames"]
    t0 = time.monotonic()
    result = compute(frames)
    assert time.monotonic() - t0 < 5.0
    assert result == expected(
        25000, 1000.0,
        [seg(0, 24999, "left", 1000.0)],
        pct(100.0, 0.0, 0.0),
        [win(0, 7500, 100.0, 0.0, 0.0), win(1, 7500, 100.0, 0.0, 0.0),
         win(2, 7500, 100.0, 0.0, 0.0), win(3, 2500, 100.0, 0.0, 0.0)])


def test_perf_http_25000_frames_under_20_s(client):
    c, _ = client
    r = c.post("/api/matches",
               files={"file": ("game.jsonl", perf_data()["data"], "application/json")})
    assert r.status_code == 201
    mid = r.json()["id"]
    t0 = time.monotonic()
    r = possession(c, mid)
    elapsed = time.monotonic() - t0
    assert r.status_code == 200
    assert elapsed < 20.0
    body = r.json()
    assert body["frame_count"] == 25000
    assert body["timeline"] == [seg(0, 24999, "left", 1000.0)]
    assert body["percentages"] == pct(100.0, 0.0, 0.0)
    assert [w["frames"] for w in body["windows"]] == [7500, 7500, 7500, 2500]
