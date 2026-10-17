"""Tests for the pitch_control feature (spec: features/pitch_control/spec.md).

Every expected value below is computed by hand from the spec.  The tests only
exercise the HTTP surface of the library app; the recording is built in code.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from sideline.library import create_library_app  # noqa: E402

FPS = 25
N_CELLS = 21 * 14  # 294

# Canonical fixture: one left player at (0, 0), one right player at (10, 0).
# Tie line x = 5 = centre of column 11 -> left cols 0..10 (154 cells),
# right cols 12..20 (126 cells), neutral col 11 (14 cells).
CANON = [
    {"id": 1, "role": "player", "team": "left", "jersey": "1", "box": [0, 0, 10, 10], "xy": [0.0, 0.0]},
    {"id": 2, "role": "player", "team": "right", "jersey": "2", "box": [0, 0, 10, 10], "xy": [10.0, 0.0]},
]
SWAP = [
    {"id": 1, "role": "player", "team": "left", "jersey": "1", "box": [0, 0, 10, 10], "xy": [10.0, 0.0]},
    {"id": 2, "role": "player", "team": "right", "jersey": "2", "box": [0, 0, 10, 10], "xy": [0.0, 0.0]},
]

A = {"left": 0.524, "right": 0.429, "neutral": 0.048}          # 154/126/14 of 294
LEFT_ROW = [1.0] * 11 + [0.0] + [0.0] * 9
RIGHT_ROW = [0.0] * 11 + [0.0] + [1.0] * 9


def person(pid, team, x, y, role="player", jersey="1"):
    return {"id": pid, "role": role, "team": team, "jersey": jersey,
            "box": [0, 0, 10, 10], "xy": [x, y]}


def frame(n, people, ball=None):
    return {"frame": n, "delay": 0, "people": people, "ball": ball,
            "colors": {"left": "#d03030", "right": "#2050c0"}}


def recording(frames):
    return ("\n".join(json.dumps(f) for f in frames) + "\n").encode("utf-8")


def frames_const(people, n):
    return [frame(i, list(people)) for i in range(n)]


def frames_offset(people, n, first):
    return [frame(first + i, list(people)) for i in range(n)]


def seconds_frames(per_second_people):
    """25 frames per entry; entry k occupies frames 25k..25k+24."""
    out = []
    for k, people in enumerate(per_second_people):
        out.extend(frame(k * FPS + i, list(people)) for i in range(FPS))
    return out


@pytest.fixture()
def app(tmp_path):
    return TestClient(create_library_app(tmp_path)), tmp_path


def upload(client, frames, name="game"):
    r = client.post("/api/matches",
                    files={"file": (name + ".jsonl", recording(frames), "application/json")},
                    data={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def pc(client, mid, start=None, end=None):
    params = {}
    if start is not None:
        params["start"] = str(start)
    if end is not None:
        params["end"] = str(end)
    return client.get("/api/matches/%s/pitch-control" % mid, params=params)


def canonical_upload(app):
    client, root = app
    return client, root, upload(client, frames_const(CANON, 25))


# -- 404 / 400 validation -------------------------------------------------

def test_unknown_match_is_404(app):
    client, root = app
    assert client.get("/api/matches/m_0000000000/pitch-control").status_code == 404


def test_malformed_match_ids_are_404(app):
    client, root = app
    for mid in ("m_123", "M_0123456789", "pitch-control"):
        assert client.get("/api/matches/%s/pitch-control" % mid).status_code == 404


def test_start_greater_than_end_is_400(app):
    client, root, mid = canonical_upload(app)
    r = pc(client, mid, start=10, end=5)
    assert r.status_code == 400
    assert r.json()["error"] == "start must not be greater than end"


def test_non_integer_start_is_400(app):
    client, root, mid = canonical_upload(app)
    r = pc(client, mid, start="abc")
    assert r.status_code == 400
    assert r.json()["error"] == "start must be a non-negative integer"


def test_negative_end_is_400(app):
    client, root, mid = canonical_upload(app)
    r = pc(client, mid, end="-1")
    assert r.status_code == 400
    assert r.json()["error"] == "end must be a non-negative integer"


# -- canonical 25-frame match ---------------------------------------------

def test_canonical_top_level_fields(app):
    client, root, mid = canonical_upload(app)
    r = pc(client, mid)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["match_id"] == mid
    assert b["start"] == 0 and b["end"] == 24
    assert b["fps"] == 25
    assert b["seconds"] == 1
    assert b["speed_ms"] == 5.0
    assert b["grid"] == {"nx": 21, "ny": 14, "x_range": [-52.5, 52.5], "y_range": [-34.0, 34.0]}


def test_canonical_series_and_share(app):
    client, root, mid = canonical_upload(app)
    b = pc(client, mid).json()
    assert b["series"] == [{"second": 0, "left": 0.524, "right": 0.429, "neutral": 0.048}]
    assert b["share"] == {"left": 0.524, "right": 0.429, "neutral": 0.048}


def test_canonical_control_maps(app):
    client, root, mid = canonical_upload(app)
    b = pc(client, mid).json()
    assert b["control"]["left"] == [LEFT_ROW] * 14
    assert b["control"]["right"] == [RIGHT_ROW] * 14
    assert len(b["control"]["left"]) == 14 and len(b["control"]["left"][0]) == 21


def test_symmetric_teams_tie_column(app):
    # left (-10, 0), right (10, 0): tie line x = 0 = centre of column 10.
    people = [person(1, "left", -10.0, 0.0), person(2, "right", 10.0, 0.0)]
    client, root, mid = app[0], app[1], upload(app[0], frames_const(people, 25))
    b = pc(client, mid).json()
    assert b["share"] == {"left": 0.476, "right": 0.476, "neutral": 0.048}
    assert b["control"]["left"][0] == [1.0] * 10 + [0.0] * 11
    assert b["control"]["right"][0] == [0.0] * 10 + [0.0] + [1.0] * 10


def test_team_uses_nearest_of_its_players(app):
    # left: player (0, 0) and goalkeeper (-45, 0); right: player (-40, 0).
    # left owns c0,c1 (gk closest), c2..c5 right, c6 tie (20 m both ways),
    # c7..c20 left (the player at (0, 0) beats the right player everywhere
    # from c7 on).  Without the goalkeeper, c0 and c1 would go to right.
    people = [person(1, "left", 0.0, 0.0),
              person(3, "left", -45.0, 0.0, role="goalkeeper"),
              person(2, "right", -40.0, 0.0)]
    client, root, mid = app[0], app[1], upload(app[0], frames_const(people, 25))
    b = pc(client, mid).json()
    assert b["share"] == {"left": 0.762, "right": 0.190, "neutral": 0.048}
    assert b["control"]["left"][0] == \
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert b["control"]["right"][0] == \
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_goalkeeper_counts_for_its_team(app):
    # left has only a goalkeeper at (-45, 0); right a player at (10, 0).
    # c0..c6 left (98 cells), c7..c20 right (196 cells), no ties.
    people = [person(1, "left", -45.0, 0.0, role="goalkeeper"),
              person(2, "right", 10.0, 0.0)]
    client, root, mid = app[0], app[1], upload(app[0], frames_const(people, 25))
    b = pc(client, mid).json()
    assert b["share"] == {"left": 0.333, "right": 0.667, "neutral": 0.0}
    assert b["control"]["left"][0] == [1.0] * 7 + [0.0] * 14


def test_referee_does_not_count(app):
    # an adversarial "left" referee at (8, 0) must not win column 11.
    people = [person(1, "left", 0.0, 0.0), person(2, "right", 10.0, 0.0),
              person(3, "left", 8.0, 0.0, role="referee")]
    client, root, mid = app[0], app[1], upload(app[0], frames_const(people, 25))
    b = pc(client, mid).json()
    assert b["share"] == A
    assert b["control"]["left"][0] == LEFT_ROW


def test_team_null_is_ignored(app):
    # an unassigned person at (0, 0) helps nobody; right (10, 0) takes all.
    people = [person(1, None, 0.0, 0.0), person(2, "right", 10.0, 0.0)]
    client, root, mid = app[0], app[1], upload(app[0], frames_const(people, 25))
    b = pc(client, mid).json()
    assert b["share"] == {"left": 0.0, "right": 1.0, "neutral": 0.0}
    assert b["control"]["right"][0] == [1.0] * 21


def test_off_pitch_position_counts(app):
    # left player off the pitch at (-60, 0); right at (10, 0):
    # tie at x = -25 (column 5), left c0..c4, right c6..c20.
    people = [person(1, "left", -60.0, 0.0), person(2, "right", 10.0, 0.0)]
    client, root, mid = app[0], app[1], upload(app[0], frames_const(people, 25))
    b = pc(client, mid).json()
    assert b["share"] == {"left": 0.238, "right": 0.714, "neutral": 0.048}
    assert b["control"]["left"][0] == [1.0] * 5 + [0.0] * 16


def test_duplicate_id_first_entry_wins(app):
    # the duplicated left id at (100, 0) must be ignored:
    # if it won, left would own no cell at all.
    people = [person(1, "left", 0.0, 0.0), person(1, "left", 100.0, 0.0),
              person(2, "right", 10.0, 0.0)]
    client, root, mid = app[0], app[1], upload(app[0], frames_const(people, 25))
    b = pc(client, mid).json()
    assert b["share"] == A
    assert b["control"]["left"][0] == LEFT_ROW


# -- neutral / absent inputs ----------------------------------------------

def test_ball_is_ignored(app):
    # a positioned ball with no people controls nothing.
    frames = [frame(i, [], ball={"box": [0, 0, 4, 4], "xy": [0.0, 0.0]}) for i in range(25)]
    client, root, mid = app[0], app[1], upload(app[0], frames)
    b = pc(client, mid).json()
    assert b["series"] == [{"second": 0, "left": 0.0, "right": 0.0, "neutral": 1.0}]
    assert b["share"] == {"left": 0.0, "right": 0.0, "neutral": 1.0}
    assert b["control"]["left"] == [[0.0] * 21] * 14
    assert b["control"]["right"] == [[0.0] * 21] * 14


def test_no_people_all_neutral(app):
    client, root, mid = app[0], app[1], upload(app[0], frames_const([], 25))
    b = pc(client, mid).json()
    assert b["series"] == [{"second": 0, "left": 0.0, "right": 0.0, "neutral": 1.0}]
    assert b["share"] == {"left": 0.0, "right": 0.0, "neutral": 1.0}
    assert b["control"]["left"] == [[0.0] * 21] * 14
    assert b["control"]["right"] == [[0.0] * 21] * 14


# -- one bad frame cannot flip a second ------------------------------------

def test_dropout_first_frame(app):
    # frame 0 has no right player at all; the other 24 frames are canonical.
    frames = [frame(0, [person(1, "left", 0.0, 0.0)])]
    frames.extend(frame(i, list(CANON)) for i in range(1, 25))
    client, root, mid = app[0], app[1], upload(app[0], frames)
    b = pc(client, mid).json()
    assert b["share"] == A
    assert b["control"]["left"][0] == LEFT_ROW


def test_dropout_middle_frame(app):
    frames = [frame(i, list(CANON)) for i in range(25)]
    frames[17] = frame(17, [person(1, "left", 0.0, 0.0)])
    client, root, mid = app[0], app[1], upload(app[0], frames)
    b = pc(client, mid).json()
    assert b["share"] == A


def test_corrupt_first_frame_position(app):
    # frame 0 has a corrupted left position far off the pitch; the other
    # 24 frames are canonical.  1 bad frame of 25 is not a majority.
    frames = [frame(0, [person(1, "left", -1000.0, 0.0),
                        person(2, "right", 10.0, 0.0)])]
    frames.extend(frame(i, list(CANON)) for i in range(1, 25))
    client, root, mid = app[0], app[1], upload(app[0], frames)
    b = pc(client, mid).json()
    assert b["share"] == A
    assert b["control"]["left"][0] == LEFT_ROW


def test_split_second_12_12_is_all_neutral(app):
    # 24 frames: canonical for 12, swapped for 12 -> 12 vs 12 everywhere,
    # which is not a strict majority (> 12), so every cell is neutral.
    frames = [frame(i, list(CANON if i < 12 else SWAP)) for i in range(24)]
    client, root, mid = app[0], app[1], upload(app[0], frames)
    b = pc(client, mid).json()
    assert b["seconds"] == 1
    assert b["series"] == [{"second": 0, "left": 0.0, "right": 0.0, "neutral": 1.0}]
    assert b["share"] == {"left": 0.0, "right": 0.0, "neutral": 1.0}
    assert b["control"]["left"] == [[0.0] * 21] * 14
    assert b["control"]["right"] == [[0.0] * 21] * 14


def test_split_second_13_12_majority_wins(app):
    # 25 frames: canonical for 13, swapped for 12 -> the 13-frame side wins
    # the contested cells, so the answer equals the canonical one.
    frames = [frame(i, list(CANON if i < 13 else SWAP)) for i in range(25)]
    client, root, mid = app[0], app[1], upload(app[0], frames)
    b = pc(client, mid).json()
    assert b["share"] == A


# -- time series, windows, single frame ------------------------------------

def test_multi_second_series_and_share(app):
    # second 0 canonical, second 1 swapped.
    client, root, mid = app[0], app[1], upload(app[0], seconds_frames([CANON, SWAP]))
    b = pc(client, mid).json()
    assert b["seconds"] == 2 and b["end"] == 49
    assert b["series"] == [
        {"second": 0, "left": 0.524, "right": 0.429, "neutral": 0.048},
        {"second": 1, "left": 0.429, "right": 0.524, "neutral": 0.048},
    ]
    # 154 + 126 = 280 of 588 cell-seconds -> 0.476190...
    assert b["share"] == {"left": 0.476, "right": 0.476, "neutral": 0.048}
    # each team owns c0..c10 in one second and c12..c20 in the other.
    row = [0.5] * 11 + [0.0] + [0.5] * 9
    assert b["control"]["left"] == [row] * 14
    assert b["control"]["right"] == [row] * 14


def test_window_subset(app):
    client, root, mid = app[0], app[1], upload(app[0], seconds_frames([CANON, SWAP]))
    b = pc(client, mid, start=25, end=49).json()
    assert b["start"] == 25 and b["end"] == 49 and b["seconds"] == 1
    assert b["series"] == [{"second": 0, "left": 0.429, "right": 0.524, "neutral": 0.048}]
    assert b["share"] == {"left": 0.429, "right": 0.524, "neutral": 0.048}


def test_window_with_missing_second(app):
    # 25 frames recorded, window 0..49 -> second 1 has no frames at all.
    client, root, mid = canonical_upload(app)
    b = pc(client, mid, start=0, end=49).json()
    assert b["seconds"] == 2
    assert b["series"][1] == {"second": 1, "left": 0.0, "right": 0.0, "neutral": 1.0}
    # 154 / 588 = 0.261904..., 126 / 588 = 0.214285..., 308 / 588 = 0.523809...
    assert b["share"] == {"left": 0.262, "right": 0.214, "neutral": 0.524}


def test_window_beyond_last_frame(app):
    # 25 frames recorded, window 0..99 -> seconds 1..3 are all neutral.
    client, root, mid = canonical_upload(app)
    b = pc(client, mid, start=0, end=99).json()
    assert b["seconds"] == 4
    for i in (1, 2, 3):
        assert b["series"][i] == {"second": i, "left": 0.0, "right": 0.0, "neutral": 1.0}
    # 154 / 1176 = 0.130952..., 126 / 1176 = 0.107142..., 896 / 1176 = 0.761904...
    assert b["share"] == {"left": 0.131, "right": 0.107, "neutral": 0.762}


def test_window_far_beyond_match(app):
    client, root, mid = canonical_upload(app)
    b = pc(client, mid, start=1000, end=1024).json()
    assert b["seconds"] == 1
    assert b["series"] == [{"second": 0, "left": 0.0, "right": 0.0, "neutral": 1.0}]
    assert b["share"] == {"left": 0.0, "right": 0.0, "neutral": 1.0}


def test_single_frame_match(app):
    client, root, mid = app[0], app[1], upload(app[0], frames_const(CANON, 1))
    b = pc(client, mid).json()
    assert b["start"] == 0 and b["end"] == 0 and b["seconds"] == 1
    assert b["series"] == [{"second": 0, "left": 0.524, "right": 0.429, "neutral": 0.048}]
    assert b["control"]["left"] == [LEFT_ROW] * 14


def test_buckets_align_to_first_frame(app):
    # frames 5..29: the default window (start=5) is one whole second even
    # though the frames straddle the absolute multiple of 25 (frame 25).
    client, root, mid = app[0], app[1], upload(app[0], frames_offset(CANON, 25, first=5))
    b = pc(client, mid).json()
    assert b["start"] == 5 and b["end"] == 29
    assert b["seconds"] == 1
    assert b["series"] == [{"second": 0, "left": 0.524, "right": 0.429, "neutral": 0.048}]


# -- idempotence, caching, no mutation, discovery ---------------------------

def test_idempotent_response(app):
    client, root, mid = canonical_upload(app)
    r1 = pc(client, mid)
    r2 = pc(client, mid)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()


def test_result_cached_to_file(app):
    client, root, mid = canonical_upload(app)
    r = pc(client, mid)
    assert r.status_code == 200
    path = root / mid / "features" / "pitch-control-0-24.json"
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == r.json()


def test_frames_not_mutated(app):
    client, root, mid = app[0], app[1], upload(app[0], seconds_frames([CANON, SWAP]))
    before = client.get("/api/matches/%s/frames" % mid).json()
    pc(client, mid)
    pc(client, mid, start=0, end=49)
    pc(client, mid, start=10, end=60)
    after = client.get("/api/matches/%s/frames" % mid).json()
    assert before == after


def test_feature_is_listed(app):
    client, root = app
    r = client.get("/api/features")
    assert r.status_code == 200
    feature = next(f for f in r.json()["features"] if f["name"] == "pitch_control")
    assert feature["ok"] is True


# -- performance ------------------------------------------------------------

def perf_recording(n=25000):
    """25,000 frames with 11 left + 11 right players moving on a fixed pattern."""
    lines = []
    for f in range(n):
        people = []
        for i in range(22):
            team = "left" if i < 11 else "right"
            x = round(-52.0 + (i * 7.3 + 0.2 * f) % 105.0, 2)
            y = round(-34.0 + (i * 3.1 + 0.1 * f) % 68.0, 2)
            people.append({"id": i + 1, "role": "player", "team": team,
                           "jersey": str(i % 11), "box": [0, 0, 10, 10], "xy": [x, y]})
        ball = {"box": [0, 0, 4, 4], "xy": [0.0, 0.0]} if f % 2 == 0 else None
        lines.append(json.dumps({"frame": f, "delay": 0, "people": people, "ball": ball,
                                 "colors": {"left": "#d03030", "right": "#2050c0"}}))
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_performance_25000_frames(app):
    client, root = app
    r = client.post("/api/matches",
                    files={"file": ("big.jsonl", perf_recording(), "application/json")},
                    data={"name": "big"})
    assert r.status_code == 201, r.text
    mid = r.json()["id"]
    t0 = time.monotonic()
    r = client.get("/api/matches/%s/pitch-control" % mid)
    elapsed = time.monotonic() - t0
    assert r.status_code == 200, r.text
    assert elapsed < 30.0, "25,000-frame match took %.1f s (budget 30 s)" % elapsed
    b = r.json()
    assert b["seconds"] == 1000
    assert len(b["series"]) == 1000
    assert b["series"][0]["second"] == 0 and b["series"][-1]["second"] == 999
    s = b["share"]
    assert 0.0 <= s["left"] <= 1.0 and 0.0 <= s["right"] <= 1.0 and 0.0 <= s["neutral"] <= 1.0
    assert s["left"] + s["right"] + s["neutral"] <= 1.002
