"""Tests for the intensity feature (spec: features/intensity/spec.md).

Every expected value below is computed by hand from the spec.  The tests only
exercise the HTTP surface of the library app; the recordings are built in code.
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
ZONES = ["walk", "jog", "run", "high", "sprint"]
ZERO_ROW = [0.0, 0.0, 0.0, 0.0, 0.0]


# -- fixture builders --------------------------------------------------------

def person(pid, team, x, y, role="player", jersey="1", xy=True):
    return {"id": pid, "role": role, "team": team, "jersey": jersey,
            "box": [0, 0, 10, 10], "xy": [x, y] if xy else None}


def frame(n, people, ball=None):
    return {"frame": n, "delay": 0, "people": people, "ball": ball,
            "colors": {"left": "#d03030", "right": "#2050c0"}}


def recording(frames):
    return ("\n".join(json.dumps(f) for f in frames) + "\n").encode("utf-8")


def dense_frames(n, at):
    """n frames numbered 0..n-1; frame f carries the people list at[f] if present."""
    return [frame(i, at.get(i, [])) for i in range(n)]


def canonical_frames():
    """1500 frames (0..1499, 60.0 s, one minute).  Every 5 frames, seven people
    appear moving in +x with a constant 5-frame displacement:

      left   id 1: 0.2 m per step  -> v = 1.0  -> walk
      left   id 2: 0.6 m per step  -> v = 3.0  -> jog
      right  id 3: 1.0 m per step  -> v = 5.0  -> run
      right  id 4: 1.5 m per step  -> v = 7.5  -> sprint (goalkeeper)
      right  id 5: 1.3 m per step  -> v = 6.5  -> high
      left   id 9: 2.0 m per step  -> referee  (excluded)
      (null) id 10: 1.0 m per step -> unassigned (excluded)

    299 counted steps per track (frames 0, 5, ..., 1495):
      id 1: 59.8 m, id 2: 179.4 m, id 3: 299.0 m, id 4: 448.5 m, id 5: 388.7 m.
    """
    at = {}
    for k in range(300):
        at[5 * k] = [
            person(1, "left", 0.2 * k, 0.0, jersey="1"),
            person(2, "left", 0.6 * k, 0.0, jersey="2"),
            person(3, "right", k, 0.0, jersey="3"),
            person(4, "right", 1.5 * k, 0.0, role="goalkeeper", jersey="4"),
            person(5, "right", 1.3 * k, 0.0, jersey="5"),
            person(9, "left", 2.0 * k, 0.0, role="referee", jersey="9"),
            person(10, None, k, 0.0, jersey="10"),
        ]
    return dense_frames(1500, at)


def boundary_frames():
    """6 frames (0..5).  Eight left-side people step once, gap 5 (t = 0.2 s):

      id 20: d = 0.4  -> v = 2.0               -> jog     (0.4 m)
      id 21: d = 0.8  -> v = 4.0               -> run     (0.8 m)
      id 22: d = 1.2  -> v = 5.999999999999999 -> run     (1.2 m)
      id 23: d = 1.21 -> v = 6.05              -> high    (1.21 m)
      id 24: d = 1.4  -> v = 6.999999999999999 -> high    (1.4 m)
      id 25: d = 1.41 -> v = 7.049999999999999 -> sprint  (1.41 m)
      id 26: d = 2.4  -> v = 11.999999999999998 -> sprint, counted (2.4 m)
      id 27: d = 2.5  -> v = 12.5              -> discarded (0.0 m)

    Team left: walk 0.0, jog 0.4, run 2.0, high 2.61, sprint 3.81, total 8.82 m;
    duration 0.24 s = 0.004 min -> team work rate 8.82 / 0.004 = 2205.0 m/min.
    """
    ds = {20: 0.4, 21: 0.8, 22: 1.2, 23: 1.21, 24: 1.4, 25: 1.41, 26: 2.4, 27: 2.5}
    out = [frame(0, [person(i, "left", 0.0, 0.0, jersey=str(i)) for i in ds])]
    out.extend(frame(f, []) for f in range(1, 5))
    out.append(frame(5, [person(i, "left", ds[i], 0.0, jersey=str(i)) for i in ds]))
    return out


def per_minute_frames():
    """3000 frames (0..2999, 120.0 s, two minutes).  id 30 (left) appears every
    5 frames at x = 1.0 * k: 599 counted steps of 1.0 m (v = 5.0, run).  300
    steps start in minute 0 (frames 0..1495), 299 in minute 1 (frames 1500..2990)
    -> 300.0 m and 299.0 m, total 599.0 m, work rate 599.0 / 2.0 = 299.5 m/min.
    """
    at = {}
    for k in range(600):
        at[5 * k] = [person(30, "left", 1.0 * k, 0.0, jersey="30")]
    return dense_frames(3000, at)


def minute_boundary_frames():
    """3000 frames (0..2999).  id 40 (left) appears at frame 1495 at (0, 0), at
    frame 1500 at (1, 0) and at frame 1505 at (2, 0): two steps of 1.0 m
    (v = 5.0, run) starting at 1495 (minute 0) and 1500 (minute 1).
    """
    at = {1495: [person(40, "left", 0.0, 0.0, jersey="40")],
          1500: [person(40, "left", 1.0, 0.0, jersey="40")],
          1505: [person(40, "left", 2.0, 0.0, jersey="40")]}
    return dense_frames(3000, at)


def rounding_frames():
    """3000 frames (0..2999, two minutes).  id 31 (left) steps 0.004 m in minute
    0 (frames 0 -> 5); id 32 (left) steps 0.004 m in minute 1 (frames 1500 ->
    1505).  Each player's exact total is 0.004 m (displays 0.0); the team's
    exact total is 0.008 m, which rounds to 0.01 - summing the rounded player
    values (0.0 + 0.0) or the rounded per-minute rows would give 0.0.
    """
    at = {0: [person(31, "left", 0.0, 0.0, jersey="31")],
          5: [person(31, "left", 0.004, 0.0, jersey="31")],
          1500: [person(32, "left", 0.0, 0.0, jersey="32")],
          1505: [person(32, "left", 0.004, 0.0, jersey="32")]}
    return dense_frames(3000, at)


def fresh_frames(d):
    """6 frames (0..5).  id 60 (left) steps once, gap 5, distance d.
    d = 0.8 -> v = 4.0 -> run 0.8 m, work rate 0.8 / 0.004 = 200.0 m/min.
    d = 0.0 -> v = 0.0 -> walk, every value 0.0.
    """
    out = [frame(0, [person(60, "left", 0.0, 0.0, jersey="60")])]
    out.extend(frame(f, []) for f in range(1, 5))
    out.append(frame(5, [person(60, "left", d, 0.0, jersey="60")]))
    return out


def gap_frames():
    """7 frames (0..6, duration 0.28 s).
    id 28 (left): (0,0) -> (0.5,0) over 6 frames: v = 2.0833 <= 12 but
    gap 6 > 5 -> discarded -> 0.0 m.
    id 29 (left): (0,0) -> (0.2,0) over 2 frames: v = 2.5 -> jog, 0.2 m,
    work rate 0.2 / (0.28 / 60) = 42.86 m/min.
    """
    at = {0: [person(28, "left", 0.0, 0.0, jersey="28"),
              person(29, "left", 0.0, 0.0, jersey="29")],
          2: [person(29, "left", 0.2, 0.0, jersey="29")],
          6: [person(28, "left", 0.5, 0.0, jersey="28")]}
    return dense_frames(7, at)


def goalkeeper_frames():
    """11 frames (0..10, duration 0.44 s).  id 1 is a left goalkeeper stepping
    1.0 m every 5 frames (v = 5.0, run): 2.0 m, work rate 2.0 / (0.44 / 60)
    = 272.73 m/min.  id 2 (right) stands still: 0.0 m.
    """
    at = {}
    for k, x in enumerate((0.0, 1.0, 2.0)):
        at[5 * k] = [person(1, "left", x, 0.0, role="goalkeeper", jersey="1"),
                     person(2, "right", 0.0, 0.0, jersey="2")]
    return dense_frames(11, at)


def tie_frames():
    """6 frames (0..5).  ids 50 and 51 (both left) make the identical step
    (0,0) -> (1,0) over 5 frames: 1.0 m each (v = 5.0, run).  The tie in
    top_distance and top_sprint is broken by id ascending.
    """
    at = {0: [person(50, "left", 0.0, 0.0, jersey="50"),
              person(51, "left", 0.0, 0.0, jersey="51")],
          5: [person(50, "left", 1.0, 0.0, jersey="50"),
              person(51, "left", 1.0, 0.0, jersey="51")]}
    return dense_frames(6, at)


def duplicate_frames():
    """6 frames (0..5).  id 70 (left) has two entries in frame 0: (0,0) first
    and (100,0) second; in frame 5 it is at (0.2,0).  First entry wins -> one
    step of 0.2 m (v = 1.0, walk).  If the duplicate won, the 99.8 m jump would
    be discarded (v = 499 > 12) and the total would be 0.0.
    """
    at = {0: [person(70, "left", 0.0, 0.0, jersey="70"),
              person(70, "left", 100.0, 0.0, jersey="70")],
          5: [person(70, "left", 0.2, 0.0, jersey="70")]}
    return dense_frames(6, at)


def missing_position_frames():
    """7 frames (0..6, duration 0.28 s).
    id 80: (0,0) at 0, xy null at 3, (1,0) at 5 -> one step over gap 5,
    d = 1.0, v = 5.0 -> run 1.0 m, work rate 1.0 / (0.28 / 60) = 214.29.
    id 81: (0,0) at 0, xy null at 3, (1,0) at 6 -> gap 6 > 5 -> discarded.
    id 82: xy null in frames 0 and 1 -> a zero entry, jersey stays null.
    """
    at = {0: [person(80, "left", 0.0, 0.0, jersey="80"),
              person(81, "left", 0.0, 0.0, jersey="81"),
              person(82, "left", 0.0, 0.0, jersey=None, xy=False)],
          1: [person(82, "left", 0.0, 0.0, jersey=None, xy=False)],
          3: [person(80, "left", 0.0, 0.0, jersey="80", xy=False),
              person(81, "left", 0.0, 0.0, jersey="81", xy=False)],
          5: [person(80, "left", 1.0, 0.0, jersey="80")],
          6: [person(81, "left", 1.0, 0.0, jersey="81")]}
    return dense_frames(7, at)


def off_pitch_frames():
    """6 frames (0..5).  id 60 (left) steps (60,0) -> (61,0) over 5 frames:
    d = 1.0, v = 5.0 -> run 1.0 m although x = 60 is far off the pitch.
    """
    at = {0: [person(60, "left", 60.0, 0.0, jersey="60")],
          5: [person(60, "left", 61.0, 0.0, jersey="60")]}
    return dense_frames(6, at)


def perf_recording(n=25000):
    """25,000 frames with 11 left + 11 right players moving on a fixed pattern.

    Per-frame displacement ~ (0.2, 0.1) m -> v ~ 5.6 m/s (run zone); the
    modulo wrap-around produces single-frame jumps of ~100 m (v > 12) that the
    step cap must discard.  48,058,202 bytes.
    """
    lines = []
    for f in range(n):
        people = []
        for i in range(22):
            team = "left" if i < 11 else "right"
            x = round(-52.0 + (i * 7.3 + 0.2 * f) % 105.0, 2)
            y = round(-34.0 + (i * 3.1 + 0.1 * f) % 68.0, 2)
            people.append({"id": i + 1, "role": "player", "team": team,
                           "jersey": str(i % 11 + 1), "box": [0, 0, 10, 10],
                           "xy": [x, y]})
        ball = {"box": [0, 0, 4, 4], "xy": [0.0, 0.0]} if f % 2 == 0 else None
        lines.append(json.dumps({"frame": f, "delay": 0, "people": people,
                                 "ball": ball,
                                 "colors": {"left": "#d03030", "right": "#2050c0"}}))
    return ("\n".join(lines) + "\n").encode("utf-8")


# -- app / request helpers ---------------------------------------------------

@pytest.fixture()
def app(tmp_path):
    return TestClient(create_library_app(tmp_path)), tmp_path


def upload(client, frames, name="game"):
    r = client.post("/api/matches",
                    files={"file": (name + ".jsonl", recording(frames),
                                    "application/json")},
                    data={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def intensity(client, mid, **params):
    return client.get("/api/matches/%s/intensity" % mid, params=params or None)


def canonical_upload(app):
    client, root = app
    return client, root, upload(client, canonical_frames())


# -- 404 validation -----------------------------------------------------------

def test_unknown_match_is_404(app):
    client, root = app
    assert intensity(client, "m_0000000000").status_code == 404


def test_malformed_match_ids_are_404(app):
    client, root = app
    for mid in ("m_123", "M_0123456789", "intensity"):
        assert intensity(client, mid).status_code == 404


def test_query_parameters_are_ignored(app):
    client, root, mid = canonical_upload(app)
    r = intensity(client, mid, start="0", end="49", team="left", what="x")
    assert r.status_code == 200, r.text
    b = r.json()
    # the whole match is processed regardless: hand values, not a window.
    assert b["first_frame"] == 0 and b["last_frame"] == 1499
    assert b["teams"]["left"]["distance_m"] == 239.2
    assert b["teams"]["right"]["distance_m"] == 1136.2


# -- canonical 1500-frame match -----------------------------------------------

def test_canonical_top_level_fields(app):
    client, root, mid = canonical_upload(app)
    r = intensity(client, mid)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["match_id"] == mid
    assert b["fps"] == 25
    assert b["first_frame"] == 0 and b["last_frame"] == 1499
    assert b["duration_s"] == 60.0
    assert b["minutes"] == 1
    assert b["zones"] == ZONES


def test_canonical_player_entries(app):
    client, root, mid = canonical_upload(app)
    b = intensity(client, mid).json()
    assert [p["id"] for p in b["players"]] == [1, 2, 3, 4, 5]
    expect = {
        1: ("left", "1", 59.8, {"walk": 59.8, "jog": 0.0, "run": 0.0,
                                "high": 0.0, "sprint": 0.0}),
        2: ("left", "2", 179.4, {"walk": 0.0, "jog": 179.4, "run": 0.0,
                                 "high": 0.0, "sprint": 0.0}),
        3: ("right", "3", 299.0, {"walk": 0.0, "jog": 0.0, "run": 299.0,
                                  "high": 0.0, "sprint": 0.0}),
        4: ("right", "4", 448.5, {"walk": 0.0, "jog": 0.0, "run": 0.0,
                                  "high": 0.0, "sprint": 448.5}),
        5: ("right", "5", 388.7, {"walk": 0.0, "jog": 0.0, "run": 0.0,
                                  "high": 388.7, "sprint": 0.0}),
    }
    players = {p["id"]: p for p in b["players"]}
    for pid, (team, jersey, dist, zones) in expect.items():
        p = players[pid]
        assert p["team"] == team and p["jersey"] == jersey
        assert p["distance_m"] == dist
        assert p["zones_m"] == zones
        # the zones must add up to the total (all values already 2-decimal here)
        assert round(sum(p["zones_m"].values()), 2) == dist


def test_canonical_teams(app):
    client, root, mid = canonical_upload(app)
    b = intensity(client, mid).json()
    left = b["teams"]["left"]
    assert left["distance_m"] == 239.2
    assert left["zones_m"] == {"walk": 59.8, "jog": 179.4, "run": 0.0,
                               "high": 0.0, "sprint": 0.0}
    right = b["teams"]["right"]
    assert right["distance_m"] == 1136.2
    assert right["zones_m"] == {"walk": 0.0, "jog": 0.0, "run": 299.0,
                                "high": 388.7, "sprint": 448.5}
    assert set(b["teams"]) == {"left", "right"}


def test_canonical_per_minute_rows(app):
    client, root, mid = canonical_upload(app)
    b = intensity(client, mid).json()
    players = {p["id"]: p for p in b["players"]}
    for pid in (1, 2, 3, 4, 5):
        assert len(players[pid]["per_minute"]) == 1
    assert players[1]["per_minute"] == [[59.8, 0.0, 0.0, 0.0, 0.0]]
    assert players[4]["per_minute"] == [[0.0, 0.0, 0.0, 0.0, 448.5]]
    assert b["teams"]["left"]["per_minute"] == [[59.8, 179.4, 0.0, 0.0, 0.0]]
    assert b["teams"]["right"]["per_minute"] == [[0.0, 0.0, 299.0, 388.7, 448.5]]


def test_canonical_work_rates(app):
    client, root, mid = canonical_upload(app)
    b = intensity(client, mid).json()
    # the duration is exactly 1.0 minute, so work rate equals the distance
    players = {p["id"]: p for p in b["players"]}
    for pid, wr in ((1, 59.8), (2, 179.4), (3, 299.0), (4, 448.5), (5, 388.7)):
        assert players[pid]["work_rate_m_per_min"] == wr
    assert b["teams"]["left"]["work_rate_m_per_min"] == 239.2
    assert b["teams"]["right"]["work_rate_m_per_min"] == 1136.2


def test_canonical_top_lists(app):
    client, root, mid = canonical_upload(app)
    b = intensity(client, mid).json()
    assert b["top_distance"] == [
        {"id": 4, "team": "right", "jersey": "4", "distance_m": 448.5},
        {"id": 5, "team": "right", "jersey": "5", "distance_m": 388.7},
        {"id": 3, "team": "right", "jersey": "3", "distance_m": 299.0},
        {"id": 2, "team": "left", "jersey": "2", "distance_m": 179.4},
        {"id": 1, "team": "left", "jersey": "1", "distance_m": 59.8},
    ]
    assert b["top_sprint"] == [
        {"id": 4, "team": "right", "jersey": "4", "sprint_m": 448.5},
        {"id": 1, "team": "left", "jersey": "1", "sprint_m": 0.0},
        {"id": 2, "team": "left", "jersey": "2", "sprint_m": 0.0},
        {"id": 3, "team": "right", "jersey": "3", "sprint_m": 0.0},
        {"id": 5, "team": "right", "jersey": "5", "sprint_m": 0.0},
    ]


def test_excluded_people_do_not_appear(app):
    # referee 9 (598 m, team "left") and unassigned 10 (299 m) move a lot: if
    # either leaked in, the players list or the team totals would change.
    client, root, mid = canonical_upload(app)
    b = intensity(client, mid).json()
    assert [p["id"] for p in b["players"]] == [1, 2, 3, 4, 5]
    assert b["teams"]["left"]["distance_m"] == 239.2
    assert b["teams"]["right"]["distance_m"] == 1136.2
    body = json.dumps(b)
    assert '"id": 9' not in body and '"id": 10' not in body


def test_goalkeeper_counts_for_its_team(app):
    client, root = app
    mid = upload(client, goalkeeper_frames())
    b = intensity(client, mid).json()
    players = {p["id"]: p for p in b["players"]}
    assert [p["id"] for p in b["players"]] == [1, 2]
    assert players[1]["zones_m"]["run"] == 2.0
    assert players[1]["distance_m"] == 2.0
    assert players[1]["work_rate_m_per_min"] == 272.73
    assert players[2]["distance_m"] == 0.0
    assert b["teams"]["left"]["distance_m"] == 2.0
    assert b["teams"]["left"]["zones_m"]["run"] == 2.0
    assert b["teams"]["right"]["distance_m"] == 0.0


# -- zones, caps, gaps ---------------------------------------------------------

def test_zone_boundaries(app):
    client, root = app
    mid = upload(client, boundary_frames())
    b = intensity(client, mid).json()
    assert b["duration_s"] == 0.24
    players = {p["id"]: p for p in b["players"]}
    assert [p["id"] for p in b["players"]] == [20, 21, 22, 23, 24, 25, 26, 27]
    # v = 2.0 exactly is jog, v = 4.0 exactly is run
    assert players[20]["zones_m"]["jog"] == 0.4 and players[20]["distance_m"] == 0.4
    assert players[20]["work_rate_m_per_min"] == 100.0
    assert players[21]["zones_m"]["run"] == 0.8
    # v = 5.999999999999999 stays run; v = 6.05 is high
    assert players[22]["zones_m"]["run"] == 1.2
    assert players[23]["zones_m"]["high"] == 1.21
    # v = 6.999999999999999 stays high; v = 7.049999999999999 is sprint
    assert players[24]["zones_m"]["high"] == 1.4
    assert players[25]["zones_m"]["sprint"] == 1.41
    # v = 11.999999999999998 is still <= 12: counted, sprint
    assert players[26]["zones_m"]["sprint"] == 2.4
    assert players[26]["work_rate_m_per_min"] == 600.0
    # v = 12.5 > 12: the step is discarded, everything is zero
    assert players[27]["distance_m"] == 0.0
    assert players[27]["zones_m"] == {"walk": 0.0, "jog": 0.0, "run": 0.0,
                                      "high": 0.0, "sprint": 0.0}
    assert players[27]["work_rate_m_per_min"] == 0.0
    left = b["teams"]["left"]
    assert left["zones_m"] == {"walk": 0.0, "jog": 0.4, "run": 2.0,
                               "high": 2.61, "sprint": 3.81}
    assert left["distance_m"] == 8.82
    assert left["work_rate_m_per_min"] == 2205.0
    assert b["teams"]["right"]["distance_m"] == 0.0


def test_gap_rules(app):
    client, root = app
    mid = upload(client, gap_frames())
    b = intensity(client, mid).json()
    players = {p["id"]: p for p in b["players"]}
    # gap 6 with a plausible speed (2.0833 m/s) is discarded
    assert players[28]["distance_m"] == 0.0
    # gap 2 is counted
    assert players[29]["zones_m"]["jog"] == 0.2
    assert players[29]["work_rate_m_per_min"] == 42.86
    assert b["teams"]["left"]["distance_m"] == 0.2


# -- per-minute buckets ---------------------------------------------------------

def test_step_at_minute_boundary_goes_to_start_minute(app):
    client, root = app
    mid = upload(client, minute_boundary_frames())
    b = intensity(client, mid).json()
    assert b["minutes"] == 2
    players = {p["id"]: p for p in b["players"]}
    p = players[40]
    assert p["distance_m"] == 2.0
    assert p["zones_m"]["run"] == 2.0
    # the step 1495 -> 1500 belongs to minute 0, not minute 1
    # (1.0 m in the run column of each minute)
    assert p["per_minute"] == [[0.0, 0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0, 0.0]]
    assert p["work_rate_m_per_min"] == 1.0


def test_two_minute_per_minute_split(app):
    client, root = app
    mid = upload(client, per_minute_frames())
    b = intensity(client, mid).json()
    assert b["duration_s"] == 120.0
    assert b["minutes"] == 2
    p = b["players"][0]
    assert p["id"] == 30
    assert p["zones_m"]["run"] == 599.0
    assert p["distance_m"] == 599.0
    assert p["per_minute"] == [[0.0, 0.0, 300.0, 0.0, 0.0],
                               [0.0, 0.0, 299.0, 0.0, 0.0]]
    assert p["work_rate_m_per_min"] == 299.5
    assert b["teams"]["left"]["per_minute"] == [[0.0, 0.0, 300.0, 0.0, 0.0],
                                                [0.0, 0.0, 299.0, 0.0, 0.0]]
    assert b["teams"]["left"]["work_rate_m_per_min"] == 299.5
    assert b["teams"]["right"]["distance_m"] == 0.0


# -- rounding -------------------------------------------------------------------

def test_team_total_rounds_exact_sum(app):
    client, root = app
    mid = upload(client, rounding_frames())
    b = intensity(client, mid).json()
    assert b["minutes"] == 2
    players = {p["id"]: p for p in b["players"]}
    for pid in (31, 32):
        # 0.004 m rounds to 0.0 per player
        assert players[pid]["distance_m"] == 0.0
        assert players[pid]["zones_m"]["walk"] == 0.0
        assert players[pid]["per_minute"] == [ZERO_ROW, ZERO_ROW]
        assert players[pid]["work_rate_m_per_min"] == 0.0
    left = b["teams"]["left"]
    # 0.004 + 0.004 = 0.008 rounds to 0.01; the rounded parts sum to 0.0
    assert left["distance_m"] == 0.01
    assert left["zones_m"]["walk"] == 0.01
    assert left["per_minute"] == [ZERO_ROW, ZERO_ROW]
    assert left["work_rate_m_per_min"] == 0.0
    assert b["teams"]["right"]["distance_m"] == 0.0


# -- top lists -------------------------------------------------------------------

def test_ties_broken_by_id(app):
    client, root = app
    mid = upload(client, tie_frames())
    b = intensity(client, mid).json()
    assert b["top_distance"] == [
        {"id": 50, "team": "left", "jersey": "50", "distance_m": 1.0},
        {"id": 51, "team": "left", "jersey": "51", "distance_m": 1.0},
    ]
    assert b["top_sprint"] == [
        {"id": 50, "team": "left", "jersey": "50", "sprint_m": 0.0},
        {"id": 51, "team": "left", "jersey": "51", "sprint_m": 0.0},
    ]
    assert b["teams"]["left"]["distance_m"] == 2.0
    assert b["teams"]["left"]["work_rate_m_per_min"] == 500.0


def test_top_lists_truncated_to_ten(app):
    client, root = app
    mid = upload(client, [frame(0, [person(i, "left", 0.0, 0.0, jersey=str(i))
                                    for i in range(1, 12)])])
    b = intensity(client, mid).json()
    assert len(b["players"]) == 11
    assert len(b["top_distance"]) == 10
    assert [e["id"] for e in b["top_distance"]] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert all(e["distance_m"] == 0.0 for e in b["top_distance"])
    assert len(b["top_sprint"]) == 10
    assert [e["id"] for e in b["top_sprint"]] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# -- ids and positions -----------------------------------------------------------

def test_duplicate_id_first_entry_wins(app):
    client, root = app
    mid = upload(client, duplicate_frames())
    b = intensity(client, mid).json()
    p = b["players"][0]
    assert p["id"] == 70
    # (0,0) -> (0.2,0) is the step; the (100,0) duplicate must be ignored
    assert p["distance_m"] == 0.2
    assert p["zones_m"]["walk"] == 0.2
    assert p["work_rate_m_per_min"] == 50.0


def test_missing_positions(app):
    client, root = app
    mid = upload(client, missing_position_frames())
    b = intensity(client, mid).json()
    players = {p["id"]: p for p in b["players"]}
    assert [p["id"] for p in b["players"]] == [80, 81, 82]
    # the null appearance at frame 3 does not split the step: 0 -> 5, d = 1.0
    assert players[80]["distance_m"] == 1.0
    assert players[80]["zones_m"]["run"] == 1.0
    assert players[80]["work_rate_m_per_min"] == 214.29
    # the same layout over 6 frames is discarded by the gap rule
    assert players[81]["distance_m"] == 0.0
    # all-unpositioned person: a zero entry, null jersey stays null
    assert players[82]["distance_m"] == 0.0
    assert players[82]["jersey"] is None
    assert b["teams"]["left"]["distance_m"] == 1.0


def test_off_pitch_positions_count(app):
    client, root = app
    mid = upload(client, off_pitch_frames())
    b = intensity(client, mid).json()
    p = b["players"][0]
    assert p["id"] == 60
    assert p["distance_m"] == 1.0
    assert p["zones_m"]["run"] == 1.0
    assert p["work_rate_m_per_min"] == 250.0


# -- degenerate matches ------------------------------------------------------------

def test_one_frame_match(app):
    client, root = app
    mid = upload(client, [frame(0, [person(1, "left", 0.0, 0.0, jersey="1"),
                                    person(2, "right", 5.0, 0.0, jersey="2")])])
    b = intensity(client, mid).json()
    assert b["duration_s"] == 0.04
    assert b["minutes"] == 1
    assert [p["id"] for p in b["players"]] == [1, 2]
    for p in b["players"]:
        assert p["distance_m"] == 0.0
        assert p["per_minute"] == [ZERO_ROW]
        assert p["work_rate_m_per_min"] == 0.0
    assert b["teams"]["left"]["distance_m"] == 0.0
    assert b["teams"]["right"]["distance_m"] == 0.0
    assert [e["id"] for e in b["top_distance"]] == [1, 2]


def test_empty_people_match(app):
    client, root = app
    mid = upload(client, [frame(i, []) for i in range(25)])
    b = intensity(client, mid).json()
    assert b["duration_s"] == 1.0
    assert b["minutes"] == 1
    assert b["players"] == []
    assert b["top_distance"] == [] and b["top_sprint"] == []
    for team in b["teams"].values():
        assert team["distance_m"] == 0.0
        assert team["zones_m"] == {z: 0.0 for z in ZONES}
        assert team["per_minute"] == [ZERO_ROW]
        assert team["work_rate_m_per_min"] == 0.0


def test_ball_is_ignored(app):
    client, root = app
    frames = [frame(i, [], ball={"box": [0, 0, 4, 4], "xy": [0.0, 0.0]})
              for i in range(25)]
    mid = upload(client, frames)
    b = intensity(client, mid).json()
    assert b["players"] == []
    assert b["teams"]["left"]["distance_m"] == 0.0
    assert b["teams"]["right"]["distance_m"] == 0.0


# -- idempotence, caching, mutation, discovery -------------------------------------

def test_idempotent_response(app):
    client, root, mid = canonical_upload(app)
    r1 = intensity(client, mid)
    r2 = intensity(client, mid)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()
    # pinned to the hand values, not just to each other
    assert r1.json()["teams"]["left"]["distance_m"] == 239.2
    assert r1.json()["teams"]["right"]["distance_m"] == 1136.2
    assert r1.json()["players"][0]["distance_m"] == 59.8


def test_result_cached_to_file(app):
    client, root, mid = canonical_upload(app)
    r = intensity(client, mid)
    assert r.status_code == 200
    path = root / mid / "features" / "intensity.json"
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == r.json()


def test_stale_cache_is_recomputed(app):
    client, root = app
    mid = upload(client, fresh_frames(0.8))
    r1 = intensity(client, mid)
    assert r1.status_code == 200, r1.text
    assert r1.json()["teams"]["left"]["distance_m"] == 0.8
    cache = root / mid / "features" / "intensity.json"
    assert cache.is_file()
    # replace the recording (same match folder) with a stationary variant
    frames_path = root / mid / "frames.jsonl"
    frames_path.write_bytes(recording(fresh_frames(0.0)))
    st = frames_path.stat()
    os.utime(cache, (st.st_mtime - 10, st.st_mtime - 10))
    # a fresh app (no frame cache in the context) must see the stale cache
    client2 = TestClient(create_library_app(root))
    r2 = intensity(client2, mid)
    assert r2.status_code == 200, r2.text
    b = r2.json()
    assert b["teams"]["left"]["distance_m"] == 0.0
    assert b["players"][0]["distance_m"] == 0.0
    assert b["players"][0]["work_rate_m_per_min"] == 0.0
    assert json.loads(cache.read_text(encoding="utf-8")) == b


def test_frames_not_mutated(app):
    client, root, mid = canonical_upload(app)
    before = client.get("/api/matches/%s/frames" % mid).json()
    intensity(client, mid)
    intensity(client, mid, start="0")
    after = client.get("/api/matches/%s/frames" % mid).json()
    assert before == after


def test_feature_is_listed(app):
    client, root = app
    r = client.get("/api/features")
    assert r.status_code == 200
    feature = next(f for f in r.json()["features"] if f["name"] == "intensity")
    assert feature["ok"] is True


# -- performance -------------------------------------------------------------------

def test_performance_25000_frames(app):
    client, root = app
    r = client.post("/api/matches",
                    files={"file": ("big.jsonl", perf_recording(),
                                    "application/json")},
                    data={"name": "big"})
    assert r.status_code == 201, r.text
    mid = r.json()["id"]
    t0 = time.monotonic()
    r = intensity(client, mid)
    elapsed = time.monotonic() - t0
    assert r.status_code == 200, r.text
    assert elapsed < 15.0, "25,000-frame match took %.1f s (budget 15 s)" % elapsed
    b = r.json()
    assert b["duration_s"] == 1000.0
    assert b["minutes"] == 17
    assert len(b["players"]) == 22
    assert all(len(p["per_minute"]) == 17 for p in b["players"])
    assert all(len(t["per_minute"]) == 17 for t in b["teams"].values())
    assert len(b["top_distance"]) == 10
    assert len(b["top_sprint"]) == 10
    dists = [e["distance_m"] for e in b["top_distance"]]
    assert dists == sorted(dists, reverse=True)
    assert all(d >= 0.0 for d in dists)
