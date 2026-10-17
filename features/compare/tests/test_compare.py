"""Tests for the compare feature (spec: features/compare/spec.md).

Every expected value below is computed by hand from the spec.  The tests
only exercise the HTTP surface of the library app; the recordings are
built in code.  Every moving track uses a constant displacement per
appearance, so the jitter smoothing the library app applies to the cached
summary (smooth=True) is the identity on these recordings and the hand
values apply unchanged.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from sideline.library import create_library_app  # noqa: E402

METRIC_NAMES = [
    "duration_s",
    "tracked_people",
    "team_width_left_m",
    "team_width_right_m",
    "team_length_left_m",
    "team_length_right_m",
    "avg_distance_m",
    "ball_found_pct",
]


# -- fixture builders --------------------------------------------------------

def person(pid, team, x, y, role="player", jersey="1", xy=True):
    return {"id": pid, "role": role, "team": team, "jersey": jersey,
            "box": [0, 0, 10, 10], "xy": [x, y] if xy else None}


def make_frame(n, people, ball=None):
    return {"frame": n, "delay": 0, "people": people, "ball": ball,
            "colors": {"left": "#d03030", "right": "#2050c0"}}


def recording(frames):
    return ("\n".join(json.dumps(f) for f in frames) + "\n").encode("utf-8")


def dense_frames(n, at, ball_frames=()):
    """n frames numbered 0..n-1; frame f carries the people list at[f] when
    present and a placed ball when f is in ball_frames."""
    ball_frames = set(ball_frames)
    out = []
    for i in range(n):
        ball = {"box": [0, 0, 4, 4], "xy": [0.0, 0.0]} if i in ball_frames else None
        out.append(make_frame(i, at.get(i, []), ball))
    return out


def match_a_frames():
    """100 frames (0..99, 4.0 s).  People appear every 5 frames
    (k = 0..19: 20 appearances, 19 steps of gap 5, t = 0.2 s each):

      left   id 1: x = 0 + 0.25k,  y = 0   -> 19 * 0.25 = 4.75 m (v = 1.25)
      left   id 2: x = 2 + 0.25k,  y = 5   -> 4.75 m
      left   id 3: x = 4 + 0.25k,  y = 10  -> 4.75 m
      right  id 4: x = 30 + 0.5k,  y = 0   -> 19 * 0.5 = 9.5 m (v = 2.5)
      right  id 5: x = 31 + 0.5k,  y = 5   -> 9.5 m

    Ball placed in the 80 frames with f % 5 != 2 -> 80.0 %.
    Left: 20 qualifying frames, width 10.0 (y 0..10), length 4.0 (x 0..4).
    Right: only 2 positioned players per frame -> no qualifying frame ->
    width null, length null.  Tracked people 5; total distance
    33.25 -> average 6.65.
    """
    at = {}
    for k in range(20):
        at[5 * k] = [
            person(1, "left", 0.25 * k, 0.0, jersey="1"),
            person(2, "left", 2.0 + 0.25 * k, 5.0, jersey="2"),
            person(3, "left", 4.0 + 0.25 * k, 10.0, jersey="3"),
            person(4, "right", 30.0 + 0.5 * k, 0.0, jersey="4"),
            person(5, "right", 31.0 + 0.5 * k, 5.0, jersey="5"),
        ]
    return dense_frames(100, at, ball_frames=[f for f in range(100)
                                              if f % 5 != 2])


def match_b_frames():
    """100 frames (0..99, 4.0 s).  People appear every 5 frames
    (k = 0..19: 20 appearances, 19 steps of gap 5):

      left   id 1: x = 0 + 0.5k,   y = 0   -> 19 * 0.5 = 9.5 m (v = 2.5)
      left   id 2: x = 1 + 0.5k,   y = 10  -> 9.5 m
      left   id 3: x = 2 + 0.5k,   y = 20  -> 9.5 m
      right  id 4: x = 40 + 0.25k, y = 0   -> 19 * 0.25 = 4.75 m (v = 1.25)
      right  id 5: x = 41 + 0.25k, y = 2   -> 4.75 m
      right  id 6: x = 42 + 0.25k, y = 4   -> 4.75 m

    Ball placed in the 20 frames with f % 5 == 0 -> 20.0 %.
    Left: width 20.0 (y 0..20), length 2.0 (x 0..2).
    Right: width 4.0 (y 0..4), length 2.0 (x 40..42).
    Tracked people 6; total distance 42.75 -> average 7.125 -> 7.12.
    """
    at = {}
    for k in range(20):
        at[5 * k] = [
            person(1, "left", 0.5 * k, 0.0, jersey="1"),
            person(2, "left", 1.0 + 0.5 * k, 10.0, jersey="2"),
            person(3, "left", 2.0 + 0.5 * k, 20.0, jersey="3"),
            person(4, "right", 40.0 + 0.25 * k, 0.0, jersey="4"),
            person(5, "right", 41.0 + 0.25 * k, 2.0, jersey="5"),
            person(6, "right", 42.0 + 0.25 * k, 4.0, jersey="6"),
        ]
    return dense_frames(100, at, ball_frames=[f for f in range(100)
                                              if f % 5 == 0])


def match_c_frames():
    """25 frames (1.0 s), no people.  Ball placed in frames 0..11 ->
    12 / 25 = 48.0 %.  Tracked people 0; every width/length null;
    average distance 0.0 (no players).
    """
    return dense_frames(25, {}, ball_frames=range(12))


def match_d_frames():
    """5 recorded frames numbered 0, 1, 2, 10, 11.  Only 5 frames exist, so
    the duration is 5 / 25 = 0.2 s (the 0..11 span would give 12 / 25 =
    0.48 s).  One left player standing still at (0, 0) -> 0.0 m.  Ball
    placed in frames 0, 1, 2 -> 3 / 5 = 60.0 %.
    """
    out = []
    for n in (0, 1, 2, 10, 11):
        ball = {"box": [0, 0, 4, 4], "xy": [0.0, 0.0]} if n in (0, 1, 2) else None
        out.append(make_frame(n, [person(1, "left", 0.0, 0.0, jersey="1")], ball))
    return out


def match_e_frames():
    """25 frames (1.0 s).  At frames 0, 5, 10, 15, 20 (5 appearances, 4
    steps of gap 5, t = 0.2 s each):

      left   id 1:  (0, 0) standing still          -> 0.0 m
      ref    id 9:  x = k,  team null              -> 4 * 1.0 = 4.0 m (v = 5.0)
      (null) id 10: x = 0.5k, role player          -> 4 * 0.5 = 2.0 m
      left   id 11: xy null in all 5 frames        -> 0.0 m

    All four distinct ids are tracked: 4.  Total distance 6.0 ->
    average 1.5.  No team reaches 3 positioned players -> every width and
    length null.  Ball never placed -> 0.0 %.
    """
    at = {}
    for k in range(5):
        at[5 * k] = [
            person(1, "left", 0.0, 0.0, jersey="1"),
            person(9, None, 1.0 * k, 0.0, role="referee", jersey="9"),
            person(10, None, 0.5 * k, 0.0, jersey="10"),
            person(11, "left", 0.0, 0.0, jersey="11", xy=False),
        ]
    return dense_frames(25, at)


def match_f_frames():
    """10 frames (0.4 s), no people.  The ball is placed (valid xy) in
    frames 0..3, found but not placed (xy null) in frames 4, 5 and missing
    in frames 6..9 -> only the 4 valid positions count: 4 / 10 = 40.0 %.
    """
    out = []
    for n in range(10):
        if n <= 3:
            ball = {"box": [0, 0, 4, 4], "xy": [0.0, 0.0]}
        elif n <= 5:
            ball = {"box": [0, 0, 4, 4], "xy": None}
        else:
            ball = None
        out.append(make_frame(n, [], ball))
    return out


def perf_frames(dx, n=25000):
    """n frames (0..n-1) with 11 left + 11 right people in every frame.

    Left player j (j = 0..10): x = -52 + j + dx * f, y = 2j.  Right player
    j: x = 41 + j + dx * f, y = 2j - 20.  The per-frame displacement is the
    constant dx (exact in binary for dx = 1/16 and dx = 1/32), so every
    person makes 24,999 counted steps of dx metres and the smoothing is
    the identity: dx = 1/16 -> 1562.4375 m (1562.44); dx = 1/32 ->
    781.21875 m (781.22).  Both teams in both matches: width 20.0 m,
    length 10.0 m, in every frame.  Ball placed in the even frames ->
    12,500 / 25,000 = 50.0 %.  About 62 MB per recording.
    """
    lines = []
    for f in range(n):
        people = []
        for i in range(22):
            left = i < 11
            j = i if left else i - 11
            x = (-52.0 if left else 41.0) + j + dx * f
            y = 2.0 * j if left else 2.0 * j - 20.0
            people.append({"id": i + 1, "role": "player",
                           "team": "left" if left else "right",
                           "jersey": str(j + 1), "box": [0, 0, 10, 10],
                           "xy": [x, y]})
        ball = {"box": [0, 0, 4, 4], "xy": [0.0, 0.0]} if f % 2 == 0 else None
        lines.append(json.dumps({"frame": f, "delay": 0, "people": people,
                                 "ball": ball,
                                 "colors": {"left": "#d03030",
                                            "right": "#2050c0"}}))
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


def compare(client, params):
    return client.get("/api/compare", params=params)


def compare_ab(client, a, b):
    return compare(client, {"a": a, "b": b})


def rows(body):
    return {m["name"]: m for m in body["metrics"]}


def upload_ab(app):
    client, root = app
    mid_a = upload(client, match_a_frames(), "match_a")
    mid_b = upload(client, match_b_frames(), "match_b")
    return client, root, mid_a, mid_b


# -- validation ---------------------------------------------------------------

def test_missing_parameters_are_400(app):
    client, root, mid_a, mid_b = upload_ab(app)
    for params in ({}, {"a": mid_a}, {"b": mid_b}):
        r = compare(client, params)
        assert r.status_code == 400, (params, r.text)


def test_empty_parameter_is_400(app):
    client, root, mid_a, mid_b = upload_ab(app)
    assert compare(client, {"a": "", "b": mid_b}).status_code == 400
    assert compare(client, {"a": mid_a, "b": ""}).status_code == 400


def test_same_id_is_400(app):
    client, root, mid_a, mid_b = upload_ab(app)
    assert compare_ab(client, mid_a, mid_a).status_code == 400


def test_same_unknown_id_is_400_not_404(app):
    # equality is checked before the ids are looked up
    assert compare_ab(client, "m_ffffffffff", "m_ffffffffff").status_code == 400


def test_same_malformed_id_is_400(app):
    assert compare_ab(client, "m_123", "m_123").status_code == 400


def test_unknown_a_is_404(app):
    client, root, mid_a, mid_b = upload_ab(app)
    assert compare_ab(client, "m_0000000000", mid_b).status_code == 404


def test_unknown_b_is_404(app):
    client, root, mid_a, mid_b = upload_ab(app)
    assert compare_ab(client, mid_a, "m_0000000001").status_code == 404


def test_malformed_ids_are_404(app):
    client, root, mid_a, mid_b = upload_ab(app)
    for bad in ("m_123", "M_0123456789", "compare", "m_012345678901"):
        assert compare_ab(client, bad, mid_b).status_code == 404, bad
        assert compare_ab(client, mid_a, bad).status_code == 404, bad


# -- canonical matches A and B --------------------------------------------------

def test_envelope_and_metric_order(app):
    client, root, mid_a, mid_b = upload_ab(app)
    r = compare_ab(client, mid_a, mid_b)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["a"] == mid_a and b["b"] == mid_b
    assert [m["name"] for m in b["metrics"]] == METRIC_NAMES
    for m in b["metrics"]:
        assert set(m) == {"name", "a", "b", "diff"}


def test_duration_row(app):
    client, root, mid_a, mid_b = upload_ab(app)
    m = rows(compare_ab(client, mid_a, mid_b).json())["duration_s"]
    assert m == {"name": "duration_s", "a": 4.0, "b": 4.0, "diff": 0.0}


def test_tracked_people_row(app):
    client, root, mid_a, mid_b = upload_ab(app)
    m = rows(compare_ab(client, mid_a, mid_b).json())["tracked_people"]
    assert m == {"name": "tracked_people", "a": 5, "b": 6, "diff": -1}
    assert isinstance(m["diff"], int)


def test_team_width_rows(app):
    client, root, mid_a, mid_b = upload_ab(app)
    m = rows(compare_ab(client, mid_a, mid_b).json())
    assert m["team_width_left_m"] == {"name": "team_width_left_m",
                                      "a": 10.0, "b": 20.0, "diff": -10.0}
    assert m["team_width_right_m"] == {"name": "team_width_right_m",
                                       "a": None, "b": 4.0, "diff": None}


def test_team_length_rows(app):
    client, root, mid_a, mid_b = upload_ab(app)
    m = rows(compare_ab(client, mid_a, mid_b).json())
    assert m["team_length_left_m"] == {"name": "team_length_left_m",
                                       "a": 4.0, "b": 2.0, "diff": 2.0}
    assert m["team_length_right_m"] == {"name": "team_length_right_m",
                                        "a": None, "b": 2.0, "diff": None}


def test_avg_distance_row(app):
    client, root, mid_a, mid_b = upload_ab(app)
    m = rows(compare_ab(client, mid_a, mid_b).json())["avg_distance_m"]
    # A: (3 * 4.75 + 2 * 9.5) / 5 = 6.65; B: 42.75 / 6 = 7.125 -> 7.12
    assert m == {"name": "avg_distance_m", "a": 6.65, "b": 7.12, "diff": -0.47}


def test_ball_found_row(app):
    client, root, mid_a, mid_b = upload_ab(app)
    m = rows(compare_ab(client, mid_a, mid_b).json())["ball_found_pct"]
    assert m == {"name": "ball_found_pct", "a": 80.0, "b": 20.0, "diff": 60.0}


def test_reversed_arguments_swap_values(app):
    client, root, mid_a, mid_b = upload_ab(app)
    r = compare_ab(client, mid_b, mid_a)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["a"] == mid_b and b["b"] == mid_a
    m = rows(b)
    assert m["tracked_people"] == {"name": "tracked_people",
                                   "a": 6, "b": 5, "diff": 1}
    assert m["team_width_left_m"] == {"name": "team_width_left_m",
                                      "a": 20.0, "b": 10.0, "diff": 10.0}
    assert m["team_width_right_m"] == {"name": "team_width_right_m",
                                       "a": 4.0, "b": None, "diff": None}
    assert m["team_length_left_m"] == {"name": "team_length_left_m",
                                       "a": 2.0, "b": 4.0, "diff": -2.0}
    assert m["team_length_right_m"] == {"name": "team_length_right_m",
                                        "a": 2.0, "b": None, "diff": None}
    assert m["avg_distance_m"] == {"name": "avg_distance_m",
                                   "a": 7.12, "b": 6.65, "diff": 0.47}
    assert m["ball_found_pct"] == {"name": "ball_found_pct",
                                   "a": 20.0, "b": 80.0, "diff": -60.0}


def test_compare_reflects_the_analytics_summary(app):
    # every row must equal the value of the app's own cached summary for
    # that match (the wiring, not just the hand values)
    client, root, mid_a, mid_b = upload_ab(app)
    m = rows(compare_ab(client, mid_a, mid_b).json())
    for side, mid in (("a", mid_a), ("b", mid_b)):
        s = client.get("/api/matches/%s/analytics" % mid).json()
        assert m["duration_s"][side] == s["duration_s"]
        assert m["tracked_people"][side] == len(s["players"])
        assert m["team_width_left_m"][side] == s["teams"]["left"]["avg_width"]
        assert m["team_width_right_m"][side] == s["teams"]["right"]["avg_width"]
        assert m["team_length_left_m"][side] == s["teams"]["left"]["avg_length"]
        assert m["team_length_right_m"][side] == s["teams"]["right"]["avg_length"]
        distances = [p["distance_m"] for p in s["players"]]
        avg = round(sum(distances) / len(distances), 2) if distances else 0.0
        assert m["avg_distance_m"][side] == avg
        pct = round(100.0 * s["ball"]["frames_with_ball"] / s["frame_count"], 2)
        assert m["ball_found_pct"][side] == pct
    for name in METRIC_NAMES:
        va, vb = m[name]["a"], m[name]["b"]
        expected = va - vb if name == "tracked_people" \
            else (None if va is None or vb is None else round(va - vb, 2))
        assert m[name]["diff"] == expected, name


# -- empty, null and special matches ---------------------------------------------

def test_match_without_people(app):
    client, root = app
    mid_a = upload(client, match_a_frames(), "match_a")
    mid_c = upload(client, match_c_frames(), "match_c")
    m = rows(compare_ab(client, mid_c, mid_a).json())
    assert m["duration_s"] == {"name": "duration_s",
                               "a": 1.0, "b": 4.0, "diff": -3.0}
    assert m["tracked_people"] == {"name": "tracked_people",
                                   "a": 0, "b": 5, "diff": -5}
    assert m["team_width_left_m"] == {"name": "team_width_left_m",
                                      "a": None, "b": 10.0, "diff": None}
    assert m["team_width_right_m"] == {"name": "team_width_right_m",
                                       "a": None, "b": None, "diff": None}
    assert m["team_length_left_m"] == {"name": "team_length_left_m",
                                       "a": None, "b": 4.0, "diff": None}
    assert m["team_length_right_m"] == {"name": "team_length_right_m",
                                        "a": None, "b": None, "diff": None}
    assert m["avg_distance_m"] == {"name": "avg_distance_m",
                                   "a": 0.0, "b": 6.65, "diff": -6.65}
    assert m["ball_found_pct"] == {"name": "ball_found_pct",
                                   "a": 48.0, "b": 80.0, "diff": -32.0}


def test_tracked_people_includes_referees_unassigned_and_unpositioned(app):
    client, root = app
    mid_a = upload(client, match_a_frames(), "match_a")
    mid_e = upload(client, match_e_frames(), "match_e")
    m = rows(compare_ab(client, mid_e, mid_a).json())
    # the E side tracks 4 distinct ids: player 1, referee 9, unassigned 10
    # and unpositioned 11
    assert m["tracked_people"] == {"name": "tracked_people",
                                   "a": 4, "b": 5, "diff": -1}
    # (0.0 + 4.0 + 2.0 + 0.0) / 4 = 1.5
    assert m["avg_distance_m"] == {"name": "avg_distance_m",
                                   "a": 1.5, "b": 6.65, "diff": -5.15}
    assert m["ball_found_pct"] == {"name": "ball_found_pct",
                                   "a": 0.0, "b": 80.0, "diff": -80.0}
    assert m["duration_s"] == {"name": "duration_s",
                               "a": 1.0, "b": 4.0, "diff": -3.0}
    # only one positioned left player (id 11 has no valid position)
    assert m["team_width_left_m"]["a"] is None


def test_ball_found_but_not_placed_does_not_count(app):
    client, root = app
    mid_a = upload(client, match_a_frames(), "match_a")
    mid_f = upload(client, match_f_frames(), "match_f")
    m = rows(compare_ab(client, mid_f, mid_a).json())
    # 4 valid positions out of 10 frames; xy-null and null balls do not count
    assert m["ball_found_pct"] == {"name": "ball_found_pct",
                                   "a": 40.0, "b": 80.0, "diff": -40.0}
    assert m["duration_s"] == {"name": "duration_s",
                               "a": 0.4, "b": 4.0, "diff": -3.6}


def test_duration_counts_recorded_frames_not_the_span(app):
    client, root = app
    mid_a = upload(client, match_a_frames(), "match_a")
    mid_d = upload(client, match_d_frames(), "match_d")
    m = rows(compare_ab(client, mid_d, mid_a).json())
    # 5 recorded frames -> 5 / 25 = 0.2 s, not 12 / 25 = 0.48 s from the span
    assert m["duration_s"] == {"name": "duration_s",
                               "a": 0.2, "b": 4.0, "diff": -3.8}
    assert m["tracked_people"] == {"name": "tracked_people",
                                   "a": 1, "b": 5, "diff": -4}
    assert m["ball_found_pct"] == {"name": "ball_found_pct",
                                   "a": 60.0, "b": 80.0, "diff": -20.0}
    assert m["avg_distance_m"] == {"name": "avg_distance_m",
                                   "a": 0.0, "b": 6.65, "diff": -6.65}


# -- idempotence, mutation, discovery ---------------------------------------------

def test_idempotent_response(app):
    client, root, mid_a, mid_b = upload_ab(app)
    r1 = compare_ab(client, mid_a, mid_b)
    r2 = compare_ab(client, mid_a, mid_b)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()
    m = rows(r1.json())
    assert m["avg_distance_m"] == {"name": "avg_distance_m",
                                   "a": 6.65, "b": 7.12, "diff": -0.47}


def test_extra_query_parameters_are_ignored(app):
    client, root, mid_a, mid_b = upload_ab(app)
    r = compare(client, {"a": mid_a, "b": mid_b, "team": "left", "x": "1"})
    assert r.status_code == 200, r.text
    assert r.json() == compare_ab(client, mid_a, mid_b).json()


def test_frames_not_mutated(app):
    client, root, mid_a, mid_b = upload_ab(app)
    before = [client.get("/api/matches/%s/frames" % mid).json()
              for mid in (mid_a, mid_b)]
    compare_ab(client, mid_a, mid_b)
    compare_ab(client, mid_b, mid_a)
    after = [client.get("/api/matches/%s/frames" % mid).json()
             for mid in (mid_a, mid_b)]
    assert before == after


def test_feature_is_listed(app):
    client, root = app
    r = client.get("/api/features")
    assert r.status_code == 200
    feature = next(f for f in r.json()["features"] if f["name"] == "compare")
    assert feature["ok"] is True


# -- performance ---------------------------------------------------------------------

def test_performance_two_25000_frame_matches(app):
    client, root = app
    ids = []
    for name, dx in (("perf_a", 1 / 16), ("perf_b", 1 / 32)):
        r = client.post("/api/matches",
                        files={"file": (name + ".jsonl", perf_frames(dx),
                                        "application/json")},
                        data={"name": name})
        assert r.status_code == 201, r.text
        ids.append(r.json()["id"])
    t0 = time.monotonic()
    r = compare_ab(client, ids[0], ids[1])
    elapsed = time.monotonic() - t0
    assert r.status_code == 200, r.text
    assert elapsed < 45.0, \
        "comparing two 25,000-frame matches took %.1f s (budget 45 s)" % elapsed
    m = rows(r.json())
    assert m["duration_s"] == {"name": "duration_s",
                               "a": 1000.0, "b": 1000.0, "diff": 0.0}
    assert m["tracked_people"] == {"name": "tracked_people",
                                   "a": 22, "b": 22, "diff": 0}
    assert m["team_width_left_m"] == {"name": "team_width_left_m",
                                      "a": 20.0, "b": 20.0, "diff": 0.0}
    assert m["team_width_right_m"] == {"name": "team_width_right_m",
                                       "a": 20.0, "b": 20.0, "diff": 0.0}
    assert m["team_length_left_m"] == {"name": "team_length_left_m",
                                       "a": 10.0, "b": 10.0, "diff": 0.0}
    assert m["team_length_right_m"] == {"name": "team_length_right_m",
                                        "a": 10.0, "b": 10.0, "diff": 0.0}
    # 24,999 steps of 1/16 m -> 1562.4375 -> 1562.44; of 1/32 m -> 781.22
    assert m["avg_distance_m"] == {"name": "avg_distance_m",
                                   "a": 1562.44, "b": 781.22, "diff": 781.22}
    assert m["ball_found_pct"] == {"name": "ball_found_pct",
                                   "a": 50.0, "b": 50.0, "diff": 0.0}
