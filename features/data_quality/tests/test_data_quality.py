"""Acceptance tests for the data_quality feature (spec: features/data_quality/spec.md).

Recordings are built in code by the small helpers below; every expected value is
derived by hand from the spec (see the spec's worked examples).  The suite assumes
the feature registry of tasks/feature_registry.md (create_library_app mounting the
feature router under /api/matches/{match_id}, /api/features, FeatureContext with
require/frames/cache_json); that registry is a declared dependency of the spec.
"""
import copy
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

import pytest
from fastapi.testclient import TestClient

from sideline import library
from sideline.features import data_quality


def person(i, x, y, team="left", role="player", jersey="1", xy="__auto__", id_=None):
    return {"id": i if id_ is None else id_, "role": role, "team": team,
            "jersey": jersey, "box": [0, 0, 10, 20],
            "xy": [x, y] if xy == "__auto__" else xy}


def frame(n, people, ball=None):
    return {"frame": n, "delay": 0, "people": [dict(p) for p in people],
            "ball": dict(ball) if ball is not None else None, "colors": None}


def to_bytes(frames):
    return ("\n".join(json.dumps(f) for f in frames) + "\n").encode("utf-8")


def make_client(tmp_path):
    return TestClient(library.create_library_app(tmp_path / "store"))


def upload(client, data, name="game"):
    r = client.post("/api/matches",
                    files={"file": (name + ".jsonl", data, "application/json")},
                    data={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def get_quality(client, mid, params=None):
    return client.get("/api/matches/%s/quality" % mid, params=params)


# --- fixtures ---------------------------------------------------------------

BALL = {"box": [5, 5, 6, 6], "xy": [1.0, -2.0]}


def basic_people():
    """The spec's canonical 25-person frame (see the spec's canonical fixture)."""
    people = []
    for i in range(1, 12):
        people.append(person(i, -38 + 2 * (i - 1), (i - 1) - 5, jersey=str(i)))
    for j in range(1, 12):
        people.append(person(11 + j, 20 + 2 * (j - 1), 5 - (j - 1),
                             team="right", jersey=str(j)))
    people.append(person(120, -48.0, 0.0, role="goalkeeper", jersey="1"))
    people.append(person(130, 48.0, 0.0, team="right", role="goalkeeper", jersey="1"))
    people.append(person(900, 0.0, 8.0, team=None, role="referee", jersey=None))
    assert len(people) == 25
    return people


def basic_recording():
    return [frame(f, basic_people(), ball=BALL) for f in range(50)]


def fragment_recording():
    """8 tracks over 75 frames (spec's Rule 6 examples).

    id 2 ends at frame 24 at (10, 3); id 1 starts at frame 25 at (11.6, 5.2):
    0.04 s, distance sqrt(1.6^2 + 2.2^2) ~= 2.72 -> fragment.  id 3 starts at
    frame 25 at (13.2, 5.4): distance 4.0 -> not a fragment.  id 4 ends at
    frame 0 at (-10, -2); id 5 starts at frame 37 (1.48 s) at (-10.6, -2.8),
    distance 1.0 -> fragment; id 6 starts at frame 38 (1.52 s) at
    (-10.5, -2.7) -> too late.  id 7 starts at frame 10, while id 8 (which
    ends at frame 24 at (10, 3)) is still alive -> an overlap, not a
    fragment, even though 0.56 s / 0.71 m fit the window.
    """
    tracks = {
        2: (0, 24, (10.0, 3.0)),
        1: (25, 74, (11.6, 5.2)),
        3: (25, 74, (13.2, 5.4)),
        4: (0, 0, (-10.0, -2.0)),
        5: (37, 74, (-10.6, -2.8)),
        6: (38, 74, (-10.5, -2.7)),
        7: (10, 49, (10.5, 3.5)),
        8: (20, 24, (10.0, 3.0)),
    }
    frames = []
    for f in range(75):
        people = [person(tid, x, y, jersey=str(tid))
                  for tid, (a, b, (x, y)) in tracks.items() if a <= f <= b]
        frames.append(frame(f, people, ball=None))
    return frames


def gap_recording():
    """150 frames: 8 left players, frames 25..49 empty, only id 1 after 124."""
    eight = [person(i, -18 + 2 * (i - 1), 2 * (i - 1), jersey=str(i))
             for i in range(1, 9)]
    frames = []
    for f in range(150):
        if 25 <= f <= 49:
            people = []
        elif f >= 125:
            people = [eight[0]]
        else:
            people = list(eight)
        frames.append(frame(f, people, ball={"box": [1, 1, 3, 3], "xy": [0.0, 0.0]}))
    return frames


def rounding_recording():
    """3 frames pinning 'round the exact fraction once' (Rule 12) and the 50.0
    jersey threshold boundary: 4 jersey reads over 8 tracked people = 50.0%,
    while the per-frame rounded percents (100.0, 33.3, 33.3) average to 55.5.
    id 3 starts at frame 1, the frame in which id 2 ends (0 s apart, allowed
    by Rule 6), but ~39.2 m away -> not a fragment; id 4 starts ~46 m from
    every track's end position -> not a fragment either.
    """
    f0 = [person(1, 0.0, 0.0, jersey="1"), person(2, 5.0, 2.0, jersey="2")]
    f1 = [person(1, 0.1, 0.0, jersey="1"), person(2, 5.1, 2.1, jersey=None),
          person(3, 40.0, 20.0, jersey=None)]
    f2 = [person(1, 0.2, 0.0, jersey=None), person(3, 40.1, 20.1, jersey="3"),
          person(4, -41.0, -21.0, jersey=None)]
    return [frame(0, f0, ball={"box": [1, 1, 3, 3], "xy": [2.0, -1.0]}),
            frame(1, f1, ball=None),
            frame(2, f2, ball=None)]


def small_range_recording():
    """100 frames, 9 players, frames 40..54 (15 frames) empty: the range is
    reported but too short (< 25) for a warning or a grade penalty."""
    nine = [person(i, float(i), float(i - 5), jersey=str(i)) for i in range(1, 10)]
    frames = []
    for f in range(100):
        people = [] if 40 <= f <= 54 else list(nine)
        frames.append(frame(f, people, ball={"box": [2, 2, 4, 4], "xy": [10.0, 0.0]}))
    return frames


def untracked_recording():
    """30 frames in which the only people are found but not placed (xy null):
    zero tracked people, one range covering the whole match."""
    people = [person(1, 0.0, 0.0, xy=None), person(2, 1.0, 1.0, xy=None)]
    return [frame(f, people, ball=None) for f in range(30)]


def big_recording(n):
    """22 stable tracks for the performance test (11 per side)."""
    left = [person(i, -40 + 2 * (i - 1), (i - 1) - 5, jersey=str(i))
            for i in range(1, 12)]
    right = [person(11 + j, 20 + 2 * (j - 1), 5 - (j - 1), team="right", jersey=str(j))
             for j in range(1, 12)]
    return [frame(f, left + right, ball={"box": [0, 0, 5, 5], "xy": [0.0, 0.0]})
            for f in range(n)]


# --- route: the canonical fixture -------------------------------------------

def test_basic_route_document(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    r = get_quality(client, mid)
    assert r.status_code == 200, r.text
    # 25 tracked per frame -> 25.0; all 50 frames good -> share 1.0; the
    # referee (team null, jersey null) drops team and jersey to 24/25 = 96.0.
    assert r.json() == {
        "match_id": mid, "fps": 25, "first_frame": 0, "last_frame": 49,
        "frame_count": 50, "tracked_per_frame": 25.0, "good_frames_share": 1.0,
        "no_people_ranges": [], "fragment_tracks": 0, "team_assigned_pct": 96.0,
        "jersey_read_pct": 96.0, "ball_found_pct": 100.0,
        "warnings": [], "grade": "A"}


def test_unknown_query_params_are_ignored(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    r = get_quality(client, mid, params={"team": "left", "x": "1"})
    assert r.status_code == 200
    assert r.json() == get_quality(client, mid).json()


def test_unknown_match_is_404(tmp_path):
    client = make_client(tmp_path)
    r = client.get("/api/matches/m_0000000000/quality")
    assert r.status_code == 404
    assert r.json() == {"detail": {"error": "match not found"}}


@pytest.mark.parametrize("bad_id", ["m_short", "m_123456789a1", "nope", "m_"])
def test_malformed_match_id_is_404(tmp_path, bad_id):
    client = make_client(tmp_path)
    r = client.get("/api/matches/%s/quality" % bad_id)
    assert r.status_code == 404
    assert r.json() == {"detail": {"error": "match not found"}}


def test_repeated_gets_are_idempotent(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    r1, r2 = get_quality(client, mid), get_quality(client, mid)
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    # pinned to the spec's numbers, not just to the first response
    assert r1.json()["team_assigned_pct"] == 96.0
    assert r2.json()["grade"] == "A"


def test_cache_file_written_and_matches_response(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    doc = get_quality(client, mid).json()
    cache = tmp_path / "store" / mid / "features" / "quality.json"
    assert cache.is_file()
    assert json.loads(cache.read_text(encoding="utf-8")) == doc


def test_cache_recomputed_when_frames_replaced(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    assert get_quality(client, mid).json()["last_frame"] == 49
    # the match is "replaced": the same 25 people for 100 frames
    path = tmp_path / "store" / mid / "frames.jsonl"
    time.sleep(0.02)
    path.write_bytes(to_bytes(basic_recording() + [
        frame(50 + i, basic_people(), ball=BALL) for i in range(50)]))
    time.sleep(0.02)
    again = make_client(tmp_path)  # fresh app: no in-memory frame LRU
    doc = get_quality(again, mid).json()
    assert doc["last_frame"] == 99 and doc["frame_count"] == 100
    assert doc["tracked_per_frame"] == 25.0 and doc["grade"] == "A"


def test_route_does_not_mutate_the_frames_file(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(basic_recording()))
    path = tmp_path / "store" / mid / "frames.jsonl"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    get_quality(client, mid)
    get_quality(client, mid, params={"team": "left"})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_feature_is_listed_in_api_features(tmp_path):
    client = make_client(tmp_path)
    r = client.get("/api/features")
    assert r.status_code == 200
    feats = r.json()["features"]
    assert "data_quality" in [f["name"] for f in feats]
    entry = [f for f in feats if f["name"] == "data_quality"][0]
    assert entry["ok"] is True and entry["error"] is None
    assert any("quality" in str(route) for route in entry["routes"])


# --- route: the edge-case recordings ----------------------------------------

def test_fragment_recording_route(tmp_path):
    rec = fragment_recording()
    # sanity: the fixture really has 8 tracks and the designed start/end frames
    ids = set()
    for f in rec:
        ids.update(p["id"] for p in f["people"])
    assert ids == {1, 2, 3, 4, 5, 6, 7, 8}
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    # 246 tracked people over 75 frames -> 3.28; no frame reaches 8; exactly
    # 2 fragments (id 1 and id 5, see the fixture's docstring); ball never found.
    doc = get_quality(client, mid)
    assert doc.status_code == 200
    assert doc.json() == {
        "match_id": mid, "fps": 25, "first_frame": 0, "last_frame": 74,
        "frame_count": 75, "tracked_per_frame": 3.28, "good_frames_share": 0.0,
        "no_people_ranges": [], "fragment_tracks": 2, "team_assigned_pct": 100.0,
        "jersey_read_pct": 100.0, "ball_found_pct": 0.0,
        "warnings": [
            "Only 0.0% of frames have 8 or more tracked people.",
            "2 tracks start within 1.5 s and 3 m of a track that just ended "
            "(possible id switches).",
            "The ball was found in only 0.0% of frames."],
        "grade": "D"}


def test_gap_recording_route(tmp_path):
    rec = gap_recording()
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    # 825 tracked over 150 frames -> 5.5; 100 good frames -> 0.667; one range
    # of exactly 25 frames (the warning threshold); checks 1 and 2 fail, the
    # other four pass -> C.
    doc = get_quality(client, mid)
    assert doc.status_code == 200
    assert doc.json() == {
        "match_id": mid, "fps": 25, "first_frame": 0, "last_frame": 149,
        "frame_count": 150, "tracked_per_frame": 5.5, "good_frames_share": 0.667,
        "no_people_ranges": [{"start": 25, "end": 49, "frames": 25}],
        "fragment_tracks": 0, "team_assigned_pct": 100.0,
        "jersey_read_pct": 100.0, "ball_found_pct": 100.0,
        "warnings": [
            "Only 66.7% of frames have 8 or more tracked people.",
            "No tracked people in frames 25..49 (25 frames) - the camera may "
            "be uncalibrated."],
        "grade": "C"}


def test_rounding_recording_route(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rounding_recording()))
    # 4 jersey reads of 8 tracked -> exactly 50.0 (a per-frame average of the
    # rounded 100.0/33.3/33.3 would give 55.5); 1 of 3 frames with the ball ->
    # 33.3; 8 tracked over 3 frames -> 2.67.  Checks 1 and 6 fail -> C, and
    # the jersey warning must NOT fire at exactly 50.0.
    doc = get_quality(client, mid)
    assert doc.status_code == 200
    assert doc.json() == {
        "match_id": mid, "fps": 25, "first_frame": 0, "last_frame": 2,
        "frame_count": 3, "tracked_per_frame": 2.67, "good_frames_share": 0.0,
        "no_people_ranges": [], "fragment_tracks": 0, "team_assigned_pct": 100.0,
        "jersey_read_pct": 50.0, "ball_found_pct": 33.3,
        "warnings": [
            "Only 0.0% of frames have 8 or more tracked people.",
            "The ball was found in only 33.3% of frames."],
        "grade": "C"}


def test_small_range_reported_without_warning(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(small_range_recording()))
    # 765 tracked over 100 frames -> 7.65; 85 good frames -> 0.85 (>= 0.8, no
    # warning); the 15-frame range is below the 25-frame warning threshold, so
    # no warning and no grade penalty -> A.
    doc = get_quality(client, mid)
    assert doc.status_code == 200
    assert doc.json() == {
        "match_id": mid, "fps": 25, "first_frame": 0, "last_frame": 99,
        "frame_count": 100, "tracked_per_frame": 7.65, "good_frames_share": 0.85,
        "no_people_ranges": [{"start": 40, "end": 54, "frames": 15}],
        "fragment_tracks": 0, "team_assigned_pct": 100.0,
        "jersey_read_pct": 100.0, "ball_found_pct": 100.0,
        "warnings": [], "grade": "A"}


def test_untracked_recording_route(tmp_path):
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(untracked_recording()))
    # zero tracked people: per-person percentages 0.0 without the two
    # team/jersey warnings (nothing to quote), one range over the whole match
    # (30 >= 25 -> warning), only check 3 passes -> D.
    doc = get_quality(client, mid)
    assert doc.status_code == 200
    assert doc.json() == {
        "match_id": mid, "fps": 25, "first_frame": 0, "last_frame": 29,
        "frame_count": 30, "tracked_per_frame": 0.0, "good_frames_share": 0.0,
        "no_people_ranges": [{"start": 0, "end": 29, "frames": 30}],
        "fragment_tracks": 0, "team_assigned_pct": 0.0, "jersey_read_pct": 0.0,
        "ball_found_pct": 0.0,
        "warnings": [
            "Only 0.0% of frames have 8 or more tracked people.",
            "No tracked people in frames 0..29 (30 frames) - the camera may "
            "be uncalibrated.",
            "The ball was found in only 0.0% of frames."],
        "grade": "D"}


def test_ball_found_but_not_placed_counts_as_found(tmp_path):
    nine = [person(i, float(i), float(i), jersey=str(i)) for i in range(1, 10)]
    not_placed = [frame(f, nine, ball={"box": [1, 1, 3, 3], "xy": None})
                  for f in range(3)]
    missing = [frame(f, nine, ball=None) for f in range(3)]
    client = make_client(tmp_path)
    m1 = upload(client, to_bytes(not_placed))
    m2 = upload(client, to_bytes(missing))
    d1 = get_quality(client, m1).json()
    d2 = get_quality(client, m2).json()
    assert d1["ball_found_pct"] == 100.0  # found in the image, not on the pitch
    assert d2["ball_found_pct"] == 0.0


def test_performance_25000_frames_within_budget(tmp_path):
    rec = big_recording(25000)
    assert len(rec) == 25000
    assert all(len(f["people"]) == 22 for f in rec)
    client = make_client(tmp_path)
    mid = upload(client, to_bytes(rec))
    t0 = time.monotonic()
    r = get_quality(client, mid)
    elapsed = time.monotonic() - t0
    assert r.status_code == 200
    assert elapsed < 30.0, "quality took %.1fs, budget is 30s" % elapsed
    doc = r.json()
    assert doc["frame_count"] == 25000 and doc["last_frame"] == 24999
    assert doc["tracked_per_frame"] == 22.0
    assert doc["good_frames_share"] == 1.0
    assert doc["no_people_ranges"] == [] and doc["fragment_tracks"] == 0
    assert doc["team_assigned_pct"] == 100.0
    assert doc["jersey_read_pct"] == 100.0 and doc["ball_found_pct"] == 100.0
    assert doc["warnings"] == [] and doc["grade"] == "A"


# --- pure functions -----------------------------------------------------------

def test_quality_report_empty_frames():
    assert data_quality.quality_report([]) == {
        "fps": 25, "first_frame": None, "last_frame": None, "frame_count": 0,
        "tracked_per_frame": 0.0, "good_frames_share": 0.0,
        "no_people_ranges": [], "fragment_tracks": 0,
        "team_assigned_pct": 0.0, "jersey_read_pct": 0.0, "ball_found_pct": 0.0,
        "warnings": [], "grade": "D"}


def test_tracked_people_rules():
    f = {"frame": 0, "delay": 0, "ball": None, "colors": None, "people": [
        person(1, 0.5, 1.5),
        person(1, 9.0, 9.0),                       # duplicate id: first wins
        person(2, 0.0, 0.0, xy=None),              # found, not placed
        person(3, 0.0, 0.0, xy=[1.0]),             # one element
        person(4, float("inf"), 0.0),              # non-finite
        person(7, 2.0, 2.0, id_="7"),              # string id
        person(True, 3.0, 3.0),                    # bool id is not an int track
        person(5, 1.0, 2.0, team=None, role="referee", jersey=None)]}
    # a referee with a valid position is still a tracked person
    assert data_quality.tracked_people(f) == [(1, 0.5, 1.5), (5, 1.0, 2.0)]


def test_duplicate_frame_numbers_last_wins():
    eight = [person(i, float(i), 0.5 * i, jersey=str(i)) for i in range(1, 9)]
    two = [person(1, 0.0, 0.0), person(2, 1.0, 0.0)]
    frames = [frame(0, two), frame(0, eight), frame(1, eight)]
    assert sum(1 for f in frames if f["frame"] == 0) == 2
    doc = data_quality.quality_report(frames)
    # last frame-0 wins: 8 + 8 tracked over 2 frames -> 8.0, not 5.0
    assert doc["frame_count"] == 2
    assert doc["tracked_per_frame"] == 8.0
    assert doc["good_frames_share"] == 1.0


def test_quality_report_does_not_mutate_inputs():
    frames = basic_recording()
    snapshot = copy.deepcopy(frames)
    data_quality.quality_report(frames)
    for f in frames:
        data_quality.tracked_people(f)
    assert frames == snapshot


def test_fps_changes_the_fragment_window():
    frames = [frame(0, [person(1, 0.0, 0.0)]),
              frame(38, [person(2, 0.0, 0.5)])]
    # 38 frames apart: 1.52 s at 25 fps (no fragment), 0.76 s at 50 fps
    # (fragment); the 0.5 m distance fits either way.
    assert data_quality.quality_report(frames)["fragment_tracks"] == 0
    d50 = data_quality.quality_report(frames, fps=50)
    assert d50["fragment_tracks"] == 1
    assert d50["fps"] == 50


def test_fragment_distance_is_euclidean_and_bounded():
    def pair(dx, dy):
        return [frame(0, [person(1, 0.0, 0.0)]),
                frame(25, [person(2, dx, dy)])]
    # (1.8, 2.2) is 2.8425 m -> fragment; a |dx|+|dy| metric (4.0) or an
    # x-only metric would disagree.  (1.8, 2.5) is 3.0806 m -> not one.
    assert data_quality.quality_report(pair(1.8, 2.2))["fragment_tracks"] == 1
    assert data_quality.quality_report(pair(1.8, 2.5))["fragment_tracks"] == 0


def test_min_tracked_is_eight():
    def one_frame(n):
        return [frame(0, [person(i, float(i), 0.0) for i in range(1, n + 1)])]
    assert data_quality.quality_report(one_frame(8))["good_frames_share"] == 1.0
    assert data_quality.quality_report(one_frame(7))["good_frames_share"] == 0.0


def test_good_share_rounds_to_3_decimals():
    nine = [person(i, float(i), float(i)) for i in range(1, 10)]
    three = [person(i, float(i), float(i)) for i in range(1, 4)]
    frames = [frame(0, nine), frame(1, nine), frame(2, three)]
    # 2 good of 3 frames: 0.6666... -> 0.667
    assert data_quality.quality_report(frames)["good_frames_share"] == 0.667


def test_jersey_must_be_a_nonempty_string():
    frames = [frame(0, [person(1, 0.0, 0.0, jersey="1"),
                        person(2, 1.0, 0.0, jersey=10),      # not a string
                        person(3, 2.0, 0.0, jersey="  ")])]  # blank
    # 1 read of 3 tracked -> 33.3
    assert data_quality.quality_report(frames)["jersey_read_pct"] == 33.3
