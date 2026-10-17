"""Acceptance tests for the events feature, built from features/events/spec.md.

Every expected value is derived by hand from the spec's rules (fps = 25,
sprint >= 7.0 m/s for >= 1.0 s, ball peak > 15.0 m/s, long ball > 25.0 m
within 2.0 s / 50 frames, possession = closest eligible player, timeline
sorted by (frame, type)).
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

import pytest
from fastapi.testclient import TestClient

from sideline.library import create_library_app

FPS = 25


# ---------- frame builders ----------

def person(pid, team, x, y, role="player"):
    return {"id": pid, "role": role, "team": team, "jersey": str(pid),
            "box": [0, 0, 10, 10], "xy": [x, y] if x is not None else None}


def ball(x, y):
    return {"box": [0, 0, 10, 10], "xy": [x, y] if x is not None else None}


def frames(n, people_fn, ball_fn, start=0):
    """n consecutive frame numbers starting at start (a gap test builds
    its list by hand)."""
    out = []
    for f in range(start, start + n):
        out.append({"frame": f, "delay": 0,
                    "people": people_fn(f) or [],
                    "ball": ball_fn(f), "colors": None})
    return out


def ev(frame, first, type_, team, player, **extra):
    d = {"frame": frame, "time_s": round((frame - first) / FPS, 3),
         "type": type_, "team": team, "player": player}
    d.update(extra)
    return d


def combined_frames():
    """Kick from (0,0) to (50,0) at 1.0 m/frame in frames 100..149, an
    exact distance tie at frame 125 (26.0 m on both sides), and player 1
    sprinting 0.32 m/frame in frames 201..225 (25 steps, 8.0 m/s)."""
    def b(f):
        if f <= 99:
            return 0.0
        if f <= 149:
            return float(f - 99)
        return 50.0

    def p1(f):
        if f <= 200:
            return 0.0
        return 0.32 * min(f - 200, 25)

    return frames(261,
                  lambda f: [person(1, "left", p1(f), 0),
                             person(2, "right", 50, 10)],
                  lambda f: ball(b(f), 0))


# ---------- app / store helpers ----------

@pytest.fixture()
def client(tmp_path):
    return TestClient(create_library_app(tmp_path))


def upload(client, rec, name="rec"):
    data = ("\n".join(json.dumps(f) for f in rec) + "\n").encode("utf-8")
    r = client.post("/api/matches",
                    files={"file": ("rec.jsonl", data, "application/octet-stream")},
                    data={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def get_events(client, match_id):
    return client.get("/api/matches/%s/events" % match_id)


def events_of(client, rec):
    """Upload rec, GET its events, return (match_id, response body)."""
    mid = upload(client, rec)
    r = get_events(client, mid)
    assert r.status_code == 200, r.text
    return mid, r.json()


# ---------- basic shapes ----------

def test_one_frame_empty_timeline(client):
    rec = frames(1, lambda f: [person(1, "left", 1, 0)], lambda f: ball(0, 0))
    mid, body = events_of(client, rec)
    assert set(body) == {"match_id", "events"}
    assert body == {"match_id": mid, "events": []}


def test_static_recording_no_events(client):
    rec = frames(50, lambda f: [person(1, "left", 6, 5),
                                person(2, "right", -6, 5)],
                 lambda f: ball(5, 5))
    _, body = events_of(client, rec)
    assert body["events"] == []


# ---------- sprints (rules R3, R4) ----------

def test_sprint_event(client):
    # player 7 runs 0.32 m/frame (8.0 m/s) from frame 100: 30 steps =
    # 1.2 s, 30 x 0.32 = 9.6 m; player 9 jogs 0.1 m/frame (2.5 m/s) and
    # must not sprint. The owner flips left->left only, no possession event.
    def p7(f):
        if f <= 100:
            return 0.0
        return 0.32 * min(f - 100, 30)

    rec = frames(150,
                 lambda f: [person(7, "left", p7(f), 0),
                            person(9, "left", 0.1 * f, 2)],
                 lambda f: ball(40, 30))
    _, body = events_of(client, rec)
    assert body["events"] == [ev(130, 0, "sprint", "left", 7,
                                 speed_m_s=8.0, distance_m=9.6, duration_s=1.2)]


def test_sprint_requires_one_second(client):
    # player 1: 24 fast steps = 0.96 s -> no sprint.
    # player 2: 25 fast steps = 1.0 s -> sprint at its last frame.
    rec = frames(31,
                 lambda f: [person(1, "left", 0.32 * min(f, 24), 0),
                            person(2, "left", 0.32 * min(f, 25), 5)],
                 lambda f: None)
    _, body = events_of(client, rec)
    assert body["events"] == [ev(25, 0, "sprint", "left", 2,
                                 speed_m_s=8.0, distance_m=8.0, duration_s=1.0)]


def test_sprint_below_speed_threshold(client):
    # 6.5 m/s (0.26 m/frame) for 30 steps: long enough, too slow.
    rec = frames(40, lambda f: [person(1, "left", 0.26 * min(f, 30), 0)],
                 lambda f: None)
    _, body = events_of(client, rec)
    assert body["events"] == []


def test_sprint_broken_by_teleport(client):
    # 14 fast steps, a 30 m one-frame jump (750 m/s > 12.0 cap: the step is
    # discarded and breaks the run), 14 fast steps. Both runs are 0.56 s,
    # under 1.0 s.
    def p(f):
        if f <= 14:
            return 0.32 * f
        if f <= 29:
            return 0.32 * 14 + 30 + 0.32 * (f - 15)
        return 0.32 * 14 + 30 + 0.32 * 14

    rec = frames(31, lambda f: [person(1, "left", p(f), 0)], lambda f: None)
    _, body = events_of(client, rec)
    assert body["events"] == []


def test_sprint_across_missing_positions(client):
    # 8.0 m/s with frames 13-15 missing: the step 12 -> 16 has g = 4
    # (<= 5), d = 1.28 m, v = 8.0 m/s, so the run continues. Total gap 29
    # frames = 1.16 s, distance 9.28 m, last appearance frame 29.
    def p(f):
        if 13 <= f <= 15:
            return person(1, "left", None, None)
        return person(1, "left", 0.32 * min(f, 29), 0)

    rec = frames(31, lambda f: [p(f)], lambda f: None)
    assert [x for x in rec if 13 <= x["frame"] <= 15 and x["people"][0]["xy"]
            is not None] == []  # the gap is really in the fixture
    _, body = events_of(client, rec)
    assert body["events"] == [ev(29, 0, "sprint", "left", 1,
                                 speed_m_s=8.0, distance_m=9.28,
                                 duration_s=1.16)]


# ---------- ball speed peaks (rules R5, R6, R7) ----------

def test_ball_speed_peak_single_event(client):
    # 2.0 m/frame flight (50.0 m/s) for 12 frames, 24 m total: one peak
    # event at the first sample frame, speed 50.0; 24 m in 0.48 s is not a
    # long ball.
    def b(f):
        if f <= 24:
            return 0.0
        return 2.0 * min(f - 24, 12)

    rec = frames(40, lambda f: [], lambda f: ball(b(f), 0))
    _, body = events_of(client, rec)
    assert body["events"] == [ev(25, 0, "ball_speed_peak", None, None,
                                 speed_m_s=50.0)]


def test_ball_speed_below_threshold(client):
    # 0.592 m/frame = 14.8 m/s for 30 frames (17.76 m): under 15.0 m/s and
    # under 25 m, no events.
    def b(f):
        if f <= 20:
            return 0.0
        return 0.592 * min(f - 20, 30)

    rec = frames(60, lambda f: [], lambda f: ball(b(f), 0))
    _, body = events_of(client, rec)
    assert body["events"] == []


def test_ball_speed_run_collapses_to_one_event(client):
    # 0.625 m/frame = 15.625 m/s for 40 frames: 40 qualifying samples are
    # one run -> exactly one event at the first sample frame; round(15.625,
    # 2) = 15.62. Total 25.0 m is not > 25.0, so no long ball.
    def b(f):
        if f <= 39:
            return 0.0
        return 0.625 * min(f - 39, 40)

    rec = frames(100, lambda f: [], lambda f: ball(b(f), 0))
    xs = [fr["ball"]["xy"][0] for fr in rec]
    assert max(xs) == 25.0 and (xs[-1] == 25.0)  # fixture really spans 25 m
    _, body = events_of(client, rec)
    assert body["events"] == [ev(40, 0, "ball_speed_peak", None, None,
                                 speed_m_s=15.62)]


# ---------- long balls (rule R8) ----------

def test_long_ball_and_peak_from_kick(client):
    # 1.0 m/frame flight (25.0 m/s) in frames 100..149: 50 m total.
    # Peak at the first sample (frame 100); the 2 s window displacement
    # first exceeds 25 m at frame 125 (26.0 m) and peaks at 50.0 m.
    def b(f):
        if f <= 99:
            return 0.0
        if f <= 149:
            return float(f - 99)
        return 50.0

    rec = frames(200, lambda f: [], lambda f: ball(b(f), 0))
    _, body = events_of(client, rec)
    assert body["events"] == [
        ev(100, 0, "ball_speed_peak", None, None, speed_m_s=25.0),
        ev(125, 0, "long_ball", None, None, distance_m=50.0),
    ]


def test_short_kick_no_events(client):
    # 0.4 m/frame for 50 frames: 20 m at 10.0 m/s - under both thresholds.
    def b(f):
        if f <= 99:
            return 0.0
        return 0.4 * min(f - 99, 50)

    rec = frames(200, lambda f: [], lambda f: ball(b(f), 0))
    _, body = events_of(client, rec)
    assert body["events"] == []


def test_exactly_25m_no_long_ball(client):
    # 0.5 m/frame for 50 frames: exactly 25.0 m in 2.0 s at 12.5 m/s. The
    # long-ball rule is strict (distance > 25.0 m), so no event.
    def b(f):
        if f <= 99:
            return 0.0
        return 0.5 * min(f - 99, 50)

    rec = frames(200, lambda f: [], lambda f: ball(b(f), 0))
    xs = [fr["ball"]["xy"][0] for fr in rec]
    assert max(xs) == 25.0  # fixture really travels exactly 25 m
    _, body = events_of(client, rec)
    assert body["events"] == []


# ---------- possession (rule R9) ----------

def test_possession_change_with_tie(client):
    # Ball slides from (0,0) to (20,0) at 0.5 m/frame (12.5 m/s); id 1
    # (left) stands at (1,0), id 2 (right) at (20,0). At frame 21 the ball
    # is at (10.5,0): 9.5 m from each -> the tie goes to id 1 (left). The
    # change to right happens at frame 22, where id 2 is 9 m away vs 10 m.
    rec = frames(60,
                 lambda f: [person(1, "left", 1, 0), person(2, "right", 20, 0)],
                 lambda f: ball(0.5 * min(f, 40), 0))
    _, body = events_of(client, rec)
    assert body["events"] == [ev(22, 0, "possession_change", "right", 2,
                                 from_team="left")]


def test_same_team_pass_no_event(client):
    # The ball moves from id 1 to id 3, both left: the owner's player
    # changes but the team does not, so no possession change. 10 m at
    # 6.25 m/s triggers nothing else.
    rec = frames(60,
                 lambda f: [person(1, "left", 0, 0), person(3, "left", 10, 0),
                            person(2, "right", 40, 0)],
                 lambda f: ball(0.25 * min(f, 40), 0))
    _, body = events_of(client, rec)
    assert body["events"] == []


def test_missing_ball_pauses_ownership(client):
    # Owner left (ball at (0,0)), ball missing for 25 frames, ball back at
    # (10,0): the re-appearance is 10 m away 26 frames later (9.6 m/s, no
    # peak, no long ball) and the change is reported at frame 35, not at
    # the disappearance.
    def b(f):
        if f <= 9:
            return 0.0
        if f <= 34:
            return None
        return 10.0

    rec = frames(50,
                 lambda f: [person(1, "left", 0, 0), person(2, "right", 10, 0)],
                 lambda f: ball(b(f), 0) if b(f) is not None else None)
    _, body = events_of(client, rec)
    assert body["events"] == [ev(35, 0, "possession_change", "right", 2,
                                 from_team="left")]


def test_referee_never_owns(client):
    # Only a referee is near the ball (team null, role referee): no frame
    # has an owner, so no possession event can occur.
    rec = frames(50, lambda f: [person(99, None, 0.5, 0, role="referee")],
                 lambda f: ball(0, 0))
    _, body = events_of(client, rec)
    assert body["events"] == []


# ---------- people edge cases (rule R2) ----------

def test_duplicate_id_first_entry_wins(client):
    # id 7 appears twice in every frame: the first entry stays at (0,0),
    # the second at (60,0) for frames 0-14 and (5,0) for frames 15-29.
    # id 8 (right) stands at (20,0) and the ball at (0,0). Reading the
    # first entry, the owner is id 7 (left) on every frame: no events.
    # Reading the last entry instead would switch the owner to id 8 at
    # frame 15.
    rec = frames(30,
                 lambda f: [person(7, "left", 0, 0),
                            person(7, "left", 60.0 if f < 15 else 5.0, 0),
                            person(8, "right", 20, 0)],
                 lambda f: ball(0, 0))
    for fr in rec:  # the fixture really has two conflicting id-7 entries
        e7 = [p for p in fr["people"] if p["id"] == 7]
        assert len(e7) == 2 and e7[0]["xy"] != e7[1]["xy"]
    _, body = events_of(client, rec)
    assert body["events"] == []


# ---------- frame gaps (rules R1, R6, R8) ----------

def test_frame_gap_uses_frame_numbers(client):
    # Frames 0-4: ball at (0,0); the next recorded frame is 30, ball at
    # (26,0). Sample v = 26 / (26/25) = 25.0 m/s -> peak at frame 30;
    # anchor distance 26.0 m > 25.0 -> long ball at frame 30 too. Same
    # frame, so the alphabetical type order applies: ball_speed_peak
    # before long_ball. time_s = 30/25 = 1.2.
    rec = []
    for fnum in (0, 1, 2, 3, 4, 30, 31, 32):
        x = 0.0 if fnum < 30 else 26.0
        rec.append({"frame": fnum, "delay": 0, "people": [],
                    "ball": ball(x, 0), "colors": None})
    _, body = events_of(client, rec)
    assert body["events"] == [
        ev(30, 0, "ball_speed_peak", None, None, speed_m_s=25.0),
        ev(30, 0, "long_ball", None, None, distance_m=26.0),
    ]


def test_time_relative_to_first_frame(client):
    # The recording starts at frame 120. Kick at 1.0 m/frame in frames
    # 220..245 (26 m): peak at frame 220 -> time_s (220-120)/25 = 4.0;
    # long ball first at frame 245 -> time_s (245-120)/25 = 5.0.
    def b(f):
        if f <= 219:
            return 0.0
        if f <= 245:
            return float(f - 219)
        return 26.0

    rec = frames(161, lambda f: [], lambda f: ball(b(f), 0), start=120)
    assert rec[0]["frame"] == 120  # the base really is non-zero
    _, body = events_of(client, rec)
    assert body["events"] == [
        ev(220, 120, "ball_speed_peak", None, None, speed_m_s=25.0),
        ev(245, 120, "long_ball", None, None, distance_m=26.0),
    ]


def test_no_people_no_possession(client):
    # No people at all: the ball's 20 m slide (12.5 m/s) cannot create a
    # possession change, and it is neither a peak nor a long ball.
    rec = frames(60, lambda f: [], lambda f: ball(0.5 * min(f, 40), 0))
    _, body = events_of(client, rec)
    assert body["events"] == []


# ---------- combined timeline (rules R10, R11, R12) ----------

def test_combined_sorted_timeline(client):
    rec = combined_frames()
    mid, body = events_of(client, rec)
    assert set(body) == {"match_id", "events"}
    assert body["match_id"] == mid
    assert body["events"] == [
        ev(100, 0, "ball_speed_peak", None, None, speed_m_s=25.0),
        ev(125, 0, "long_ball", None, None, distance_m=50.0),
        ev(126, 0, "possession_change", "right", 2, from_team="left"),
        ev(225, 0, "sprint", "left", 1, speed_m_s=8.0, distance_m=8.0,
           duration_s=1.0),
    ]
    base = {"frame", "time_s", "type", "team", "player"}
    for e in body["events"]:
        if e["type"] == "sprint":
            assert set(e) == base | {"speed_m_s", "distance_m", "duration_s"}
        elif e["type"] == "ball_speed_peak":
            assert set(e) == base | {"speed_m_s"}
        elif e["type"] == "long_ball":
            assert set(e) == base | {"distance_m"}
        else:
            assert e["type"] == "possession_change"
            assert set(e) == base | {"from_team"}


# ---------- input validation ----------

def test_unknown_match_404(client):
    r = client.get("/api/matches/m_0000000000/events")
    assert r.status_code == 404
    assert "error" in r.json()


def test_malformed_match_id_404(client):
    r = client.get("/api/matches/m_123456789/events")  # 9 hex digits
    assert r.status_code == 404
    assert "error" in r.json()


def test_traversal_match_id_404(client):
    r = client.get("/api/matches/m_..%2F..%2Fx/events")
    assert r.status_code == 404
    assert "error" in r.json()


def test_rejected_upload_400_stores_nothing(client):
    # non-increasing frame numbers are rejected by the store (400) and
    # store nothing.
    bad = [
        {"frame": 5, "delay": 0, "people": [person(1, "left", 0, 0)],
         "ball": ball(0, 0), "colors": None},
        {"frame": 3, "delay": 0, "people": [], "ball": None, "colors": None},
    ]
    r = client.post("/api/matches",
                    files={"file": ("bad.jsonl",
                                    ("\n".join(json.dumps(f) for f in bad)
                                     + "\n").encode("utf-8"),
                                    "application/octet-stream")},
                    data={"name": "bad"})
    assert r.status_code == 400
    assert "error" in r.json()
    assert client.get("/api/matches").json() == {"items": []}


# ---------- idempotence and the cache side channel ----------

def test_repeated_gets_idempotent(client):
    mid = upload(client, combined_frames())
    bodies = [get_events(client, mid).json() for _ in range(3)]
    assert bodies[0] == bodies[1] == bodies[2]
    assert bodies[0]["match_id"] == mid
    assert len(bodies[0]["events"]) == 4


def test_cache_file_written(tmp_path, client):
    mid = upload(client, combined_frames())
    r = get_events(client, mid)
    assert r.status_code == 200
    cache = tmp_path / mid / "features" / "events.json"
    assert cache.is_file()
    assert json.loads(cache.read_text(encoding="utf-8")) == r.json()["events"]


def test_stale_cache_is_recomputed(tmp_path, client):
    mid = upload(client, combined_frames())
    first = get_events(client, mid)
    assert first.status_code == 200
    cache = tmp_path / mid / "features" / "events.json"
    mtime_ns = cache.stat().st_mtime_ns
    now = time.time() + 10  # make frames.jsonl newer than the cache
    os.utime(tmp_path / mid / "frames.jsonl", (now, now))
    second = get_events(client, mid)
    assert second.status_code == 200
    assert second.json() == first.json()  # still the right answer
    assert cache.stat().st_mtime_ns > mtime_ns  # ...by recomputing


def test_no_leftover_temp_files(tmp_path, client):
    mid = upload(client, combined_frames())
    assert get_events(client, mid).status_code == 200
    assert get_events(client, mid).status_code == 200
    names = {str(p.relative_to(tmp_path / mid))
             for p in (tmp_path / mid).rglob("*") if p.is_file()}
    assert names == {"frames.jsonl", "meta.json", "features/events.json"}


# ---------- purity of the module function ----------

def test_detect_events_pure_and_matches_http(client):
    from sideline.features.events import detect_events
    rec = combined_frames()
    mid = upload(client, rec)
    r = get_events(client, mid)
    assert r.status_code == 200
    snapshot = copy.deepcopy(rec)
    out = detect_events(rec)
    assert rec == snapshot  # the input frames were not mutated
    assert out == r.json()["events"]


# ---------- performance (edge case: huge input) ----------

def test_performance_25000_frames(client):
    # 25,000 frames, 10 standing players, one 26 m kick at frame 10000:
    # exactly the predicted 2 events, the first (uncached) GET under 15.0 s
    # and the cached second under 5.0 s.
    def b(f):
        if f <= 9999:
            return 0.0
        if f <= 10025:
            return float(f - 9999)
        return 26.0

    rec = frames(25000,
                 lambda f: [person(i, "left" if i % 2 else "right", 30, 30)
                            for i in range(1, 11)],
                 lambda f: ball(b(f), 0))
    assert len(rec) == 25000
    mid = upload(client, rec)
    t0 = time.monotonic()
    r = get_events(client, mid)
    first_s = time.monotonic() - t0
    assert r.status_code == 200
    assert first_s < 15.0
    assert r.json()["events"] == [
        ev(10000, 0, "ball_speed_peak", None, None, speed_m_s=25.0),
        ev(10025, 0, "long_ball", None, None, distance_m=26.0),
    ]
    t0 = time.monotonic()
    r2 = get_events(client, mid)
    second_s = time.monotonic() - t0
    assert second_s < 5.0
    assert r2.status_code == 200
    assert r2.json() == r.json()
