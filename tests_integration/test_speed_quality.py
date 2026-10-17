"""Speed quality: jitter smoothing of the analytics (task: speed_quality).

The tracker's pitch positions contain single-frame jumps, so the raw per-frame
steps pin every player's top speed just under the 12 m/s discard cap.  The
library app computes its analytics summary with the smoothing switched on
(``analytics.summarize(frames, fps, smooth=True)``); with smoothing off the
output must stay exactly what it was before.
"""
import json
from pathlib import Path

import pytest

from sideline import analytics

FIXTURES = Path(__file__).resolve().parent / "fixtures"
UNSMOOTHED_SUMMARY = FIXTURES / "sample_summary_unsmoothed.json"


def tops(summary):
    """[(id, max_speed_ms)] of every player of a summary."""
    return [(p["id"], p["max_speed_ms"]) for p in summary["players"]]


def assert_reachable_top_speeds(tops):
    """The acceptance bar: fewer than 4 players at 10.5+ m/s, none above 11.5."""
    high = [t for t in tops if t[1] >= 10.5]
    assert len(high) < 4 and all(v <= 11.5 for _, v in tops), (
        "top speeds (id, max_speed_ms): %s; players with max_speed_ms >= 10.5: %s"
        % (sorted(tops, key=lambda t: -t[1]), sorted(high, key=lambda t: -t[1])))


def person(pid, x, y=0.0):
    return {"id": pid, "role": "player", "team": "left", "jersey": None,
            "box": [0, 0, 10, 20], "xy": [x, y]}


def frame(n, people):
    return {"frame": n, "delay": 0, "people": people, "ball": None, "colors": None}


def steady_recording(n=51):
    """One player moving at a steady 5 m/s (0.2 m per frame at 25 fps)."""
    return [frame(i, [person(1, 0.2 * i)]) for i in range(n)]


def upload(client, data, name):
    return client.post(
        "/api/matches",
        files={"file": (name + ".jsonl", data, "application/json")},
        data={"name": name},
    )


# ------------------------------------------------------------------ the sample


def test_smoothing_brings_the_sample_top_speeds_down(sample_frames):
    """With smoothing on, the sample stops showing 11.5-12 m/s for everyone."""
    summary = analytics.summarize(sample_frames, smooth=True)
    assert_reachable_top_speeds(tops(summary))


def test_smoothing_never_adds_path(sample_frames):
    """Every player's total distance with smoothing on is at most the raw one."""
    raw = {p["id"]: p["distance_m"] for p in analytics.player_stats(sample_frames)}
    smooth = {p["id"]: p["distance_m"]
              for p in analytics.player_stats(sample_frames, smooth=True)}
    assert set(raw) == set(smooth)
    added = {pid: (raw[pid], smooth[pid]) for pid in raw
             if smooth[pid] > raw[pid] + 1e-9}
    assert not added, "smoothed distance exceeds unsmoothed for %s" % (added,)


def test_smoothing_off_keeps_the_output_byte_identical(sample_frames):
    """Default (smoothing off) summarize output is exactly the pre-smoothing one."""
    assert json.dumps(analytics.summarize(sample_frames)) == \
        UNSMOOTHED_SUMMARY.read_text(encoding="utf-8")


# ------------------------------------------------------------------ synthetic


def test_steady_speed_is_unchanged_by_smoothing():
    """A steady 5 m/s player covers the same distance with smoothing on and off."""
    frames = steady_recording()
    off = analytics.player_stats(frames)[0]
    on = analytics.player_stats(frames, smooth=True)[0]
    assert off["distance_m"] == pytest.approx(10.0)
    assert on["distance_m"] == pytest.approx(off["distance_m"], rel=0.01)
    assert on["max_speed_ms"] == pytest.approx(5.0, abs=0.1)


def test_single_frame_jump_is_removed_by_smoothing():
    """A 3 m single-frame jump is gone after smoothing (position back on track)."""
    xs = [0.2 * i for i in range(51)]
    xs[25] += 3.0  # one frame off by 3 m
    frames = [frame(i, [person(1, xs[i])]) for i in range(51)]
    raw = analytics.player_stats(frames)[0]
    on = analytics.player_stats(frames, smooth=True)[0]
    # the raw track does contain the spike ...
    assert raw["distance_m"] < 10.0  # the two spike steps are discarded (> 12 m/s)
    # ... and smoothing puts the track back: same distance and speed as steady.
    assert on["distance_m"] == pytest.approx(10.0, rel=0.01)
    assert on["max_speed_ms"] == pytest.approx(5.0, abs=0.1)
    apps = analytics._appearances(analytics._clean_sequence(frames))[1]
    smoothed = {f: xy for f, _, _, _, xy in analytics._smoothed_appearances(apps)
                if xy is not None}
    assert smoothed[25][0] == pytest.approx(5.0, abs=0.01)


def test_single_frame_jitter_does_not_hide_a_sprint():
    """Sprint recovery: a 1.2 s burst at 8 m/s with one single-frame 3 m jump.

    Without smoothing the jump's two 67-83 m/s steps are discarded, which
    fragments the burst into two 0.56 s runs below the 1.0 s minimum, so no
    sprint is counted; with smoothing the jump is removed and the whole
    30-frame (1.2 s) run counts as a sprint.
    """
    xs = [0.32 * i for i in range(31)]
    xs[15] += 3.0  # one frame off by 3 m in the middle of the burst
    frames = [frame(i, [person(1, xs[i])]) for i in range(31)]
    off = analytics.player_stats(frames)[0]
    on = analytics.player_stats(frames, smooth=True)[0]
    assert off["sprints"] == 0  # the jitter fragments the run below 1.0 s
    assert on["sprints"] >= 1
    assert on["max_speed_ms"] == pytest.approx(8.0, abs=0.1)


# ------------------------------------------------------------------ the app


def test_app_analytics_endpoint_uses_smoothing(client, sample_bytes):
    """The library app's analytics endpoint computes the smoothed summary."""
    # The steady 5 m/s synthetic recording uploaded through the app returns the
    # same distance_m as the smoothed function produces for the same frames.
    frames = steady_recording()
    data = ("\n".join(json.dumps(f) for f in frames) + "\n").encode("utf-8")
    meta = upload(client, data, "steady").json()
    got = client.get("/api/matches/%s/analytics" % meta["id"]).json()
    expected = analytics.summarize(frames, smooth=True)["players"][0]["distance_m"]
    assert got["players"][0]["distance_m"] == expected == pytest.approx(10.0)
    # And the recorded sample's top speeds through the app meet the same bar.
    upload(client, sample_bytes, "sample")
    items = client.get("/api/matches").json()["items"]
    sample_id = next(m["id"] for m in items if m["name"] == "sample")
    sample_summary = client.get("/api/matches/%s/analytics" % sample_id).json()
    assert_reachable_top_speeds(tops(sample_summary))
