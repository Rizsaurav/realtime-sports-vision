"""Extra adversarial tests for the UI pages task (tasks/ui_pages.md).

The pages (library.html / match.html) render the analytics summary produced
by src/sideline/analytics.py.  docs/WEBAPP_CONTRACT.md fixes that shape ("the
result is JSON-serialisable and has exactly this shape"), and tasks/analytics.md
adds the exact rules.  These tests cover the summary-shape rules the page
renderers depend on that the existing suites do not check: both team keys when
only one side has players, empty input with a custom fps, integer heatmap
grids, the exact player row shape a referee produces (the en-dash cells of the
player table), and a large realistic summary that must survive a strict JSON
round-trip and repeated calls with independent results.
"""

import json
import os
import sys

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

import pytest  # noqa: E402

from sideline import analytics  # noqa: E402


def person(pid, x, y=0.0, role="player", team="left", jersey=None):
    p = {"id": pid, "role": role, "team": team, "jersey": jersey, "box": [0, 0, 10, 20]}
    if x is None:
        p["xy"] = None
    else:
        p["xy"] = [x, y]
    return p


def frame(n, people=(), ball=None):
    return {"frame": n, "delay": 0, "people": list(people), "ball": ball, "colors": None}


def total(grid):
    return sum(sum(row) for row in grid)


def test_summarize_teams_always_has_both_keys_when_only_right_has_players():
    # tasks/analytics.md section 6: "teams: always both keys 'left' and
    # 'right'".  The existing suite only exercises the left-side case, so the
    # page would see a missing "left" key if the rule were broken the other
    # way round.
    f = [
        frame(0, [person(1, 0.0, 0.0, team="right"),
                  person(2, 10.0, 10.0, team="right"),
                  person(3, 20.0, -10.0, team="right")]),
        frame(1, [person(1, 0.0, 0.0, team="right"),
                  person(2, 20.0, 0.0, team="right"),
                  person(3, 10.0, 30.0, team="right")]),
    ]
    teams = analytics.summarize(f)["teams"]
    assert set(teams) == {"left", "right"}
    # The absent side is the all-None shape the team cards render as dashes.
    assert teams["left"] == {"frames": 0, "avg_centroid": None, "avg_width": None, "avg_length": None}
    # The present side: the contract's worked example, on the right team.
    assert teams["right"] == {"frames": 2, "avg_centroid": [10.0, 5.0], "avg_width": 25.0, "avg_length": 20.0}


def test_summarize_empty_with_custom_fps():
    # "fps: the argument as given" plus the empty-input rules (section 6).
    out = analytics.summarize([], fps=50)
    assert out["fps"] == 50
    assert out["frame_count"] == 0
    assert out["first_frame"] is None and out["last_frame"] is None
    assert out["duration_s"] == 0.0
    assert out["players"] == []
    assert out["teams"] == {
        "left": {"frames": 0, "avg_centroid": None, "avg_width": None, "avg_length": None},
        "right": {"frames": 0, "avg_centroid": None, "avg_width": None, "avg_length": None},
    }
    for which in ("left", "right", "ball"):
        grid = out["heatmaps"][which]
        assert len(grid) == 14 and all(len(row) == 21 for row in grid)
        assert total(grid) == 0
    assert out["ball"] == {"frames_with_ball": 0, "distance_m": 0.0}


def test_summarize_heatmaps_are_grids_of_plain_ints():
    # Contract section 2: "21 integers per row, 14 rows".  The page divides
    # these by the max (normalizeGrid), so the cells must be real ints, not
    # floats or bools.
    f = [
        frame(0, [person(1, 0.0, 0.0), person(2, 0.0, 0.0, team="right")], {"xy": [1.0, 2.0]}),
        frame(1, [person(1, 5.0, 0.0), person(2, -5.0, 0.0, team="right")], {"xy": [6.0, 0.0]}),
    ]
    out = analytics.summarize(f)
    for which in ("left", "right", "ball"):
        grid = out["heatmaps"][which]
        assert len(grid) == 14
        for row in grid:
            assert len(row) == 21
            for cell in row:
                assert isinstance(cell, int) and not isinstance(cell, bool)
    assert total(out["heatmaps"]["left"]) == 2
    assert total(out["heatmaps"]["right"]) == 2
    assert total(out["heatmaps"]["ball"]) == 2


def test_summarize_referee_row_is_the_shape_the_table_renders_as_dashes():
    # The player table shows the en dash for team/jersey "null, undefined,
    # empty string or not a finite number" (tasks/ui_pages.md).  A referee's
    # summary row must therefore carry null team and jersey, with the other
    # table fields present and numeric.
    f = [frame(i, [person(5, 0.0, role="referee", team=None, jersey=None)]) for i in range(3)]
    out = analytics.summarize(f)
    assert out["players"] == [
        {"id": 5, "role": "referee", "team": None, "jersey": None, "frames": 3,
         "seconds": 0.12, "distance_m": 0.0, "avg_speed_ms": 0.0, "max_speed_ms": 0.0, "sprints": 0},
    ]


def test_large_realistic_summary_json_shape_and_independence():
    # 400 frames: 4 left + 4 right players drifting 0.02 m per frame, one
    # referee, and a ball absent on every 4th frame.  The summary is the
    # exact input the match page renders, so it must be JSON-safe, keep the
    # contract shape, and be independent across repeated calls.
    n = 400
    frames = []
    for i in range(n):
        people = [person(k, -20.0 + 10.0 * k + 0.02 * i, 0.0) for k in range(4)]
        people += [person(10 + k, 20.0 - 10.0 * k + 0.02 * i, 0.0, team="right") for k in range(4)]
        people.append(person(99, 0.0, role="referee", team=None, jersey=None))
        ball = None if i % 4 == 3 else {"xy": [0.01 * i, 0.0]}
        frames.append(frame(i, people, ball))

    out = analytics.summarize(frames)
    assert out["fps"] == 25
    assert out["frame_count"] == n
    assert out["first_frame"] == 0 and out["last_frame"] == n - 1
    assert out["duration_s"] == 16.0
    # players: sorted by id, both teams' players plus the referee.
    assert [p["id"] for p in out["players"]] == [0, 1, 2, 3, 10, 11, 12, 13, 99]
    p0 = out["players"][0]
    assert p0["team"] == "left" and p0["role"] == "player"
    # 399 steps of 0.02 m at 25 fps: 7.98 m, 0.5 m/s.
    assert p0["frames"] == n
    assert p0["seconds"] == 16.0
    assert p0["distance_m"] == 7.98
    assert p0["avg_speed_ms"] == 0.5
    assert p0["max_speed_ms"] == 0.5
    ref = out["players"][8]
    assert ref["team"] is None and ref["jersey"] is None and ref["distance_m"] == 0.0
    # teams: both sides qualify in every frame.
    assert out["teams"]["left"] == {"frames": n, "avg_centroid": [-1.01, 0.0], "avg_width": 0.0, "avg_length": 30.0}
    assert out["teams"]["right"] == {"frames": n, "avg_centroid": [8.99, 0.0], "avg_width": 0.0, "avg_length": 30.0}
    # heatmaps: every appearance stays on the pitch; the ball misses 100 frames.
    assert total(out["heatmaps"]["left"]) == 4 * n
    assert total(out["heatmaps"]["right"]) == 4 * n
    assert total(out["heatmaps"]["ball"]) == n - 100
    # ball: 300 present frames spanning 0.01 * 398 m (last ball frame is 398).
    assert out["ball"] == {"frames_with_ball": 300, "distance_m": 3.98}
    # The page fetches the summary as JSON: it must round-trip strictly.
    assert json.loads(json.dumps(out, allow_nan=False)) == out
    # Repeated calls give independent results (no shared sub-objects).
    out["teams"]["left"]["avg_width"] = -1
    out["players"][0]["distance_m"] = -1
    out["heatmaps"]["left"][7][14] = 999
    out["ball"]["distance_m"] = -1
    out2 = analytics.summarize(frames)
    assert out2["teams"]["left"]["avg_width"] == 0.0
    assert out2["players"][0]["distance_m"] == 7.98
    assert out2["heatmaps"]["left"][7][14] == 25
    assert out2["ball"]["distance_m"] == 3.98
