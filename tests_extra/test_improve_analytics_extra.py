"""Extra adversarial tests for src/sideline/analytics.py.

Every assertion is required by tasks/analytics.md (which fixes the input and
output shapes via docs/WEBAPP_CONTRACT.md).  These tests cover spec rules the
existing suite does not check: invalid-position edge cases (bool, NaN, inf,
wrong length, tuples), most-common tie-breaking, sprint runs ended by
discarded or slow steps, duplicate id/frame rules for team_shape, heatmap
grid-size edges and result independence, ball with non-dict or invalid xy,
summarize consistency and JSON safety, large input, and repeated/concurrent
calls.
"""

import copy
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

from sideline import analytics  # noqa: E402


# ---------------------------------------------------------------- helpers

def person(pid, x, y=0.0, role="player", team="left", jersey=None):
    p = {"id": pid, "role": role, "team": team, "jersey": jersey, "box": [0, 0, 10, 20]}
    if x is None:
        p["xy"] = None
    elif isinstance(x, (list, tuple)):
        p["xy"] = x  # raw position value (may be malformed on purpose)
    else:
        p["xy"] = [x, y]
    return p


def frame(n, people=(), ball=None):
    return {"frame": n, "delay": 0, "people": list(people), "ball": ball, "colors": None}


def at(pid, pairs, **kw):
    """One person at the given (frame, x) pairs, y always 0."""
    return [frame(f, [person(pid, x, 0.0, **kw)]) for f, x in pairs]


def line(n_steps, step, x0=0.0):
    return [x0 + i * step for i in range(n_steps + 1)]


def stats(frames, fps=25, pid=1):
    return [s for s in analytics.player_stats(frames, fps) if s["id"] == pid][0]


def total(grid):
    return sum(sum(r) for r in grid)


# ---------------------------------------------------------------- load_frames

def test_load_frames_rejects_valid_json_that_is_not_an_object(tmp_path):
    # 5 -> line 1, "abc" -> line 2, true -> line 3, null -> line 4; the first
    # bad line must be named.
    p = tmp_path / "f.jsonl"
    p.write_text('5\n"abc"\ntrue\nnull\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        analytics.load_frames(str(p))
    p.write_text('{"frame": 0}\ntrue\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        analytics.load_frames(str(p))


def test_load_frames_does_not_validate_other_fields(tmp_path):
    # No validation of other fields: weird people/ball/delay values load fine
    # and come back exactly as parsed, in file order.
    p = tmp_path / "f.jsonl"
    p.write_text(
        '{"frame": 2, "people": "garbage", "ball": {"xy": [1, "x"]}}\n'
        '{"frame": 1, "delay": null, "extra": [1, {"a": null}]}\n',
        encoding="utf-8",
    )
    got = analytics.load_frames(str(p))
    assert got == [
        {"frame": 2, "people": "garbage", "ball": {"xy": [1, "x"]}},
        {"frame": 1, "delay": None, "extra": [1, {"a": None}]},
    ]


def test_load_frames_line_number_counts_leading_blanks(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text("\n\n{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        analytics.load_frames(str(p))


# ---------------------------------------------------------------- valid positions

def test_bool_position_is_not_a_position():
    # bools are not numbers per the spec: the appearance counts, position does not.
    f = [frame(0, [person(1, [True, 0])]),
         frame(1, [person(1, [0.0, 0.0])]),
         frame(2, [person(1, [0.2, 0.0])])]
    s = stats(f)
    assert s["frames"] == 3
    assert s["distance_m"] == pytest.approx(0.2)
    assert s["max_speed_ms"] == pytest.approx(5.0)


def test_nan_and_inf_positions_are_not_positions():
    f = [frame(0, [person(1, [0.0, 0.0])]),
         frame(1, [person(1, [float("nan"), 0.0])]),
         frame(2, [person(1, [float("inf"), 1.0])]),
         frame(3, [person(1, [0.6, 0.0])])]
    # The nan and inf appearances are skipped: one candidate step, 0 -> 3,
    # gap 3, d 0.6, t 0.12, 5 m/s, counted.
    s = stats(f)
    assert s["frames"] == 4
    assert s["distance_m"] == pytest.approx(0.6)
    assert s["max_speed_ms"] == pytest.approx(5.0)


def test_malformed_positions_are_not_positions():
    bad = [[1, 2, 3], [1], "ab", [1, "2"], [None, 2], 5, {}, [float("inf"), 0], [True, False]]
    f = []
    for i, xy in enumerate(bad):
        p = person(1, None)
        p["xy"] = xy
        f.append(frame(i, [p]))
    s = stats(f)
    assert s["frames"] == len(bad)
    assert s["distance_m"] == 0.0
    assert s["avg_speed_ms"] == 0.0
    assert s["max_speed_ms"] == 0.0
    assert s["sprints"] == 0


def test_tuple_position_is_valid():
    f = [frame(0, [person(1, (0.0, 0.0))]), frame(1, [person(1, (0.2, 0.0))])]
    assert stats(f)["distance_m"] == pytest.approx(0.2)
    # Tuples are also valid for the heatmap.
    g = analytics.heatmap([frame(0, [person(1, (-52.5, -34))])], "left")
    assert g[0][0] == 1 and total(g) == 1


def test_person_without_id_is_ignored():
    f = [frame(0, [{"role": "player", "team": "left", "xy": [1.0, 1.0]}])]
    assert analytics.player_stats(f) == []
    assert analytics.summarize(f)["players"] == []
    assert total(analytics.heatmap(f, "left")) == 0


# ---------------------------------------------------------------- most common

def test_three_way_tie_goes_to_smallest_value():
    f = [frame(0, [person(1, 0.0, role="referee")]),
         frame(1, [person(1, 0.0, role="player")]),
         frame(2, [person(1, 0.0, role="goalkeeper")])]
    assert stats(f)["role"] == "goalkeeper"


def test_missing_jersey_key_counts_as_null():
    p = person(1, 0.0)
    del p["jersey"]
    f = [frame(0, [p]),
         frame(1, [person(1, 0.0, jersey="7")]),
         frame(2, [person(1, 0.0, jersey="7")]),
         frame(3, [person(1, 0.0, jersey="8")])]
    assert stats(f)["jersey"] == "7"


# ---------------------------------------------------------------- steps

def test_max_speed_comes_from_an_interior_step():
    # speeds: 10, 8.333..., 5 m/s
    s = stats(at(1, [(0, 0.0), (1, 0.4), (4, 1.4), (5, 1.6)]))
    assert s["distance_m"] == pytest.approx(1.6)
    assert s["avg_speed_ms"] == pytest.approx(1.6 / 0.2)
    assert s["max_speed_ms"] == pytest.approx(10.0)
    assert s["sprints"] == 0


def test_single_appearance_has_zero_stats():
    s = stats([frame(4, [person(1, 3.0)])])
    assert s["frames"] == 1
    assert s["seconds"] == pytest.approx(1 / 25)
    assert s["distance_m"] == 0.0
    assert s["avg_speed_ms"] == 0.0
    assert s["max_speed_ms"] == 0.0
    assert s["sprints"] == 0


def test_seconds_are_frames_over_fps_unrounded():
    f = [frame(i, [person(1, 0.0)]) for i in range(7)]
    assert stats(f)["seconds"] == pytest.approx(7 / 25)


def test_gap_five_at_exactly_twelve_ms_is_kept():
    # 15 m in 5 frames at fps 4: t = 1.25 s exactly, speed exactly 12.0.
    s = stats(at(1, [(0, 0.0), (5, 15.0)]), fps=4)
    assert s["distance_m"] == pytest.approx(15.0)
    assert s["max_speed_ms"] == pytest.approx(12.0)
    over = stats(at(1, [(0, 0.0), (5, 15.5)]), fps=4)  # 12.4 m/s
    assert over["distance_m"] == 0.0


# ---------------------------------------------------------------- sprints

def test_sprint_run_ended_by_gap_discard_still_counts():
    # 25 fast steps (1.0 s) then a gap-6 jump: the jump discards the step and
    # ends the run, but the run has lasted a full second.
    pts = [(i, 0.3 * i) for i in range(26)] + [(31, 5.0)]
    s = stats(at(1, pts))
    assert s["frames"] == 27
    assert s["distance_m"] == pytest.approx(7.5)
    assert s["sprints"] == 1


def test_sprint_run_ended_by_slow_step_still_counts():
    # 25 fast (1.0 s), 1 slow, 24 fast (0.96 s) -> exactly one sprint.
    xs = line(25, 0.3)
    xs = xs + [xs[-1] + 0.1] + [xs[-1] + 0.1 + 0.3 * i for i in range(1, 25)]
    s = stats(at(1, list(enumerate(xs))))
    assert s["sprints"] == 1
    assert s["distance_m"] == pytest.approx(7.5 + 0.1 + 7.2)


def test_three_runs_three_sprints():
    # run of 25 fast, slow step, run of 25 fast, gap-6 discard, run of 25 fast.
    pts = [(i, 0.3 * i) for i in range(26)]
    slow_x = pts[-1][1] + 0.1
    pts.append((26, slow_x))
    pts += [(27 + i, slow_x + 0.3 * (i + 1)) for i in range(25)]  # frames 27..51
    jump_x = 20.0
    pts.append((57, jump_x))  # gap 6 from frame 51: discarded
    pts += [(58 + i, jump_x + 0.3 * (i + 1)) for i in range(25)]  # frames 58..82
    s = stats(at(1, pts))
    assert s["sprints"] == 3
    assert s["distance_m"] == pytest.approx(7.5 + 0.1 + 7.5 + 7.5)
    assert s["max_speed_ms"] == pytest.approx(7.5)


def test_sprint_fps30_boundary():
    # 0.25 m per frame at 30 fps is 7.5 m/s; 30 frames last exactly 1.0 s.
    assert stats(at(1, [(i, 0.25 * i) for i in range(31)]), fps=30)["sprints"] == 1
    assert stats(at(1, [(i, 0.25 * i) for i in range(30)]), fps=30)["sprints"] == 0


# ---------------------------------------------------------------- team_shape

def test_duplicate_id_does_not_pad_the_three_count():
    f = [frame(0, [person(1, 0.0, 0.0), person(1, 100.0, 100.0), person(2, 20.0, -10.0)])]
    assert analytics.team_shape(f, "left")["frames"] == 0


def test_team_shape_last_duplicate_frame_wins():
    two = [frame(0, [person(1, 0.0), person(2, 10.0)])]
    three = [frame(0, [person(1, 0.0), person(2, 10.0), person(3, 20.0)])]
    assert analytics.team_shape(two + three, "left")["frames"] == 1
    assert analytics.team_shape(three + two, "left")["frames"] == 0


def test_team_shape_unsorted_input_matches_spec_example():
    f = [frame(1, [person(10 + i, x, y) for i, (x, y) in enumerate([(0, 0), (20, 0), (10, 30)])]),
         frame(0, [person(10 + i, x, y) for i, (x, y) in enumerate([(0, 0), (10, 10), (20, -10)])])]
    r = analytics.team_shape(f, "left")
    assert r["frames"] == 2
    assert r["avg_centroid"] == pytest.approx([10.0, 5.0])
    assert r["avg_width"] == pytest.approx(25.0)
    assert r["avg_length"] == pytest.approx(20.0)


def test_team_shape_four_players():
    f = [frame(0, [person(1, 0.0, 0.0), person(2, 10.0, 0.0),
                   person(3, 5.0, 10.0), person(4, 5.0, -10.0)])]
    r = analytics.team_shape(f, "left")
    assert r["frames"] == 1
    assert r["avg_centroid"] == pytest.approx([5.0, 0.0])
    assert r["avg_width"] == pytest.approx(20.0)
    assert r["avg_length"] == pytest.approx(10.0)


# ---------------------------------------------------------------- heatmap

def test_heatmap_single_cell_grid():
    assert analytics.heatmap([frame(0, [person(1, 0.0, 0.0)])], "left", nx=1, ny=1) == [[1]]
    # The far corner still belongs to the (only) cell.
    assert analytics.heatmap([frame(0, [person(1, 52.5, 34.0)])], "left", nx=1, ny=1) == [[1]]
    # Out of range is still ignored.
    assert analytics.heatmap([frame(0, [person(1, 52.6, 0.0)])], "left", nx=1, ny=1) == [[0]]


def test_heatmap_single_row_grid():
    g = analytics.heatmap([frame(0, [person(1, 0.0, 34.0), person(2, 0.0, -34.0),
                                     person(3, 0.0, 0.0)])], "left", ny=1)
    assert len(g) == 1 and len(g[0]) == 21
    assert g[0][10] == 3 and total(g) == 3


def test_heatmap_one_point_per_column():
    people = [person(i + 1, -52.5 + 5.25 * i, 0.0) for i in range(21)]
    g = analytics.heatmap([frame(0, people)], "left")
    assert g[7] == [1] * 21
    assert total(g) == 21


def test_heatmap_ball_object_without_valid_xy_is_not_counted():
    f = [frame(0, [], {"box": [0, 0, 5, 5]}),                       # no xy key
         frame(1, [], {"box": [0, 0, 5, 5], "xy": None}),           # xy null
         frame(2, [], {"box": [0, 0, 5, 5], "xy": [1.0, 2.0, 3.0]}),  # bad xy
         frame(3, [], {"box": [0, 0, 5, 5], "xy": [0.0, 0.0]})]
    g = analytics.heatmap(f, "ball")
    assert g[7][10] == 1 and total(g) == 1


def test_heatmap_returns_independent_grids():
    f = [frame(0, [person(1, 1.0, 1.0)])]
    g1 = analytics.heatmap(f, "left")
    g2 = analytics.heatmap(f, "left")
    g1[7][10] = 99
    assert g2[7][10] == 1
    assert g1 is not g2


def test_heatmap_unsorted_frames_same_result():
    f = [frame(0, [person(1, 0.0, 0.0)]),
         frame(2, [person(1, 10.0, 0.0)]),
         frame(1, [person(1, 5.0, 0.0)])]
    assert analytics.heatmap(f, "left") == analytics.heatmap(list(reversed(f)), "left")


# ---------------------------------------------------------------- ball

def test_ball_with_invalid_xy_is_not_in_the_list_but_step_jumps_over():
    # Frame 1's ball has an invalid xy, so the ball track is f0 -> f2:
    # gap 2, d 2.0, t 0.08, 25 m/s <= 40 -> counted.
    f = [frame(0, [], {"xy": [0.0, 0.0]}),
         frame(1, [], {"xy": [1.0, "x"]}),
         frame(2, [], {"xy": [2.0, 0.0]})]
    b = analytics.summarize(f)["ball"]
    assert b["frames_with_ball"] == 2
    assert b["distance_m"] == pytest.approx(2.0)


def test_ball_that_is_not_an_object_is_not_counted():
    f = [frame(0, [], "not a ball"),
         frame(1, [], [1.0, 1.0]),
         frame(2, [], {"xy": [0.0, 0.0]})]
    b = analytics.summarize(f)["ball"]
    assert b["frames_with_ball"] == 1
    assert b["distance_m"] == 0.0


# ---------------------------------------------------------------- summarize

def mixed_frames():
    f = at(1, [(0, 0.0), (1, 0.2), (3, 0.7)], jersey="7")
    f += [frame(5, [person(2, 10.0, role="goalkeeper"),
                    person(3, 20.0, team="right")])]
    f[0]["ball"] = {"box": [0, 0, 5, 5], "xy": [0.0, 0.0]}
    f[1]["ball"] = {"box": [0, 0, 5, 5], "xy": [0.5, 0.0]}
    return f


def test_summarize_heatmaps_equal_standalone_default_grid():
    f = mixed_frames()
    out = analytics.summarize(f)
    assert out["heatmaps"]["left"] == analytics.heatmap(f, "left")
    assert out["heatmaps"]["right"] == analytics.heatmap(f, "right")
    assert out["heatmaps"]["ball"] == analytics.heatmap(f, "ball")


def test_summarize_players_consistent_with_unrounded_stats():
    f = mixed_frames()
    raw = analytics.player_stats(f)
    out = analytics.summarize(f)["players"]
    assert len(out) == len(raw)
    for r, o in zip(raw, out):
        for key in ("id", "role", "team", "jersey", "frames", "sprints"):
            assert o[key] == r[key]
        for key in ("seconds", "distance_m", "avg_speed_ms", "max_speed_ms"):
            assert o[key] == round(r[key], 2)


def test_summarize_teams_consistent_with_unrounded_shape():
    f = mixed_frames()
    out = analytics.summarize(f)["teams"]
    for side in ("left", "right"):
        r = analytics.team_shape(f, side)
        assert out[side]["frames"] == r["frames"]
        if r["avg_centroid"] is None:
            assert out[side]["avg_centroid"] is None
            assert out[side]["avg_width"] is None
            assert out[side]["avg_length"] is None
        else:
            assert out[side]["avg_centroid"] == [round(r["avg_centroid"][0], 2),
                                                  round(r["avg_centroid"][1], 2)]
            assert out[side]["avg_width"] == round(r["avg_width"], 2)
            assert out[side]["avg_length"] == round(r["avg_length"], 2)


def test_summarize_duration_uses_frame_count_over_fps():
    out = analytics.summarize([frame(0), frame(100), frame(200)], fps=50)
    assert out["frame_count"] == 3
    assert out["first_frame"] == 0 and out["last_frame"] == 200
    assert out["duration_s"] == 0.06  # 3 / 50, not (200 - 0) / 50


def test_summarize_only_duplicate_frames():
    f = [frame(7, [person(1, 0.0)]) for _ in range(3)]
    out = analytics.summarize(f)
    assert out["frame_count"] == 1
    assert out["first_frame"] == 7 and out["last_frame"] == 7
    assert out["duration_s"] == 0.04
    assert out["players"][0]["frames"] == 1


def test_summarize_output_is_strictly_json_serialisable():
    # Invalid (NaN) positions must not leak into the output in any form.
    f = [frame(0, [person(1, [float("nan"), 0.0])]),
         frame(1, [person(1, [float("inf"), 0.0])]),
         frame(2, [person(1, [0.0, 0.0])])]
    out = analytics.summarize(f)
    json.dumps(out, allow_nan=False)
    assert out["players"][0]["distance_m"] == 0.0


def test_summarize_repeated_calls_give_independent_results():
    f = mixed_frames()
    out1 = analytics.summarize(f)
    out1["heatmaps"]["left"][7][10] = 99
    out1["players"][0]["distance_m"] = -1
    out2 = analytics.summarize(f)
    assert out2["heatmaps"]["left"] == analytics.heatmap(f, "left")
    assert out2["players"][0]["distance_m"] == round(0.7, 2)


def test_inputs_not_mutated_with_tuples_and_invalid_positions():
    f = [frame(0, [person(1, (0.0, 0.0)), person(2, [1.0, "x"])]),
         frame(1, [person(1, (0.2, 0.0))]),
         frame(2, [person(1, [float("nan"), 0.0])])]
    f[0]["ball"] = {"xy": [0.0, 0.0]}
    f[1]["ball"] = None
    f[2]["people"] = None
    before = copy.deepcopy(f)
    analytics.player_stats(f)
    analytics.team_shape(f, "left")
    analytics.team_shape(f, "right")
    analytics.heatmap(f, "left", nx=5, ny=3)
    analytics.heatmap(f, "ball")
    analytics.summarize(f)
    assert f == before


# ---------------------------------------------------------------- scale and concurrency

def test_large_input():
    # 10 left-team players drifting 0.01 m per frame (0.25 m/s) for 1000
    # frames, plus a slow ball.  All positions stay inside the pitch, so the
    # heatmap counts every appearance.
    n = 1000
    frames = []
    for f in range(n):
        people = [person(i, (i - 4.5) + 0.01 * f, 0.0) for i in range(10)]
        frames.append(frame(f, people, {"xy": [-30.0 + 0.01 * f, 0.0]}))
    out = analytics.summarize(frames)
    assert out["frame_count"] == n
    assert out["first_frame"] == 0 and out["last_frame"] == n - 1
    assert out["duration_s"] == 40.0
    assert [p["id"] for p in out["players"]] == list(range(10))
    for p in out["players"]:
        assert p["frames"] == n
        assert p["seconds"] == 40.0
        # raw distance 999 * 0.01 = 9.99 m, avg speed 0.25 m/s (2-decimal round is exact here)
        assert p["distance_m"] == 9.99
        assert p["avg_speed_ms"] == 0.25
        assert p["max_speed_ms"] == 0.25
        assert p["sprints"] == 0
    assert out["teams"]["left"]["frames"] == n
    assert out["teams"]["right"]["frames"] == 0
    assert total(out["heatmaps"]["left"]) == 10 * n
    assert total(out["heatmaps"]["right"]) == 0
    assert total(out["heatmaps"]["ball"]) == n
    assert out["ball"]["frames_with_ball"] == n
    assert out["ball"]["distance_m"] == pytest.approx(9.99, abs=1e-6)


def test_concurrent_calls_are_consistent():
    frames = []
    for f in range(200):
        frames.append(frame(f, [person(i, 0.3 * f, 0.0, team=("left" if i % 2 else "right"))
                                for i in range(4)], {"xy": [0.1 * f, 0.0]}))
    expected = analytics.summarize(frames)
    results = []
    errors = []

    def work():
        try:
            results.append(analytics.summarize(frames))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 8
    for r in results:
        assert r == expected
