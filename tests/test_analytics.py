import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

from sideline import analytics  # noqa: E402

NO_XY = "no-xy"


# ---------------------------------------------------------------- helpers

def person(pid, x, y=0.0, role="player", team="left", jersey=None):
    p = {"id": pid, "role": role, "team": team, "jersey": jersey, "box": [0, 0, 10, 20]}
    p["xy"] = None if x is None else [x, y]
    return p


def frame(n, people=(), ball=None):
    """ball: None, NO_XY, or an (x, y) pair."""
    if ball is None:
        b = None
    elif ball == NO_XY:
        b = {"box": [0, 0, 5, 5], "xy": None}
    else:
        b = {"box": [0, 0, 5, 5], "xy": [ball[0], ball[1]]}
    return {"frame": n, "delay": 0, "people": list(people), "ball": b, "colors": None}


def merge(*series):
    by = {}
    for s in series:
        for f in s:
            if f["frame"] in by:
                by[f["frame"]]["people"].extend(f["people"])
                if f["ball"] is not None:
                    by[f["frame"]]["ball"] = f["ball"]
            else:
                by[f["frame"]] = copy.deepcopy(f)
    return [by[k] for k in sorted(by)]


def track(pid, xs, start=0, y=0.0, **kw):
    """One person at consecutive frames with the given x positions."""
    return [frame(start + i, [person(pid, x, y, **kw)]) for i, x in enumerate(xs)]


def line(n_steps, step, x0=0.0):
    return [x0 + i * step for i in range(n_steps + 1)]


def at(pid, pairs, **kw):
    """One person at the given (frame, x) pairs."""
    return [frame(f, [person(pid, x, 0.0, **kw)]) for f, x in pairs]


def stats(frames, fps=25, pid=1):
    return [s for s in analytics.player_stats(frames, fps) if s["id"] == pid][0]


def ball_frames(pairs):
    return [frame(f, [], (x, 0.0)) for f, x in pairs]


# ---------------------------------------------------------------- distance and speed

def test_constant_speed():
    s = stats(track(1, line(10, 0.2)))
    assert s["frames"] == 11
    assert s["distance_m"] == pytest.approx(2.0)
    assert s["avg_speed_ms"] == pytest.approx(5.0)
    assert s["max_speed_ms"] == pytest.approx(5.0)
    assert s["sprints"] == 0


def test_frames_and_seconds():
    s = stats(track(1, [0.0] * 50))
    assert s["frames"] == 50
    assert s["seconds"] == pytest.approx(2.0)
    assert s["distance_m"] == 0.0
    assert s["avg_speed_ms"] == 0.0
    assert s["max_speed_ms"] == 0.0


def test_stats_keys_and_types():
    s = stats(track(1, line(3, 0.2), jersey="7"))
    assert set(s) == {"id", "role", "team", "jersey", "frames", "seconds", "distance_m",
                      "avg_speed_ms", "max_speed_ms", "sprints"}
    assert isinstance(s["frames"], int) and isinstance(s["sprints"], int)
    assert s["jersey"] == "7"


def test_gap_of_five_is_counted():
    s = stats(at(1, [(0, 0.0), (5, 1.0)]))
    assert s["distance_m"] == pytest.approx(1.0)
    assert s["avg_speed_ms"] == pytest.approx(5.0)
    assert s["max_speed_ms"] == pytest.approx(5.0)


def test_gap_of_six_is_discarded():
    s = stats(at(1, [(0, 0.0), (6, 1.2)]))
    assert s["frames"] == 2
    assert s["distance_m"] == 0.0
    assert s["avg_speed_ms"] == 0.0
    assert s["max_speed_ms"] == 0.0


def test_gap_in_the_middle_only_loses_that_step():
    s = stats(at(1, [(0, 0.0), (1, 0.2), (7, 5.0), (8, 5.2)]))
    assert s["distance_m"] == pytest.approx(0.4)
    assert s["avg_speed_ms"] == pytest.approx(5.0)


def test_speed_uses_frame_gap():
    fast = stats(at(1, [(0, 0.0), (2, 1.0)]))   # 12.5 m/s, discarded
    assert fast["distance_m"] == 0.0
    slow = stats(at(1, [(0, 0.0), (2, 0.5)]))   # 6.25 m/s, kept
    assert slow["distance_m"] == pytest.approx(0.5)
    assert slow["max_speed_ms"] == pytest.approx(6.25)


def test_jump_is_rejected_entirely():
    s = stats(at(1, [(0, 0.0), (1, 0.6), (2, 0.8)]))  # 15 m/s then 5 m/s
    assert s["distance_m"] == pytest.approx(0.2)
    assert s["max_speed_ms"] == pytest.approx(5.0)
    assert s["avg_speed_ms"] == pytest.approx(5.0)


def test_speed_exactly_twelve_is_kept():
    s = stats(at(1, [(0, 0.0), (1, 3.0)]), fps=4)
    assert s["distance_m"] == pytest.approx(3.0)
    assert s["max_speed_ms"] == pytest.approx(12.0)


def test_speed_above_twelve_is_dropped():
    s = stats(at(1, [(0, 0.0), (1, 3.25)]), fps=4)   # 13 m/s
    assert s["distance_m"] == 0.0
    assert s["max_speed_ms"] == 0.0


def test_fps_argument_changes_speed():
    s = stats(track(1, line(4, 0.5)), fps=10)
    assert s["max_speed_ms"] == pytest.approx(5.0)
    assert s["seconds"] == pytest.approx(0.5)


def test_diagonal_distance():
    f = [frame(0, [person(1, 0.0, 0.0)]), frame(1, [person(1, 0.15, 0.2)])]
    s = stats(f)
    assert s["distance_m"] == pytest.approx(0.25)
    assert s["max_speed_ms"] == pytest.approx(6.25)
    assert s["avg_speed_ms"] == pytest.approx(6.25)


def test_avg_speed_is_distance_over_counted_time():
    # 0.2 m in 1 frame (5 m/s) and 0.5 m in 2 frames (6.25 m/s): 0.7 m over 0.12 s
    s = stats(at(1, [(0, 0.0), (1, 0.2), (3, 0.7)]))
    assert s["distance_m"] == pytest.approx(0.7)
    assert s["avg_speed_ms"] == pytest.approx(0.7 / 0.12)
    assert s["max_speed_ms"] == pytest.approx(6.25)


# ---------------------------------------------------------------- sprints

def test_sprint_detected_at_one_second():
    s = stats(track(1, line(25, 0.3)))      # 7.5 m/s for 25 frames
    assert s["sprints"] == 1
    assert s["max_speed_ms"] == pytest.approx(7.5)


def test_sprint_too_short():
    assert stats(track(1, line(24, 0.3)))["sprints"] == 0


def test_sprint_needs_speed_of_seven():
    s = stats(track(1, line(50, 0.26)))     # 6.5 m/s for two seconds
    assert s["sprints"] == 0
    assert s["max_speed_ms"] == pytest.approx(6.5)


def test_sprint_boundary_values_exact():
    # fps 4: 1.75 m per frame is exactly 7 m/s, four steps last exactly 1.0 s
    assert stats(track(1, line(4, 1.75)), fps=4)["sprints"] == 1
    assert stats(track(1, line(3, 1.75)), fps=4)["sprints"] == 0
    assert stats(track(1, line(4, 1.5)), fps=4)["sprints"] == 0   # 6 m/s


def test_two_sprints_separated_by_slow_step():
    xs = line(25, 0.3)
    xs = xs + [xs[-1] + 0.1] + [xs[-1] + 0.1 + 0.3 * i for i in range(1, 26)]
    s = stats(track(1, xs))
    assert s["sprints"] == 2


def test_slow_step_splits_a_run():
    xs = line(20, 0.3)
    xs = xs + [xs[-1] + 0.1] + [xs[-1] + 0.1 + 0.3 * i for i in range(1, 21)]
    assert stats(track(1, xs))["sprints"] == 0


def test_discarded_jump_ends_a_run():
    xs = line(15, 0.3)
    xs = xs + [xs[-1] + 0.6] + [xs[-1] + 0.6 + 0.3 * i for i in range(1, 16)]
    s = stats(track(1, xs))
    assert s["sprints"] == 0
    assert s["distance_m"] == pytest.approx(30 * 0.3)


def test_frame_gap_beyond_five_ends_a_run():
    first = [(i, 0.3 * i) for i in range(16)]
    second = [(21 + i, 5.0 + 0.3 * i) for i in range(16)]
    s = stats(at(1, first + second))
    assert s["sprints"] == 0
    assert s["distance_m"] == pytest.approx(30 * 0.3)


def test_sprint_made_of_gap_steps():
    s = stats(at(1, [(5 * i, 1.5 * i) for i in range(6)]))   # 7.5 m/s, 0.2 s per step
    assert s["sprints"] == 1
    assert s["distance_m"] == pytest.approx(7.5)
    s4 = stats(at(1, [(5 * i, 1.5 * i) for i in range(5)]))
    assert s4["sprints"] == 0


def test_long_sprint_counts_once():
    assert stats(track(1, line(60, 0.3)))["sprints"] == 1


# ---------------------------------------------------------------- identity, modes

def test_players_are_grouped_by_id_and_sorted():
    f = merge(track(9, line(4, 0.2)), track(2, line(4, 0.1), y=5.0), track(5, [1.0] * 5))
    res = analytics.player_stats(f)
    assert [r["id"] for r in res] == [2, 5, 9]
    assert res[0]["distance_m"] == pytest.approx(0.4)
    assert res[1]["distance_m"] == 0.0
    assert res[2]["distance_m"] == pytest.approx(0.8)


def test_referee_is_listed_in_stats():
    f = track(3, line(2, 0.1), role="referee", team=None)
    s = stats(f, pid=3)
    assert s["role"] == "referee" and s["team"] is None and s["jersey"] is None


def test_role_most_common():
    f = [frame(0, [person(1, 0.0, role="goalkeeper")]),
         frame(1, [person(1, 0.0, role="player")]),
         frame(2, [person(1, 0.0, role="player")])]
    assert stats(f)["role"] == "player"


def test_tie_goes_to_smallest_value():
    f = [frame(0, [person(1, 0.0, role="player", team="right")]),
         frame(1, [person(1, 0.0, role="goalkeeper", team="left")])]
    s = stats(f)
    assert s["role"] == "goalkeeper"
    assert s["team"] == "left"


def test_null_never_beats_non_null():
    f = [frame(i, [person(1, 0.0, team=None)]) for i in range(3)]
    f.append(frame(3, [person(1, 0.0, team="right")]))
    assert stats(f)["team"] == "right"


def test_all_null_gives_none():
    f = [frame(i, [person(1, 0.0, team=None, jersey=None)]) for i in range(3)]
    s = stats(f)
    assert s["team"] is None and s["jersey"] is None


def test_jersey_is_compared_as_string():
    f = [frame(0, [person(1, 0.0, jersey="9")]), frame(1, [person(1, 0.0, jersey="10")])]
    assert stats(f)["jersey"] == "10"
    g = [frame(0, [person(1, 0.0, jersey="9")]), frame(1, [person(1, 0.0, jersey="9")]),
         frame(2, [person(1, 0.0, jersey="10")])]
    assert stats(g)["jersey"] == "9"


def test_mode_counts_appearances_without_position():
    f = [frame(0, [person(1, None, team="right")]), frame(1, [person(1, None, team="right")]),
         frame(2, [person(1, 0.0, team="left")])]
    assert stats(f)["team"] == "right"


# ---------------------------------------------------------------- null positions

def test_null_xy_counts_as_frame_but_not_position():
    s = stats([frame(0, [person(1, None)]), frame(1, [person(1, 0.0)]), frame(2, [person(1, 0.2)])])
    assert s["frames"] == 3
    assert s["distance_m"] == pytest.approx(0.2)


def test_step_jumps_over_missing_position():
    f = [frame(0, [person(1, 0.0)]), frame(1, [person(1, None)]), frame(2, [person(1, 0.4)])]
    s = stats(f)
    assert s["frames"] == 3
    assert s["distance_m"] == pytest.approx(0.4)
    assert s["max_speed_ms"] == pytest.approx(5.0)


def test_missing_xy_key_and_long_hole():
    p = person(1, 0.0)
    del p["xy"]
    f = [frame(0, [person(1, 0.0)]), frame(1, [p])]
    f += [frame(i, [person(1, None)]) for i in range(2, 7)]
    f.append(frame(7, [person(1, 0.5)]))
    s = stats(f)
    assert s["frames"] == 8
    assert s["distance_m"] == 0.0     # frame gap 7


def test_person_without_people_key_and_ball_key():
    f = [{"frame": 0}, {"frame": 1, "people": None, "ball": None}]
    assert analytics.player_stats(f) == []
    assert analytics.summarize(f)["frame_count"] == 2


# ---------------------------------------------------------------- duplicates and ordering

def test_duplicate_frame_last_wins():
    f = [frame(0, [person(1, 0.0)]), frame(1, [person(1, 5.0)]), frame(1, [person(1, 0.2)])]
    s = stats(f)
    assert s["frames"] == 2
    assert s["distance_m"] == pytest.approx(0.2)


def test_duplicate_frame_counts_once_in_summary():
    f = [frame(4, [person(1, 0.0)]), frame(4, [person(1, 0.0)]), frame(5, [])]
    out = analytics.summarize(f)
    assert out["frame_count"] == 2
    assert out["players"][0]["frames"] == 1


def test_duplicate_id_in_frame_first_wins():
    f = [frame(0, [person(1, 0.0), person(1, 30.0)]), frame(1, [person(1, 0.2)])]
    s = stats(f)
    assert s["frames"] == 2
    assert s["distance_m"] == pytest.approx(0.2)


def test_unsorted_input_is_ordered_by_frame():
    f = track(1, line(10, 0.2))
    f.reverse()
    assert stats(f)["distance_m"] == pytest.approx(2.0)


# ---------------------------------------------------------------- team shape

def triple(n, pts, **kw):
    return frame(n, [person(10 + i, x, y, **kw) for i, (x, y) in enumerate(pts)])


def test_team_shape_single_frame():
    r = analytics.team_shape([triple(0, [(0, 0), (10, 10), (20, -10)])], "left")
    assert r["frames"] == 1
    assert r["avg_centroid"] == pytest.approx([10.0, 0.0])
    assert r["avg_width"] == pytest.approx(20.0)
    assert r["avg_length"] == pytest.approx(20.0)


def test_team_shape_keys():
    r = analytics.team_shape([triple(0, [(0, 0), (1, 1), (2, 2)])], "left")
    assert set(r) == {"frames", "avg_centroid", "avg_width", "avg_length"}


def test_team_shape_average_over_frames():
    f = [triple(0, [(0, 0), (10, 10), (20, -10)]), triple(1, [(0, 0), (20, 0), (10, 30)])]
    r = analytics.team_shape(f, "left")
    assert r["frames"] == 2
    assert r["avg_centroid"] == pytest.approx([10.0, 5.0])
    assert r["avg_width"] == pytest.approx(25.0)
    assert r["avg_length"] == pytest.approx(20.0)


def test_team_shape_uses_mean_not_midrange():
    r = analytics.team_shape([triple(0, [(0, 0), (0, 0), (30, 0)])], "left")
    assert r["avg_centroid"] == pytest.approx([10.0, 0.0])
    assert r["avg_length"] == pytest.approx(30.0)
    assert r["avg_width"] == pytest.approx(0.0)


def test_team_shape_fewer_than_three_skipped():
    f = [triple(0, [(0, 0), (10, 10)]), triple(1, [(0, 0), (10, 10), (20, 20)])]
    r = analytics.team_shape(f, "left")
    assert r["frames"] == 1
    assert r["avg_length"] == pytest.approx(20.0)


def test_team_shape_none_when_nothing_qualifies():
    r = analytics.team_shape([triple(0, [(0, 0), (10, 10)])], "left")
    assert r == {"frames": 0, "avg_centroid": None, "avg_width": None, "avg_length": None}
    assert analytics.team_shape([], "right") == r


def test_team_shape_ignores_null_positions_when_counting():
    f = [frame(0, [person(1, 0.0), person(2, 5.0), person(3, None)])]
    assert analytics.team_shape(f, "left")["frames"] == 0


def test_team_shape_excludes_goalkeepers_referees_other_team():
    people = [person(1, 0.0, 0.0), person(2, 10.0, 10.0), person(3, 20.0, -10.0),
              person(4, -50.0, 0.0, role="goalkeeper"),
              person(5, 40.0, 30.0, role="referee", team="left"),
              person(6, 45.0, -30.0, team="right")]
    r = analytics.team_shape([frame(0, people)], "left")
    assert r["frames"] == 1
    assert r["avg_centroid"] == pytest.approx([10.0, 0.0])
    assert r["avg_length"] == pytest.approx(20.0)
    assert analytics.team_shape([frame(0, people)], "right")["frames"] == 0


def test_team_shape_right_team():
    f = [triple(0, [(-10, 0), (-20, 4), (-30, 8)], team="right")]
    r = analytics.team_shape(f, "right")
    assert r["avg_centroid"] == pytest.approx([-20.0, 4.0])
    assert r["avg_width"] == pytest.approx(8.0)
    assert r["avg_length"] == pytest.approx(20.0)


# ---------------------------------------------------------------- heatmap

def total(grid):
    return sum(sum(r) for r in grid)


def one(x, y, which="left", **kw):
    return analytics.heatmap([frame(0, [person(1, x, y, **kw)])], which)


def test_heatmap_dimensions_and_empty():
    g = analytics.heatmap([], "left")
    assert len(g) == 14 and all(len(r) == 21 for r in g)
    assert total(g) == 0


@pytest.mark.parametrize("x,y,row,col", [
    (-52.5, -34, 0, 0),
    (52.5, 34, 13, 20),
    (52.5, -34, 0, 20),
    (-52.5, 34, 13, 0),
    (0.0, 0.0, 7, 10),
    (-47.5, -29, 1, 1),
    (-47.6, -30, 0, 0),
    (47.4, 29.5, 13, 19),
    (47.5, 30, 13, 20),
    (-0.1, -0.1, 6, 10),
])
def test_heatmap_cells(x, y, row, col):
    g = one(x, y)
    assert g[row][col] == 1
    assert total(g) == 1


@pytest.mark.parametrize("x,y", [(52.6, 0), (-52.6, 0), (0, 34.1), (0, -35), (100, 100)])
def test_heatmap_out_of_range_ignored(x, y):
    assert total(one(x, y)) == 0


def test_heatmap_counts_accumulate():
    f = [frame(i, [person(1, 1.0, 1.0), person(2, 2.0, 2.0)]) for i in range(3)]
    g = analytics.heatmap(f, "left")
    assert g[7][10] == 6 and total(g) == 6


def test_heatmap_team_and_role_filter():
    people = [person(1, 1.0, 1.0),
              person(2, 1.0, 1.0, role="goalkeeper"),
              person(3, 1.0, 1.0, role="referee", team="left"),
              person(4, 1.0, 1.0, team="right"),
              person(5, 1.0, 1.0, team=None),
              person(6, None)]
    f = [frame(0, people)]
    assert analytics.heatmap(f, "left")[7][10] == 2
    assert total(analytics.heatmap(f, "left")) == 2
    assert analytics.heatmap(f, "right")[7][10] == 1
    assert total(analytics.heatmap(f, "ball")) == 0


def test_heatmap_ball():
    f = [frame(0, [person(1, 0.0, 0.0)], (-52.5, 34)), frame(1, [], None), frame(2, [], NO_XY),
         frame(3, [], (60.0, 0.0)), frame(4, [], (0.0, 0.0))]
    g = analytics.heatmap(f, "ball")
    assert g[13][0] == 1 and g[7][10] == 1 and total(g) == 2


def test_heatmap_duplicate_frame_and_duplicate_id():
    f = [frame(0, [person(1, -50.0, -30.0)]), frame(0, [person(1, 50.0, 30.0)])]
    g = analytics.heatmap(f, "left")
    assert g[13][20] == 1 and total(g) == 1
    g2 = analytics.heatmap([frame(0, [person(1, 1.0, 1.0), person(1, 1.0, 1.0)])], "left")
    assert total(g2) == 1


def test_heatmap_unknown_name():
    with pytest.raises(ValueError):
        analytics.heatmap([], "middle")


def test_heatmap_custom_grid_keeps_ranges():
    f = [frame(0, [person(1, -10.0, 0.0), person(2, 10.0, -5.0)])]
    g = analytics.heatmap(f, "left", nx=2, ny=2)
    assert g == [[0, 1], [1, 0]]


# ---------------------------------------------------------------- ball summary

def test_ball_frames_and_distance():
    f = [frame(0, [], (0.0, 0.0)), frame(1, [], (1.0, 0.0)), frame(2, [], None),
         frame(3, [], NO_XY), frame(4, [], (4.0, 0.0))]
    b = analytics.summarize(f)["ball"]
    assert b["frames_with_ball"] == 3
    assert b["distance_m"] == pytest.approx(4.0)   # 1.0 + 3.0 over a gap of 3 frames (25 m/s)


def test_ball_not_limited_by_player_cap():
    b = analytics.summarize(ball_frames([(0, 0.0), (1, 1.0), (2, 2.0)]))["ball"]
    assert b["distance_m"] == pytest.approx(2.0)


def test_ball_cap_forty():
    b = analytics.summarize(ball_frames([(0, 0.0), (1, 2.0), (2, 3.0)]))["ball"]   # 50 then 25 m/s
    assert b["distance_m"] == pytest.approx(1.0)
    ok = analytics.summarize(ball_frames([(0, 0.0), (1, 1.6)]))["ball"]            # 40 m/s exactly allowed
    assert ok["distance_m"] == pytest.approx(1.6)


def test_ball_gap_rule():
    five = analytics.summarize(ball_frames([(0, 0.0), (5, 5.0)]))["ball"]
    six = analytics.summarize(ball_frames([(0, 0.0), (6, 6.0)]))["ball"]
    assert five["distance_m"] == pytest.approx(5.0)
    assert six["distance_m"] == 0.0 and six["frames_with_ball"] == 2


def test_ball_none_at_all():
    b = analytics.summarize([frame(0), frame(1)])["ball"]
    assert b == {"frames_with_ball": 0, "distance_m": 0.0}


# ---------------------------------------------------------------- summarize

TOP = {"fps", "frame_count", "first_frame", "last_frame", "duration_s", "players", "teams", "heatmaps", "ball"}


def sample():
    return merge(track(1, line(25, 0.3), y=1.0, jersey="10"),
                 track(2, line(25, 0.1), y=-5.0, team="left"),
                 track(3, [10.0] * 26, y=9.0, team="left"),
                 track(4, [-10.0] * 26, y=3.0, team="right", role="goalkeeper"),
                 track(5, [0.0] * 26, role="referee", team=None),
                 [frame(i, [], (i * 0.5, 0.0)) for i in range(26)])


def test_summarize_exact_keys():
    out = analytics.summarize(sample())
    assert set(out) == TOP
    assert set(out["players"][0]) == {"id", "role", "team", "jersey", "frames", "seconds", "distance_m",
                                      "avg_speed_ms", "max_speed_ms", "sprints"}
    assert set(out["teams"]) == {"left", "right"}
    assert set(out["teams"]["left"]) == {"frames", "avg_centroid", "avg_width", "avg_length"}
    assert set(out["heatmaps"]) == {"grid", "left", "right", "ball"}
    assert out["heatmaps"]["grid"] == {"nx": 21, "ny": 14, "x_range": [-52.5, 52.5], "y_range": [-34, 34]}
    assert set(out["ball"]) == {"frames_with_ball", "distance_m"}


def test_summarize_json_roundtrip():
    out = analytics.summarize(sample())
    assert json.loads(json.dumps(out)) == out


def test_summarize_values():
    out = analytics.summarize(sample())
    assert out["fps"] == 25
    assert out["frame_count"] == 26
    assert out["first_frame"] == 0 and out["last_frame"] == 25
    assert out["duration_s"] == 1.04
    assert [p["id"] for p in out["players"]] == [1, 2, 3, 4, 5]
    p1 = out["players"][0]
    assert p1["sprints"] == 1 and p1["distance_m"] == 7.5 and p1["jersey"] == "10"
    assert p1["seconds"] == 1.04
    left = out["teams"]["left"]     # players 1, 2, 3 (the goalkeeper and referee are not used)
    assert left["frames"] == 26
    assert left["avg_centroid"] == [5.0, 1.67]
    assert left["avg_width"] == 14.0
    assert left["avg_length"] == 8.75
    assert out["teams"]["right"]["frames"] == 0
    assert out["ball"] == {"frames_with_ball": 26, "distance_m": 12.5}
    assert sum(sum(r) for r in out["heatmaps"]["left"]) == 26 * 3
    assert sum(sum(r) for r in out["heatmaps"]["right"]) == 26
    assert sum(sum(r) for r in out["heatmaps"]["ball"]) == 26


def test_summarize_teams_numbers():
    f = [triple(0, [(0, 0), (10, 10), (20, -10)]), triple(1, [(0, 0), (20, 0), (10, 30)])]
    t = analytics.summarize(f)["teams"]
    assert t["left"] == {"frames": 2, "avg_centroid": [10.0, 5.0], "avg_width": 25.0, "avg_length": 20.0}
    assert t["right"] == {"frames": 0, "avg_centroid": None, "avg_width": None, "avg_length": None}


def test_summarize_rounds_but_stats_do_not():
    f = at(1, [(0, 0.0), (1, 0.123456)])
    raw = stats(f)
    assert raw["distance_m"] == pytest.approx(0.123456, abs=1e-12)
    assert raw["max_speed_ms"] == pytest.approx(3.0864, abs=1e-9)
    p = analytics.summarize(f)["players"][0]
    assert p["distance_m"] == 0.12
    assert p["max_speed_ms"] == 3.09
    assert p["avg_speed_ms"] == 3.09
    assert p["seconds"] == 0.08


def test_summarize_rounds_team_and_ball():
    f = [triple(0, [(0, 0), (0, 0), (1, 0)])]
    f[0]["ball"] = {"xy": [0.0, 0.0]}
    out = analytics.summarize(f + [frame(1, [], (0.123456, 0.0))])
    assert out["teams"]["left"]["avg_centroid"] == [0.33, 0.0]
    assert out["teams"]["left"]["avg_length"] == 1.0
    assert out["ball"]["distance_m"] == 0.12


def test_summarize_frame_numbers_with_holes():
    out = analytics.summarize([frame(30), frame(10), frame(20)])
    assert out["frame_count"] == 3
    assert out["first_frame"] == 10 and out["last_frame"] == 30
    assert out["duration_s"] == 0.12


def test_summarize_fps_argument():
    out = analytics.summarize(track(1, line(4, 0.5)), fps=10)
    assert out["fps"] == 10
    assert out["duration_s"] == 0.5
    assert out["players"][0]["max_speed_ms"] == 5.0


def test_summarize_empty():
    out = analytics.summarize([])
    zeros = [[0] * 21 for _ in range(14)]
    assert out == {
        "fps": 25, "frame_count": 0, "first_frame": None, "last_frame": None, "duration_s": 0.0,
        "players": [],
        "teams": {"left": {"frames": 0, "avg_centroid": None, "avg_width": None, "avg_length": None},
                  "right": {"frames": 0, "avg_centroid": None, "avg_width": None, "avg_length": None}},
        "heatmaps": {"grid": {"nx": 21, "ny": 14, "x_range": [-52.5, 52.5], "y_range": [-34, 34]},
                     "left": zeros, "right": zeros, "ball": zeros},
        "ball": {"frames_with_ball": 0, "distance_m": 0.0},
    }
    json.dumps(out)


def test_empty_inputs_everywhere():
    assert analytics.player_stats([]) == []
    assert total(analytics.heatmap([], "ball")) == 0
    assert analytics.team_shape([], "left")["frames"] == 0


def test_heatmap_rows_are_independent_lists():
    out = analytics.summarize([])
    out["heatmaps"]["left"][0][0] = 5
    assert out["heatmaps"]["right"][0][0] == 0
    assert out["heatmaps"]["ball"][0][0] == 0


# ---------------------------------------------------------------- load_frames

def write(tmp_path, text):
    p = tmp_path / "frames.jsonl"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_load_frames_skips_blank_lines(tmp_path):
    a, b = frame(0, [person(1, 1.0)]), frame(1, [], (1.0, 2.0))
    path = write(tmp_path, json.dumps(a) + "\n\n   \n" + json.dumps(b) + "\n")
    assert analytics.load_frames(path) == [a, b]


def test_load_frames_keeps_file_order(tmp_path):
    path = write(tmp_path, json.dumps({"frame": 5}) + "\n" + json.dumps({"frame": 2}) + "\n")
    assert [f["frame"] for f in analytics.load_frames(path)] == [5, 2]


def test_load_frames_no_trailing_newline(tmp_path):
    assert analytics.load_frames(write(tmp_path, '{"frame": 1}')) == [{"frame": 1}]


def test_load_frames_empty_file(tmp_path):
    assert analytics.load_frames(write(tmp_path, "")) == []
    assert analytics.load_frames(write(tmp_path, "\n  \n")) == []


def test_load_frames_bad_json_line_number(tmp_path):
    path = write(tmp_path, '{"frame": 0}\n\n{"frame": 1\n{"frame": 2}\n')
    with pytest.raises(ValueError, match="line 3"):
        analytics.load_frames(path)


def test_load_frames_bad_json_first_line(tmp_path):
    with pytest.raises(ValueError, match="line 1"):
        analytics.load_frames(write(tmp_path, "not json\n"))


def test_load_frames_missing_frame(tmp_path):
    path = write(tmp_path, '{"frame": 0}\n{"people": []}\n')
    with pytest.raises(ValueError, match="line 2"):
        analytics.load_frames(path)


def test_load_frames_non_object(tmp_path):
    with pytest.raises(ValueError, match="line 2"):
        analytics.load_frames(write(tmp_path, '{"frame": 0}\n[1, 2]\n'))


def test_load_then_summarize(tmp_path):
    f = track(1, line(10, 0.2))
    path = write(tmp_path, "\n".join(json.dumps(x) for x in f) + "\n")
    out = analytics.summarize(analytics.load_frames(path))
    assert out["players"][0]["distance_m"] == 2.0


# ---------------------------------------------------------------- purity

def test_inputs_not_mutated():
    f = merge(sample(), [frame(3, [person(8, None)])])
    f.append(frame(3, [person(9, 1.0)]))
    f.reverse()
    before = copy.deepcopy(f)
    analytics.player_stats(f)
    analytics.team_shape(f, "left")
    analytics.heatmap(f, "left")
    analytics.heatmap(f, "ball")
    analytics.summarize(f)
    assert f == before


def test_no_printing(capsys):
    analytics.summarize(sample())
    analytics.player_stats(sample())
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_repeatable():
    f = sample()
    assert analytics.summarize(f) == analytics.summarize(f)
