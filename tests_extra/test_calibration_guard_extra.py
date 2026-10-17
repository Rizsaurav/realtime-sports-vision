"""Extra tests: purity, gauge behaviour through the guard, Stats, and the
pipeline wiring (calibration state, guard counters, keyframe picking, and
the foot-filter gating that keeps a lenient first calibration from emptying
the minimap).
"""
import os
import sys
import threading

import numpy as np
from types import SimpleNamespace

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

from sideline.calibration_guard import (
    CalibrationGuard,
    is_valid,
    view_distance,
    view_is_plausible,
)
from sideline.pipeline import LivePipeline, Stats, _CalibState

SIZE = (1920, 1080)


def side_view(x0, y0, w, h, size=SIZE):
    ww, hh = size
    return np.array([[w / ww, 0, x0], [0, h / hh, y0], [0, 0, 1]], float)


H_GOOD = side_view(-30, -17.5, 60, 35)
H_DRIFT = side_view(-29.5, -17.5, 60, 35)
H_GLITCH = side_view(10, -17.5, 60, 35)          # 40 m away


def _pipeline(keyframe_k=10):
    p = LivePipeline(None, {"pipeline": {"keyframe_every": keyframe_k}})
    p.cal_guard = CalibrationGuard()
    p.cal = _CalibState(p.keyframe_k)
    p.cal_lock = threading.Lock()
    p.run_id = 1
    return p


# ------------------------------------------------------------------- purity

def test_pure_functions_are_reproducible():
    for H in (H_GOOD, H_GLITCH, -H_GOOD):
        assert is_valid(H, SIZE) == is_valid(H, SIZE)
        assert view_is_plausible(H, SIZE) == view_is_plausible(H, SIZE)
    assert view_distance(H_GOOD, H_GLITCH, SIZE) == view_distance(H_GOOD, H_GLITCH, SIZE)

def test_update_does_not_mutate_its_inputs():
    g = CalibrationGuard()
    a, b = H_GOOD.copy(), H_GLITCH.copy()
    g.update(a, b, SIZE)
    assert np.array_equal(a, H_GOOD) and np.array_equal(b, H_GLITCH)

# ------------------------------------------------------- gauge invariance

def test_guard_treats_gauge_flipped_views_identically():
    """The calibrator may flip the overall scale of H between keyframes;
    the decisions must not change.  Both guards see the same view sequence,
    one gauge-flipped."""
    seq = [H_GOOD, -H_DRIFT, H_GLITCH]
    g_pos, g_neg = CalibrationGuard(), CalibrationGuard()
    cur_pos = cur_neg = None
    for Hs in seq:
        cur_pos, dp = g_pos.update(Hs, cur_pos, SIZE)
        cur_neg, dn = g_neg.update(-Hs, cur_neg, SIZE)
        assert dp == dn
    assert (g_pos.accepted, g_pos.rejected) == (g_neg.accepted, g_neg.rejected)
    assert (g_pos.accepted, g_pos.rejected) == (2, 1)

def test_plausible_and_valid_are_gauge_invariant():
    for H in (H_GOOD, H_DRIFT, H_GLITCH, -H_GOOD):
        assert is_valid(H, SIZE) == is_valid(-H, SIZE)
        assert view_is_plausible(H, SIZE) == view_is_plausible(-H, SIZE)

# ------------------------------------------------------------- guard extras

def test_guard_reset_clears_remembered_run_and_streak():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    g.update(H_GLITCH, H_GOOD, SIZE)             # remembered, streak 1
    g.reset()
    # a single far candidate after the reset cannot confirm (run is empty)
    H, dec = g.update(H_GLITCH, H_GOOD, SIZE)
    assert dec == "rejected_jump"
    H2, dec2 = g.update(H_GLITCH, H_GOOD, SIZE)  # now the run has two agreeing
    assert dec2 == "accepted_confirmed"

def test_guard_rejected_and_accepted_are_ints():
    g = CalibrationGuard()
    assert isinstance(g.rejected, int)
    assert isinstance(g.accepted, int)
    g.update(H_GLITCH, H_GOOD, SIZE)
    g.update(H_GLITCH, H_GOOD, SIZE)
    assert (g.rejected, g.accepted) == (1, 1)

def test_guard_loose_tolerance_streak_counts_invalid_rejections():
    # invalid rejections extend the streak: after two invalid + one jump
    # rejection the next confirmation uses the loose 4 m tolerance
    bad = H_GOOD.copy()
    bad[0, 0] = np.nan
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    g.update(bad, H_GOOD, SIZE)
    g.update(bad, H_GOOD, SIZE)
    g.update(H_GLITCH, H_GOOD, SIZE)             # 3rd consecutive rejection
    H, dec = g.update(side_view(12.5, -17.5, 60, 35), H_GOOD, SIZE)
    # 2.5 m from the glitch: only the loose tolerance (4 m) confirms it
    assert dec == "accepted_confirmed"

# ------------------------------------------------------------------- stats

def test_stats_snapshot_includes_calibration_counters():
    s = Stats()
    s.accepted_calibrations = 3
    s.rejected_calibrations = 1
    snap = s.snapshot()
    assert snap["accepted_calibrations"] == 3
    assert snap["rejected_calibrations"] == 1

def test_apply_guard_updates_stats_counters():
    p = _pipeline()
    p._apply_guard(H_GOOD, None, True)                       # accepted_first
    p._apply_guard(H_GLITCH, H_GOOD, True)                   # rejected_jump
    p._apply_guard(H_GLITCH, H_GOOD, True)                   # accepted_confirmed
    snap = p.stats.snapshot()
    assert snap["accepted_calibrations"] == 2
    assert snap["rejected_calibrations"] == 1

# ------------------------------------------------------- pipeline cal state

def test_calib_state_carry_keeps_calibration_on_motion_miss():
    cal = _CalibState(10)
    cal.H_cur = H_GOOD
    cal.carry(None)                                          # a miss keeps it
    assert cal.H_cur is H_GOOD
    M = np.array([[1, 0, 0.5], [0, 1, 0], [0, 0, 1]], float)
    cal.carry(M)
    assert np.allclose(cal.H_cur, H_GOOD @ M)

def test_calib_state_motion_prunes_old_entries():
    cal = _CalibState(10)
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    for i in range(70):
        cal.motion(i, M)
    # only the last ~4*keyframe_k entries are kept
    assert 69 in cal.motions and 0 not in cal.motions
    assert len(cal.motions) < 60

# ------------------------------------------------- keyframe picking (wiring)

def test_pick_next_calibration_propagates_and_gaps():
    p = _pipeline(keyframe_k=10)
    cal = p.cal
    M1 = np.array([[1, 0, 1.0], [0, 1, 0], [0, 0, 1]], float)
    cal_results = {5: H_GOOD}
    p.cal_results = cal_results
    p.cal.last_offered = None
    # motions from keyframe 5 to frame 9
    for j in range(6, 10):
        cal.motion(j, M1)
    picked = p._pick_next_calibration(9)
    assert picked is not None
    Hk, k = picked
    assert k == 5
    expected = H_GOOD @ M1 @ M1 @ M1 @ M1
    assert np.allclose(Hk, expected)
    # a motion gap (a kept frame whose motion estimate failed) stops the
    # propagation at the gap: the candidate is still offered (propagated up
    # to the gap) so a sustained flow failure cannot withhold it
    cal2 = _CalibState(10)
    p.cal = cal2
    cal2.motion(6, M1)
    cal2.motion(7, None)                              # motion estimate failed
    cal2.motion(8, M1)
    p.cal_results = {5: H_GOOD}
    picked = p._pick_next_calibration(9)
    assert picked is not None and picked[1] == 5
    assert np.allclose(picked[0], H_GOOD @ M1)
    # a dropped frame (no motion recorded at all) is simply skipped
    cal2b = _CalibState(10)
    p.cal = cal2b
    cal2b.motion(6, M1)
    cal2b.motion(8, M1)                               # frame 7 was dropped
    p.cal_results = {5: H_GOOD}
    picked = p._pick_next_calibration(9)
    assert picked is not None and np.allclose(picked[0], H_GOOD @ M1 @ M1)
    # an offered keyframe is not offered again
    cal3 = _CalibState(10)
    p.cal = cal3
    p.cal_results = {5: H_GOOD, 12: H_GLITCH}
    p.cal.last_offered = 5
    picked = p._pick_next_calibration(12)
    assert picked is not None and picked[1] == 12

def test_rejection_never_loses_the_calibration():
    p = _pipeline(keyframe_k=10)
    cal = p.cal
    # first keyframe: lenient first acceptance
    H_new, dec = p._apply_guard(H_GOOD, cal.H_cur, True)
    cal.H_cur, cal.H_from, cal.strict = H_new, 9, dec != "accepted_first"
    assert dec == "accepted_first"
    assert cal.strict is False
    # a keyframe 40 m away is a jump: rejected, the carried calibration is
    # kept and the request is rewound to the next frame
    H_new2, dec2 = p._apply_guard(H_GLITCH, cal.H_cur, True)
    assert dec2 == "rejected_jump"
    assert np.array_equal(cal.H_cur, H_GOOD)
    idx = 19
    cal.last_req = idx - p.keyframe_k + 1                   # what _cpu sets
    assert cal.last_req == 10
    # the next keyframe agrees with the remembered one: a confirmed cut,
    # and from now on the calibration was accepted by the full rule
    H_new3, dec3 = p._apply_guard(H_GLITCH, cal.H_cur, True)
    assert dec3 == "accepted_confirmed"
    cal.H_cur, cal.H_from, cal.strict = H_new3, 19, dec3 != "accepted_first"
    assert cal.strict is True
    assert p.stats.rejected_calibrations == 1
    assert p.stats.accepted_calibrations == 2

# ------------------------------------------- foot filtering gating (wiring)

def _out_with_one_box():
    return SimpleNamespace(
        xyxy=np.array([[1500.0, 400.0, 1700.0, 800.0]], float),
        tracker_id=np.array([7]),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
    )

def test_lenient_first_calibration_keeps_off_pitch_feet():
    """A person whose feet project outside the pitch + 15 m stays on the
    minimap while the current calibration came from the lenient first
    acceptance (strict=False), and is dropped once the calibration was
    accepted by the full rule (strict=True)."""
    p = _pipeline()
    H = side_view(-80, -40, 200, 100)   # view wider than the pitch
    frame = np.zeros((1080, 1920, 3), np.uint8)
    out = _out_with_one_box()
    rows_lenient = p._project_people(frame, out, H, strict=False)
    rows_strict = p._project_people(frame, out, H, strict=True)
    assert len(rows_lenient) == 1 and len(rows_strict) == 1
    assert rows_lenient[0]["pitch"] is not None
    assert rows_strict[0]["pitch"] is None

def test_project_people_keeps_on_pitch_feet_under_both_modes():
    p = _pipeline()
    H = side_view(-30, -17.5, 60, 35)   # the frame is exactly the pitch area
    frame = np.zeros((1080, 1920, 3), np.uint8)
    out = _out_with_one_box()
    for strict in (True, False):
        rows = p._project_people(frame, out, H, strict=strict)
        assert len(rows) == 1
        assert rows[0]["pitch"] is not None
        assert rows[0]["track"] == 7
        assert rows[0]["cls"] == "player"

def test_project_people_without_calibration_has_no_pitch():
    p = _pipeline()
    frame = np.zeros((1080, 1920, 3), np.uint8)
    rows = p._project_people(frame, _out_with_one_box(), None, strict=True)
    assert len(rows) == 1
    assert rows[0]["pitch"] is None
