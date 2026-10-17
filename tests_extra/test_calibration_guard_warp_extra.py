"""Extra tests for the retained spec rules the guard once missed:

- a centre-pinned warp (the view centre stationary, the pitch area moved
  more than max_jump_m: a rotation or shear glitch around the view centre)
  is a jump, not an acceptance;
- recovery from a lenient first acceptance: when the carried calibration
  is only leniently valid, genuine keyframes of the real view switch over
  through the confirm path;
- is_valid keeps a usable gauge-flipped view valid, stays invalid when the
  horizon crosses the lower 60% (half the pitch area mirrored behind the
  camera plane), and accepts a view whose horizon lies above the frame.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

from sideline.calibration_guard import (
    CalibrationGuard,
    anchor,
    anchor_distance,
    is_valid,
    view_distance,
    view_is_plausible,
)

SIZE = (1920, 1080)


def side_view(x0, y0, w, h, size=SIZE):
    ww, hh = size
    return np.array([[w / ww, 0, x0], [0, h / hh, y0], [0, 0, 1]], float)


def pitch_rotation(theta, cx, cy):
    """Rotation of the pitch plane about (cx, cy), composed on the left of
    an image->pitch homography."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, cx - c * cx + s * cy],
                     [s, c, cy - s * cx - c * cy],
                     [0, 0, 1]], float)


H_GOOD = side_view(-30, -17.5, 60, 35)
A0 = anchor(H_GOOD, SIZE)                       # the view centre (0, 7)


def centre_fixed_warp(deg):
    """A `deg`-degree rotation of the good side view about its view centre:
    the view centre does not move, the pitch area does."""
    return pitch_rotation(np.radians(deg), A0[0], A0[1]) @ H_GOOD


def test_centre_fixed_20_degree_warp_is_rejected_jump():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    warped = centre_fixed_warp(20.0)
    assert anchor_distance(warped, H_GOOD, SIZE) <= 8.0   # the centre held
    assert view_distance(warped, H_GOOD, SIZE) > 8.0      # the pitch area moved
    assert view_is_plausible(warped, SIZE)                # so the rejection
    H, dec = g.update(warped, H_GOOD, SIZE)               # is a jump, not an
    assert dec == "rejected_jump"                         # invalidity
    assert np.array_equal(H, H_GOOD)
    assert g.rejected == 1
    # the next sane keyframe is accepted: the one-frame warp did not stick
    H2, dec2 = g.update(H_GOOD, H_GOOD, SIZE)
    assert dec2 == "accepted"


def test_centre_fixed_30_degree_warp_is_rejected_jump():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    warped = centre_fixed_warp(30.0)
    assert view_distance(warped, H_GOOD, SIZE) > 8.0
    H, dec = g.update(warped, H_GOOD, SIZE)
    assert dec == "rejected_jump"
    assert np.array_equal(H, H_GOOD)


def test_centre_fixed_warp_still_accepts_within_the_budget():
    # a 10-degree centre-fixed warp keeps the pitch area inside the 8 m
    # jump budget: an ordinary small drift
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    warped = centre_fixed_warp(10.0)
    assert view_distance(warped, H_GOOD, SIZE) <= 8.0
    H, dec = g.update(warped, H_GOOD, SIZE)
    assert dec == "accepted"


def test_two_agreeing_centre_fixed_warps_confirm_the_change():
    # a persistent centre-pinned change agrees like any camera cut: the
    # second matching far candidate is accepted and resets the run
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    w1 = centre_fixed_warp(20.0)
    w2 = centre_fixed_warp(21.0)
    assert view_distance(w1, w2, SIZE) <= 2.0
    H, d1 = g.update(w1, H_GOOD, SIZE)
    assert d1 == "rejected_jump"
    H2, d2 = g.update(w2, H_GOOD, SIZE)
    assert d2 == "accepted_confirmed"
    assert np.array_equal(H2, w2)


def test_recovery_from_lenient_first_acceptance():
    # the first acceptance was lenient: the carried calibration is valid
    # but implausible (view centre 100 m from the origin, outside the
    # 70 m bound).  Genuine keyframes of the real view are far from it:
    # the first is a jump, the agreeing second confirms the switch-over
    g = CalibrationGuard()
    bad_first = side_view(70.0, -17.5, 60, 35)
    assert is_valid(bad_first, SIZE) and not view_is_plausible(bad_first, SIZE)
    H, d0 = g.update(bad_first, None, SIZE)
    assert d0 == "accepted_first"
    H2, d1 = g.update(H_GOOD, H, SIZE)
    assert d1 == "rejected_jump"
    h3 = side_view(-30.5, -17.5, 60, 35)
    H3, d2 = g.update(h3, H2, SIZE)
    assert d2 == "accepted_confirmed"
    assert np.array_equal(H3, h3)


def test_is_valid_rejects_horizon_crossing_lower_area():
    # the horizon crosses the frame at v = 600: the lower 60% is split,
    # half of it the mirror of the pitch behind the camera plane -> the
    # homography is not usable
    H = np.array([[60.0 / 1920, 0, -30.0],
                  [0, 35.0 / 1080, -17.5],
                  [0, 1.0 / 600.0, -1.0]], float)
    assert not is_valid(H, SIZE)
    assert not is_valid(-H, SIZE)                    # gauge invariance
    assert not view_is_plausible(H, SIZE)


def test_is_valid_accepts_horizon_above_the_frame():
    # the horizon crosses the frame at v = 250: the lower 60% is one
    # coherent side of the view (a normal side view with stands and sky
    # above) -> usable in either gauge
    H = np.array([[60.0 / 1920, 0, -30.0],
                  [0, 35.0 / 1080, -17.5],
                  [0, 1.0 / 250.0, -1.0]], float)
    assert is_valid(H, SIZE)
    assert is_valid(-H, SIZE)
    assert view_is_plausible(H, SIZE)


def test_is_valid_first_candidate_with_split_lower_area_is_rejected():
    # even the lenient first acceptance needs a coherent pitch area: a
    # split view is rejected_invalid, not accepted_first
    g = CalibrationGuard()
    H = np.array([[60.0 / 1920, 0, -30.0],
                  [0, 35.0 / 1080, -17.5],
                  [0, 1.0 / 600.0, -1.0]], float)
    H_out, dec = g.update(H, None, SIZE)
    assert dec == "rejected_invalid"
    assert H_out is None
