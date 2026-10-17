"""Calibration guard: unit tests for the v2 rules.

The guard judges homographies from the lower 60% of the frame only (a
3x3 grid at 10/50/90% width x 40/70/95% height, view centre at 50%/70%),
is invariant under the homography gauge flip H -> -H, accepts a lenient
first calibration (valid + view centre within 120 m of the origin), and
uses an 8 m default jump limit.
"""
import math

import numpy as np

from sideline.calibration_guard import (
    CalibrationGuard,
    anchor,
    anchor_distance,
    filter_foot_points,
    footprint,
    is_valid,
    view_distance,
    view_is_plausible,
)

SIZE = (1920, 1080)


def side_view(x0, y0, w, h, size=SIZE):
    """Homography mapping a `size` image onto a pitch rectangle
    (x0, y0)-(x0+w, y0+h) in metres (axis-aligned, no perspective)."""
    ww, hh = size
    return np.array([[w / ww, 0, x0], [0, h / hh, y0], [0, 0, 1]], float)


def quad_homography(src, dst):
    """DLT homography mapping the four src image points onto dst (metres)."""
    A = []
    for (u, v), (x, y) in zip(src, dst):
        A.append([u, v, 1, 0, 0, 0, -u * x, -v * x, -x])
        A.append([0, 0, 0, u, v, 1, -u * y, -v * y, -y])
    _, _, V = np.linalg.svd(np.array(A, float))
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]


# A sane side view: 60 x 35 m of pitch across the whole frame.
H_GOOD = side_view(-30, -17.5, 60, 35)
# A small drift (0.5 m to the right).
H_DRIFT = side_view(-30 + 0.5, -17.5, 60, 35)
# A 40 m glitch (a keyframe that locked onto a different view).
H_GLITCH = side_view(-30 + 40, -17.5, 60, 35)
# Gauge-flipped copy of H_GOOD (the calibrator may flip the overall scale
# of H between keyframes): the exact same image->pitch mapping.
H_NEG = -H_GOOD
# A strong perspective: the image centre projects far above the horizon
# (normalised centre scale < 0) yet the lower 60% is split by the horizon
# too (mixed weight signs) -> the pitch area is not visible.
H_FLIP = H_GOOD @ np.array([[1, 0, 0], [0, 1, 0], [-0.001, 0, 1]], float)
# The lower 60% is split by the horizon: its 9 grid points land on both
# sides (mixed normalised weight signs) -> mirror of the pitch behind the
# camera.
H_STRADDLE = H_GOOD @ np.array([[1, 0, 0], [0, 1, 0], [0, 1, -500]], float)
# Garbage in the TOP 40% only: a tilted camera whose horizon crosses the
# frame at v = 250 (the upper rows are sky/mirror), but the lower 60% is a
# clean view of a real pitch area -> must stay plausible.
H_TOPGARBAGE = np.array([[10.0, 0.0, -9600.0], [0.0, 0.0, 2530.0], [0.0, 1.0, -250.0]], float)
# Bow-tie: the 9 grid points are coherent (all in front of the camera) but
# the two lower-edge corners of the polygon are behind it, so the projected
# quadrilateral crosses itself.
H_BOWTIE = np.array([
    [0.03, 0.0, -2.88],
    [0.02, 0.0, -15.12],
    [0.0, -0.01, 10.5],
], float)
# Anchor-boundary views, built in binary-exact arithmetic: the view centre
# (50% width, 70% height) of H_ANCHOR sits exactly at (70, 45), i.e. on
# both bounds; the others move it just outside one bound.
H_ANCHOR = np.array([[0.03125, 0, 40.0], [0, 0.0625, -2.25], [0, 0, 1]], float)
H_ANCHOR_X = np.array([[0.03125, 0, 41.0], [0, 0.0625, -2.25], [0, 0, 1]], float)
H_ANCHOR_Y = np.array([[0.03125, 0, 40.0], [0, 0.0625, -2.15], [0, 0, 1]], float)
H_ANCHOR_YIN = np.array([[0.03125, 0, 40.0], [0, 0.0625, -2.35], [0, 0, 1]], float)
H_ANCHOR_YNEG = np.array([[0.03125, 0, 40.0], [0, 0.0625, -92.25], [0, 0, 1]], float)
H_ANCHOR_YNEGOUT = np.array([[0.03125, 0, 40.0], [0, 0.0625, -92.35], [0, 0, 1]], float)
H_NAN = H_GOOD.copy()
H_NAN[0, 0] = np.nan
# Degenerate (det == 0): the second row collapses.
H_DEGEN = np.array([[0.03, 0, -30.0], [0.0, 0.0, -17.5], [0.0, 0.0, 1.0]], float)
# Ill-conditioned: two almost-identical rows.
H_ILLCOND = H_GOOD.copy()
H_ILLCOND[1] = H_ILLCOND[0] * (1 + 1e-14)
# The image centre lies exactly on the camera plane (normalised centre
# scale 0): its projection is at infinity.
H_ON_PLANE = np.array([[1.0, 0, 0], [0, 1, 0], [1 / 960, 0, -1.0]], float)


def shift(dx, dy=0.0, H=None):
    base = H if H is not None else H_GOOD
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) @ base


# ---------------------------------------------------------------- is_valid

def test_is_valid_accepts_plausible_camera():
    assert is_valid(H_GOOD, SIZE)

def test_is_valid_rejects_degenerate_matrix():
    assert not is_valid(H_DEGEN, SIZE)

def test_is_valid_rejects_nan_matrix():
    assert not is_valid(H_NAN, SIZE)

def test_is_valid_rejects_ill_conditioned_matrix():
    assert not is_valid(H_ILLCOND, SIZE)

def test_is_valid_rejects_centre_on_camera_plane():
    assert not is_valid(H_ON_PLANE, SIZE)

def test_is_valid_is_gauge_invariant():
    # Flipping the overall scale of a homography must not change its
    # validity: H and -H are the same image->pitch mapping.
    assert is_valid(H_NEG, SIZE) == is_valid(H_GOOD, SIZE)
    for H in (H_GOOD, H_DRIFT, H_GLITCH, H_FLIP, H_STRADDLE, H_TOPGARBAGE):
        assert is_valid(-H, SIZE) == is_valid(H, SIZE)

def test_is_valid_ignores_the_raw_gauge_signs():
    # a usable view seen through a flipped gauge has H[2,2] < 0 and a
    # negative raw centre weight, yet the normalised centre scale is 1
    # (not 0), so it must stay valid
    H = -H_GOOD
    assert H[2, 2] < 0
    assert float((H @ np.array([960.0, 540.0, 1.0]))[2]) < 0
    assert is_valid(H, SIZE)

# --------------------------------------------------------------- footprint

def test_footprint_returns_polygon_and_centre():
    poly, centre = footprint(H_GOOD, SIZE)
    assert len(poly) == 4
    assert len(centre) == 2
    assert np.allclose(poly, [[-30, -17.5], [30, -17.5], [30, 17.5], [-30, 17.5]])
    assert np.allclose(centre, [0, 0])

def test_footprint_none_on_nan():
    assert footprint(H_NAN, SIZE) is None

def test_footprint_none_when_weight_is_zero():
    H = np.array([[1, 0, 0], [0, 1, 0], [0, 1.0 / 1080, -1.0]], float)
    assert (H @ np.array([0, 1080, 1.0]))[2] == 0.0   # a corner on the camera plane
    assert footprint(H, SIZE) is None

# ------------------------------------------------------- view_is_plausible

def test_view_is_plausible_accepts_side_view():
    assert view_is_plausible(H_GOOD, SIZE)

def test_view_is_plausible_is_gauge_invariant():
    for H in (H_GOOD, H_DRIFT, H_GLITCH, H_FLIP, H_STRADDLE, H_TOPGARBAGE, H_BOWTIE):
        assert view_is_plausible(-H, SIZE) == view_is_plausible(H, SIZE)

def test_view_is_plausible_anchor_x_boundaries():
    # view centre (50% width, 70% height) at |x| = 70 is in, 71 is out.
    assert view_is_plausible(shift(70, 0), SIZE)
    assert not view_is_plausible(shift(71, 0), SIZE)
    assert view_is_plausible(shift(-70, 0), SIZE)
    assert not view_is_plausible(shift(-71, 0), SIZE)

def test_view_is_plausible_anchor_y_boundaries():
    # the view centre sits exactly at |y| = 45 (in) / 45.1 (out), built in
    # binary-exact arithmetic
    assert anchor(H_ANCHOR, SIZE) == [70.0, 45.0]
    assert view_is_plausible(H_ANCHOR, SIZE)           # (70, 45): on both bounds
    assert view_is_plausible(H_ANCHOR_YIN, SIZE)       # (70, 44.9)
    assert not view_is_plausible(H_ANCHOR_X, SIZE)     # (71, 45)
    assert not view_is_plausible(H_ANCHOR_Y, SIZE)     # (70, 45.1)
    assert view_is_plausible(H_ANCHOR_YNEG, SIZE)      # (70, -45)
    assert not view_is_plausible(H_ANCHOR_YNEGOUT, SIZE)  # (70, -45.1)

def test_view_is_plausible_rejects_flipped_view():
    # the view centre (50%, 70%) lands 700+ m off the pitch
    assert not view_is_plausible(H_FLIP, SIZE)

def test_view_is_plausible_rejects_horizon_splitting_lower_area():
    # mixed weight signs across the 9-point grid: part of the lower image
    # is the mirror of the pitch behind the camera
    assert not view_is_plausible(H_STRADDLE, SIZE)

def test_view_is_plausible_ignores_top_40_percent():
    # the top rows project behind the camera (sky), but the lower 60% is a
    # clean view of the pitch: it must be plausible
    top = H_TOPGARBAGE @ np.array([960.0, 0.0, 1.0])
    assert top[2] < 0.0                       # top centre behind the horizon
    assert view_is_plausible(H_TOPGARBAGE, SIZE)

def _far_view(scale=1.0):
    """A side view whose (0.9, 0.95) grid point sits at scale*250 m from
    the origin (corner x = 74), with the view centre pinned at (70, 44.9)
    and the pitch-area polygon well inside the area limits."""
    Y = scale * math.sqrt(250.0 ** 2 - 74.0 ** 2)   # corner at x = 74
    h = 4.0 * (Y - 44.9)                             # anchor y = 44.9
    return side_view(65.0, 44.9 - 0.7 * h, 10.0, h)

def test_view_is_plausible_point_distance_boundaries():
    # a grid point at 250 m (within 0.025 m) from the origin is in, just
    # over 250 m is out
    assert view_is_plausible(_far_view(0.999999), SIZE)
    assert not view_is_plausible(_far_view(1.000001), SIZE)

def test_view_is_plausible_area_boundaries():
    # pitch-area polygon (lower-edge + mid-height corners) area is
    # 0.6 * w * h: exactly 100 is in, just under is out; exactly 20000 is
    # in, just over is out.
    h_min = 100.0 / (0.6 * 20)
    assert view_is_plausible(side_view(-10, -0.7 * h_min, 20, h_min), SIZE)
    assert not view_is_plausible(side_view(-10, -0.7 * 8.28, 20, 8.28), SIZE)
    h_max = 20000.0 / (0.6 * 64)
    assert view_is_plausible(side_view(-32, -0.7 * h_max, 64, h_max), SIZE)
    assert not view_is_plausible(side_view(-32, -0.7 * 522, 64, 522), SIZE)

def test_view_is_plausible_rejects_non_finite_projection():
    assert not view_is_plausible(H_NAN, SIZE)
    assert not view_is_plausible(H_DEGEN, SIZE)

def test_view_is_plausible_rejects_self_intersecting_polygon():
    # the 9 grid points are coherent and in bounds, but the projected
    # pitch-area quadrilateral is a bow-tie (its lower-edge corners sit
    # behind the camera) -> implausible
    assert view_is_plausible(H_GOOD, SIZE)          # sanity: the helper works
    assert not view_is_plausible(H_BOWTIE, SIZE)

# ----------------------------------------------------------- view_distance

def test_view_distance_same_is_zero():
    assert view_distance(H_GOOD, H_GOOD, SIZE) == 0.0

def test_view_distance_measures_metres():
    assert np.isclose(view_distance(H_GOOD, H_DRIFT, SIZE), 0.5)
    assert np.isclose(view_distance(H_GOOD, H_GLITCH, SIZE), 40.0)

def test_view_distance_uses_the_documented_grid():
    # the distance is the max over the 9 lower-area grid points: verify
    # against a manual projection of those exact points for a warped view
    A = H_GOOD
    B = quad_homography(((0, 0), (1920, 0), (1920, 1080), (0, 1080)),
                        ((-32, -18), (31, -20), (29, 16), (-31, 19)))
    w, h = SIZE
    pts = [[fx * w, fy * h] for fy in (0.4, 0.7, 0.95) for fx in (0.1, 0.5, 0.9)]
    Qa = np.hstack([pts, np.ones((len(pts), 1))]) @ A.T
    Qb = np.hstack([pts, np.ones((len(pts), 1))]) @ B.T
    expected = float(np.max(np.linalg.norm(Qa[:, :2] / Qa[:, 2:3]
                                           - Qb[:, :2] / Qb[:, 2:3], axis=1)))
    assert np.isclose(view_distance(A, B, SIZE), expected)
    # for a warp the worst grid point moves more than the view centre
    assert view_distance(A, B, SIZE) > anchor_distance(A, B, SIZE)

def test_view_distance_infinite_when_projection_fails():
    assert view_distance(H_NAN, H_GOOD, SIZE) == float("inf")
    assert view_distance(H_STRADDLE, H_GOOD, SIZE) == float("inf")

def test_view_distance_is_gauge_invariant():
    a, b = H_DRIFT, H_GLITCH
    assert np.isclose(view_distance(a, b, SIZE), view_distance(-a, b, SIZE))
    assert np.isclose(view_distance(a, b, SIZE), view_distance(a, -b, SIZE))
    assert np.isclose(view_distance(a, b, SIZE), view_distance(-a, -b, SIZE))

def test_view_distance_respects_size_argument():
    # a warped (non-translational) view: moving the grid to a different
    # frame size moves the compared points, so the distance changes
    A = H_GOOD
    B = quad_homography(((0, 0), (1920, 0), (1920, 1080), (0, 1080)),
                        ((-31, -16), (30, -19), (28, 17), (-30, 18)))
    d_small = view_distance(A, B, (1280, 720))
    d_large = view_distance(A, B, (1920, 1080))
    assert d_small != d_large

def test_anchor_distance_measures_view_centre_move():
    assert np.isclose(anchor_distance(H_GOOD, H_GOOD, SIZE), 0.0)
    assert np.isclose(anchor_distance(H_GOOD, H_DRIFT, SIZE), 0.5)
    assert np.isclose(anchor_distance(H_GOOD, H_GLITCH, SIZE), 40.0)
    assert anchor_distance(H_NAN, H_GOOD, SIZE) == float("inf")

def test_anchor_is_gauge_invariant():
    a, b = anchor(H_GOOD, SIZE), anchor(-H_GOOD, SIZE)
    assert np.allclose(a, b)
    assert np.allclose(a, [0.0, 7.0])

# ------------------------------------------------------------------ the guard

def test_guard_accepts_first_plausible_candidate():
    g = CalibrationGuard()
    H, dec = g.update(H_GOOD, None, SIZE)
    assert dec == "accepted_first"
    assert np.array_equal(H, H_GOOD)
    assert g.accepted == 1 and g.rejected == 0

def test_guard_first_acceptance_is_lenient_on_plausibility():
    # no current calibration: validity + view centre within 120 m of the
    # origin is enough, even if the full plausibility rule fails (here the
    # view centre is 100 m from the origin, outside the 70 m anchor limit)
    g = CalibrationGuard()
    Hc = shift(100, 0)
    assert is_valid(Hc, SIZE) and not view_is_plausible(Hc, SIZE)
    H, dec = g.update(Hc, None, SIZE)
    assert dec == "accepted_first"
    assert np.array_equal(H, Hc)

def test_guard_first_candidate_too_far_is_rejected():
    g = CalibrationGuard()
    Hc = shift(130, 0)
    assert np.linalg.norm(anchor(Hc, SIZE)) > 120.0
    H, dec = g.update(Hc, None, SIZE)
    assert dec == "rejected_invalid"
    assert H is None
    assert g.rejected == 1 and g.accepted == 0
    # the next sane candidate is still the first acceptance
    H2, dec2 = g.update(H_GOOD, None, SIZE)
    assert dec2 == "accepted_first"

def test_guard_invalid_first_candidate_rejected():
    for bad in (H_NAN, H_DEGEN, H_ON_PLANE):
        g = CalibrationGuard()
        H, dec = g.update(bad, None, SIZE)
        assert dec == "rejected_invalid"
        assert H is None

def test_guard_accepts_small_drift():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    H, dec = g.update(H_DRIFT, H_GOOD, SIZE)
    assert dec == "accepted"
    assert np.array_equal(H, H_DRIFT)

def test_guard_max_jump_default_is_eight_metres():
    g = CalibrationGuard()
    assert g.max_jump_m == 8.0
    g.update(H_GOOD, None, SIZE)
    H, dec = g.update(shift(8.0), H_GOOD, SIZE)
    assert dec == "accepted"                       # boundary is inclusive
    H2, dec2 = g.update(shift(8.5), H_GOOD, SIZE)
    assert dec2 == "rejected_jump"                 # 8.5 > 8.0

def test_guard_rejects_single_glitch_and_stays_continuous():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    H, dec = g.update(H_GLITCH, H_GOOD, SIZE)
    assert dec == "rejected_jump"
    assert np.array_equal(H, H_GOOD)
    assert g.rejected == 1
    # the next sane keyframe is still accepted
    H2, dec2 = g.update(H_DRIFT, H_GOOD, SIZE)
    assert dec2 == "accepted"
    assert np.array_equal(H2, H_DRIFT)

def test_guard_two_agreeing_far_candidates_are_a_cut():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    H, d1 = g.update(shift(40.0), H_GOOD, SIZE)
    assert d1 == "rejected_jump"
    H2, d2 = g.update(shift(40.5), H_GOOD, SIZE)  # agrees within 0.5 m
    assert d2 == "accepted_confirmed"
    assert np.array_equal(H2, shift(40.5))

def test_guard_cut_accepts_the_new_view():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    cut = shift(40.0)
    g.update(cut, H_GOOD, SIZE)                    # remembered
    H, dec = g.update(shift(40.5), H_GOOD, SIZE)   # agrees within 0.5 m
    assert dec == "accepted_confirmed"
    assert np.array_equal(H, shift(40.5))
    # the remembered run is reset: one far candidate alone is rejected again
    H2, dec2 = g.update(shift(70.0), H, SIZE)
    assert dec2 == "rejected_jump"

def test_guard_disagreeing_far_candidates_never_confirm():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    g.update(shift(40, 10), H_GOOD, SIZE)
    H, dec = g.update(shift(45, 15), H_GOOD, SIZE)
    # 7.07 m apart: a moving glitch, not a cut
    assert dec == "rejected_jump"
    assert np.array_equal(H, H_GOOD)

def test_guard_alternating_glitches_never_confirm():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    g.update(H_GLITCH, H_GOOD, SIZE)               # remembered
    H, dec = g.update(H_DRIFT, H_GOOD, SIZE)       # a sane frame: run cleared
    assert dec == "accepted"
    g.update(H_GLITCH, H_DRIFT, SIZE)              # remembered again
    H2, dec2 = g.update(shift(44.0), H_DRIFT, SIZE)  # 4 m from the glitch
    assert dec2 == "rejected_jump"
    assert np.array_equal(H2, H_DRIFT)

def test_guard_confirm_tail_must_agree_pairwise():
    g = CalibrationGuard(confirm=3)
    g.update(H_GOOD, None, SIZE)
    g.update(shift(40), H_GOOD, SIZE)
    g.update(shift(50), H_GOOD, SIZE)              # [40, 50]: 10 m apart
    H, dec = g.update(shift(50.5), H_GOOD, SIZE)   # 40 vs 50.5 still disagree
    assert dec == "rejected_jump"
    H2, dec2 = g.update(shift(51.0), H_GOOD, SIZE) # [50, 50.5, 51] agree within 1 m
    assert dec2 == "accepted_confirmed"

def test_guard_only_last_confirm_candidates_decide():
    g = CalibrationGuard(confirm=2)
    g.update(H_GOOD, None, SIZE)
    g.update(shift(40), H_GOOD, SIZE)
    g.update(shift(50), H_GOOD, SIZE)              # [40, 50] disagree: no confirm
    H, dec = g.update(shift(50.5), H_GOOD, SIZE)   # [50, 50.5] agree -> confirmed
    assert dec == "accepted_confirmed"

def test_guard_confirm_tolerance_boundary():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    g.update(shift(40), H_GOOD, SIZE)
    H, dec = g.update(shift(42.0), H_GOOD, SIZE)   # exactly 2.0 m: inclusive
    assert dec == "accepted_confirmed"
    g2 = CalibrationGuard()
    g2.update(H_GOOD, None, SIZE)
    g2.update(shift(40), H_GOOD, SIZE)
    H2, dec2 = g2.update(shift(42.5), H_GOOD, SIZE)  # 2.5 m > 2.0 m
    assert dec2 == "rejected_jump"

def test_guard_loose_tolerance_after_three_rejections():
    # 2.5 m steps: never confirm at the 2.0 m tolerance, but the 4th
    # candidate (after 3 consecutive rejections) confirms at 4.0 m.
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    H, d1 = g.update(shift(40.0), H_GOOD, SIZE)
    H, d2 = g.update(shift(42.5), H_GOOD, SIZE)
    H, d3 = g.update(shift(45.0), H_GOOD, SIZE)
    H, d4 = g.update(shift(47.5), H_GOOD, SIZE)
    assert (d1, d2, d3, d4) == ("rejected_jump", "rejected_jump",
                                "rejected_jump", "accepted_confirmed")
    assert g.rejected == 3 and g.accepted == 2

def test_guard_rejection_streak_resets_on_acceptance():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    g.update(shift(40.0), H_GOOD, SIZE)
    g.update(shift(42.5), H_GOOD, SIZE)            # 2 rejections, 2.5 m apart
    H, dec = g.update(H_DRIFT, H_GOOD, SIZE)        # sane: accepted, streak reset
    assert dec == "accepted"
    # after the reset only the base tolerance applies: 2.5 m steps do not
    # confirm
    g.update(shift(40.5, 0, H_DRIFT), H_DRIFT, SIZE)
    H2, dec2 = g.update(shift(43.0, 0, H_DRIFT), H_DRIFT, SIZE)
    assert dec2 == "rejected_jump"

def test_guard_confirm_one():
    g = CalibrationGuard(confirm=1)
    g.update(H_GOOD, None, SIZE)
    H, dec = g.update(H_GLITCH, H_GOOD, SIZE)      # a single far candidate is a cut
    assert dec == "accepted_confirmed"
    assert np.array_equal(H, H_GLITCH)

def test_guard_confirm_three():
    g = CalibrationGuard(confirm=3)
    g.update(H_GOOD, None, SIZE)
    H, d1 = g.update(shift(40.0), H_GOOD, SIZE)
    H, d2 = g.update(shift(40.5), H_GOOD, SIZE)
    assert d1 == d2 == "rejected_jump"
    H, d3 = g.update(shift(41.0), H_GOOD, SIZE)
    assert d3 == "accepted_confirmed"

def test_guard_counters_count_every_decision():
    g = CalibrationGuard()
    g.update(H_NAN, None, SIZE)                            # rejected_invalid
    g.update(H_GOOD, None, SIZE)                           # accepted_first
    g.update(H_DRIFT, H_GOOD, SIZE)                        # accepted
    g.update(H_GLITCH, H_DRIFT, SIZE)                      # rejected_jump
    g.update(shift(40.5, 0, H_DRIFT), H_DRIFT, SIZE)       # accepted_confirmed
    assert g.rejected == 2
    assert g.accepted == 3

def test_guard_rejection_keeps_current_and_reset_clears():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    g.update(H_GLITCH, H_GOOD, SIZE)
    H, dec = g.update(H_GLITCH, H_GOOD, SIZE)
    assert dec == "accepted_confirmed"
    g.update(shift(70.0), H, SIZE)
    assert g.rejected == 2
    g.reset()
    assert g.rejected == 0 and g.accepted == 0
    # after reset the remembered run is gone: one far candidate alone
    H2, dec2 = g.update(shift(70.0), H, SIZE)
    assert dec2 == "rejected_jump"

def test_guard_moving_argument_does_not_change_decision():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    H, d_moving = g.update(H_GLITCH, H_GOOD, SIZE, moving=True)
    g.reset()
    g.update(H_GOOD, None, SIZE)
    H2, d_still = g.update(H_GLITCH, H_GOOD, SIZE, moving=False)
    assert d_moving == d_still == "rejected_jump"

def test_guard_invalid_candidate_does_not_join_or_reset_run():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    g.update(shift(40.0), H_GOOD, SIZE)            # remembered
    H, dec = g.update(H_NAN, H_GOOD, SIZE)         # invalid: neither remembered
    assert dec == "rejected_invalid"               # nor a reset of the run
    H2, dec2 = g.update(shift(40.5), H_GOOD, SIZE) # the remembered run survives
    assert dec2 == "accepted_confirmed"

def test_guard_repeated_identical_far_candidate_confirms():
    g = CalibrationGuard()
    g.update(H_GOOD, None, SIZE)
    g.update(H_GLITCH, H_GOOD, SIZE)
    H, dec = g.update(H_GLITCH, H_GOOD, SIZE)
    assert dec == "accepted_confirmed"
    assert np.array_equal(H, H_GLITCH)

def test_guard_returns_held_or_new_value():
    g = CalibrationGuard()
    H, dec = g.update(H_DRIFT, H_GOOD, SIZE)
    assert np.array_equal(H, H_DRIFT)              # accepted: the candidate
    H2, dec2 = g.update(H_GLITCH, H_DRIFT, SIZE)
    assert np.array_equal(H2, H_DRIFT)             # rejected: the held value

# --------------------------------------------------------- filter_foot_points

def test_filter_foot_points_includes_margin():
    # default margin is 15 m on every side
    assert filter_foot_points([0, 0]) == [True]
    assert filter_foot_points([52.5, 34]) == [True]          # the box itself
    assert filter_foot_points([67.5, 0]) == [True]           # 52.5 + 15
    assert filter_foot_points([67.6, 0]) == [False]
    assert filter_foot_points([0, 49.0]) == [True]           # 34 + 15
    assert filter_foot_points([0, 49.1]) == [False]
    assert filter_foot_points([-52.5, -34]) == [True]

def test_filter_foot_points_custom_margin():
    assert filter_foot_points([58.0, 0], margin=3.0) == [False]
    assert filter_foot_points([58.0, 0], margin=15.0) == [True]

def test_filter_foot_points_accepts_shapes_and_returns_bools():
    pts = np.array([[0.0, 0.0], [99.0, 0.0], [0.0, 120.0]])
    res = filter_foot_points(pts)
    assert res == [True, False, False]
    assert all(isinstance(v, bool) for v in res)
    assert filter_foot_points([]) == []

# ------------------------------------------------- the 300-frame simulation

def _sim_frames(kframe_k=10, n=300, glitch_frames=(50, 90, 130, 170, 210, 250)):
    """Keyframes every kframe_k: a slow sinusoidal drift plus 40 m glitches
    at the listed keyframe indices."""
    for k in range(0, n, kframe_k):
        t = k / 10.0
        drift = 0.5 * math.sin(t / 7.0)
        if k in glitch_frames:
            H = side_view(-30 + 40.0, -17.5, 60, 35)
        else:
            H = side_view(-30 + drift, -17.5, 60, 35)
        yield H

def test_simulation_guard_keeps_trajectory_continuous():
    guard = CalibrationGuard()
    H_cur = None
    prev = None
    max_step = 0.0
    n_acc = n_rej = 0
    for H in _sim_frames():
        H_cur, dec = guard.update(H, H_cur, SIZE)
        if dec.startswith("accepted"):
            n_acc += 1
            if prev is not None:
                max_step = max(max_step, anchor_distance(H_cur, prev, SIZE))
            prev = H_cur
        else:
            n_rej += 1
    assert n_rej == 6                                   # exactly the glitches
    assert n_acc > 0
    assert max_step < 1.0                               # per keyframe, with the guard

def test_simulation_without_guard_has_large_jumps():
    prev = None
    max_step = 0.0
    for H in _sim_frames():
        if prev is not None:
            max_step = max(max_step, anchor_distance(H, prev, SIZE))
        prev = H
    assert max_step > 20.0
