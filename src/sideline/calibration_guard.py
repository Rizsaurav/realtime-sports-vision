"""Guards against bad keyframe calibrations reaching the live pipeline.

The keyframe calibrator occasionally returns a wrong image->pitch homography.
These pure functions (numpy only) judge one homography in isolation (valid,
plausible view), measure how far apart two homographies' views are, and
filter foot positions back onto the pitch.  CalibrationGuard is the stateful
filter: small drifts are accepted, a single-frame glitch is ignored, and a
large change is accepted as a camera cut only once a run of agreeing far
candidates confirms it.

Only the lower 60% of a broadcast frame shows the pitch; the top rows show
stands and sky, which project hundreds of metres away, so plausibility is
judged from a 3x3 grid at 10%, 50%, 90% of the width and 40%, 70%, 95% of
the height.  The top 40% of the image never decides anything.  The view
centre is the grid point at 50% width, 70% height.  The jump between two
views is a move of the view as a whole, beyond max_jump_m; the centre and
the grid move together for a real drift, so the view-centre distance is
the usual test.  The 9-point view_distance is the whole-area check: a
candidate whose view centre has barely moved (at most a quarter of
max_jump_m) while the pitch area moved more than max_jump_m is a warp -
a rotation or shear glitch around a pivot near the view centre - and
would otherwise pass as a small drift, so it is a jump too.

A homography is defined only up to a global scale, and the calibrator is
free to flip that scale between keyframes.  Every judgement is therefore
made in the normalised form H/H[2,2]; in that form the sign of a projected
weight is gauge-invariant and says which side of the horizon (the image line
where pitch points go to infinity) the point lies on.  The pitch area is
visible only when its grid points all lie on one side of the horizon;
which of the two sides the camera sits on is not determined by H alone (H
and -H are the same image->pitch map), so no single image point - not even
the image centre, which a real side view can legitimately leave above the
horizon - is pinned to a side, and side-of-horizon judgements are made for
the pitch area as a region, never for one point.

Pitch convention: metres, origin at the centre spot, x in [-52.5, 52.5],
y in [-34, 34].  All functions take a 3x3 image->pitch homography and an
image size (w, h), default 1920x1080.
"""
import numpy as np

PITCH_X = 52.5
PITCH_Y = 34.0
DEFAULT_SIZE = (1920, 1080)
GRID_X = (0.1, 0.5, 0.9)      # width fractions of the pitch-area grid
GRID_Y = (0.4, 0.7, 0.95)     # height fractions: the lower 60% of the frame
ANCHOR = (0.5, 0.7)           # the view-centre point (50% width, 70% height)
ANCHOR_X_MAX = 70.0           # view_is_plausible: max |x| of the view centre, m
ANCHOR_Y_MAX = 45.0           # view_is_plausible: max |y| of the view centre, m
POINT_MAX = 250.0             # view_is_plausible: max grid-point distance from origin, m
AREA_MIN = 100.0              # view_is_plausible: min pitch-area polygon area, m^2
AREA_MAX = 20000.0            # view_is_plausible: max pitch-area polygon area, m^2
LOW_CORNERS = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.4), (0.0, 0.4))  # pitch-area polygon
FIRST_MAX = 120.0             # lenient first acceptance: view-centre radius from origin, m
DET_MIN = 1e-12               # is_valid: |det| must be above this
COND_MAX = 1e9                # is_valid: condition number must stay below this


def _project(H, pts):
    """Homogeneous image->pitch projection of (N, 2) points; (N, 3) result."""
    P = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    Q = np.hstack([P, np.ones((P.shape[0], 1))])
    return Q @ np.asarray(H, dtype=np.float64).T


def _normalise(H):
    """H in the gauge where H[2,2] == 1, or None when that is impossible
    (wrong shape, non-finite entries, H[2,2] == 0).  A homography is defined
    up to scale, so this normalised form is the gauge-invariant
    representation; in it, a projected weight's sign says which side of the
    horizon the point is on."""
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or H[2, 2] == 0.0:
        return None
    return H / H[2, 2]


def is_valid(H, size=DEFAULT_SIZE):
    """True if H is usable: finite entries, |det| above DET_MIN, condition
    number below COND_MAX, H[2,2] != 0, the image centre's H[2,2]-
    normalised projection scale is non-zero (the centre is not on the camera
    plane, so its projection is a proper finite point), and the pitch area
    is a coherent view: its 9 grid points project finite on one side of the
    horizon, so the lower 60% of the image is one connected side of the
    view; a horizon that crosses the lower 60% would leave half of it the
    mirror of the pitch behind the camera plane, which no projection rule
    of the guard can use.  Every check is gauge-invariant: multiplying H by
    any non-zero scalar leaves the normalised weights unchanged, so a
    gauge-flipped copy of a usable homography stays usable.  The side of
    the horizon the camera sits on is not determined by H alone, so it is
    judged per region (the pitch area), never as the raw sign at one
    point."""
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return False
    if abs(np.linalg.det(H)) <= DET_MIN:
        return False
    cond = np.linalg.cond(H)
    if not np.isfinite(cond) or cond >= COND_MAX:
        return False
    Hn = _normalise(H)
    if Hn is None:
        return False
    w, h = size
    centre = Hn[2, 0] * (w / 2.0) + Hn[2, 1] * (h / 2.0) + 1.0
    if abs(centre) <= 1e-9:
        return False
    return _lower_points(H, size) is not None


def footprint(H, size=DEFAULT_SIZE):
    """Project the 4 image corners and the centre; returns (polygon, centre)
    in metres (4-point list and 2-point list), or None if any projection is
    not finite."""
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return None
    w, h = size
    pts = [[0, 0], [w, 0], [w, h], [0, h], [w / 2.0, h / 2.0]]
    Q = _project(H, pts)
    if not np.isfinite(Q).all() or np.any(Q[:, 2] == 0.0):
        return None                        # zero weight: the division would blow up
    P = Q[:, :2] / Q[:, 2:3]
    return P[:4].tolist(), P[4].tolist()


def _lower_points(H, size=DEFAULT_SIZE):
    """Project the 9 pitch-area grid points (10/50/90% width x 40/70/95%
    height); (N, 2) metres or None when the grid does not project: a point
    at the horizon (zero weight), on the far side of the visible pitch area
    (a mix of weight signs: part of the lower image is the mirror of the
    pitch behind the camera), or a non-finite result."""
    Hn = _normalise(H)
    if Hn is None:
        return None
    w, h = size
    pts = [[fx * w, fy * h] for fy in GRID_Y for fx in GRID_X]
    Q = _project(Hn, pts)
    wts = Q[:, 2]
    if np.any(wts == 0.0) or not np.isfinite(Q).all():
        return None
    P = Q[:, :2] / wts[:, None]
    if not np.isfinite(P).all():
        return None
    if not (np.all(wts > 0) or np.all(wts < 0)):
        return None
    return P


def anchor(H, size=DEFAULT_SIZE):
    """The view-centre point (50% width, 70% height) in metres as a 2-list,
    or None when it does not project (zero weight or non-finite)."""
    Hn = _normalise(H)
    if Hn is None:
        return None
    w, h = size
    Q = _project(Hn, [[ANCHOR[0] * w, ANCHOR[1] * h]])[0]
    if Q[2] == 0.0 or not np.isfinite(Q).all():
        return None
    return (Q[:2] / Q[2]).tolist()


def _area(poly):
    P = np.asarray(poly, dtype=np.float64)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _self_intersects(poly):
    """True if the quadrilateral (in vertex order) crosses itself."""
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in poly)

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def cross(a, b, c, d):
        return ((orient(c, d, a) > 0) != (orient(c, d, b) > 0)) and \
               ((orient(a, b, c) > 0) != (orient(a, b, d) > 0))

    return cross(p0, p1, p2, p3) or cross(p1, p2, p3, p0)


def view_is_plausible(H, size=DEFAULT_SIZE):
    """True if the lower 60% of the image is a sane side view of a real
    pitch: the 9 grid points (10/50/90% width x 40/70/95% height) all
    project finite on one side of the horizon; the view centre (50% width,
    70% height) is within |x| <= 70 and |y| <= 45; no grid point is further
    than 250 m from the origin; and the polygon of the lower-edge corners
    (0%,100%), (100%,100%) and the mid-height corners (0%,40%), (100%,40%)
    projects finite, is not self-intersecting, and has an area between 100
    and 20000 square metres.  The top 40% of the image never decides."""
    P = _lower_points(H, size)
    if P is None:
        return False
    a = anchor(H, size)
    if a is None or abs(a[0]) > ANCHOR_X_MAX or abs(a[1]) > ANCHOR_Y_MAX:
        return False
    if np.any(np.linalg.norm(P, axis=1) > POINT_MAX):
        return False
    Hn = _normalise(H)
    w, h = size
    Q = _project(Hn, [[fx * w, fy * h] for fx, fy in LOW_CORNERS])
    if not np.isfinite(Q).all() or np.any(Q[:, 2] == 0.0):
        return False
    poly = (Q[:, :2] / Q[:, 2:3]).tolist()
    if _self_intersects(poly):
        return False
    return AREA_MIN <= _area(poly) <= AREA_MAX


def view_distance(H_a, H_b, size=DEFAULT_SIZE):
    """Largest distance in metres between the projections of the 9
    pitch-area grid points (10/50/90% width x 40/70/95% height) under the
    two homographies; infinity if either projection fails."""
    pa, pb = _lower_points(H_a, size), _lower_points(H_b, size)
    if pa is None or pb is None:
        return float("inf")
    return float(np.max(np.linalg.norm(pa - pb, axis=1)))


def anchor_distance(H_a, H_b, size=DEFAULT_SIZE):
    """Distance in metres between the two views' centres (their 50% width,
    70% height points); infinity if either view centre does not project.
    This is how far the view as a whole moved; the 9-point view_distance is
    the worst case over the whole pitch area, which also reflects rotation
    and zoom, and is the whole-area check of the jump rule."""
    a, b = anchor(H_a, size), anchor(H_b, size)
    if a is None or b is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def filter_foot_points(points, margin=15.0):
    """Boolean list for foot positions in metres (flat [x, y, ...] or (N, 2)):
    True where the point is inside the pitch rectangle extended by `margin`
    metres on every side."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return [bool(abs(x) <= PITCH_X + margin and abs(y) <= PITCH_Y + margin)
            for x, y in P]


class CalibrationGuard:
    """Decides which keyframe homography may replace the carried one.

    update(H_candidate, H_current, size, moving) returns (H_to_use, decision):
      - invalid or implausible candidate (plausibility is only judged once a
        current calibration exists): keep H_current, "rejected_invalid";
      - no current calibration: accept the first candidate that is valid
        and whose view centre is within 120 m of the origin, even if the
        full plausibility rule fails, "accepted_first" (never leave the
        minimap uncalibrated);
      - candidate whose view centre is within max_jump_m of H_current's
        (anchor_distance): accept, "accepted" - unless the view centre has
        barely moved (at most a quarter of max_jump_m) while the whole
        pitch area moved more than max_jump_m (view_distance): that is a
        warp (a rotation or shear glitch around a pivot near the view
        centre), which the view-centre distance alone would let through, so
        it takes the rejected_jump path below;
      - otherwise (view centre farther than max_jump_m, or a warp): keep
        H_current and remember the candidate, "rejected_jump";
        if the last `confirm` remembered candidates in a row (including this
        one) agree with each other within confirm_tol_m it is a real change
        (a camera cut): accept and reset the remembered run,
        "accepted_confirmed".  After 3 consecutive rejections the
        confirmation uses the looser tolerance loose_confirm_tol_m (4.0 m),
        so a persistent genuine change is accepted.
    `moving` records that the camera was moving when the candidate was
    measured; the decision rules above do not depend on it.  An invalid
    candidate is neither remembered nor a reset of the remembered run, but
    it does extend the rejection streak.
    """

    def __init__(self, max_jump_m=8.0, confirm=2, confirm_tol_m=2.0,
                 loose_confirm_tol_m=4.0):
        self.max_jump_m = max_jump_m
        # a view-centre move this small, with the pitch area moved more than
        # max_jump_m, is a warp (a rotation or shear glitch around a pivot
        # near the view centre), not a camera drift; a genuine drift moves
        # the centre and the pitch area together
        self.warp_anchor_m = 0.25 * max_jump_m
        self.confirm = max(1, int(confirm))
        self.confirm_tol_m = confirm_tol_m
        self.loose_confirm_tol_m = loose_confirm_tol_m
        self._pending = []
        self._streak = 0
        self._rejected = 0
        self._accepted = 0

    @property
    def rejected(self):
        return self._rejected

    @property
    def accepted(self):
        return self._accepted

    def reset(self):
        self._pending = []
        self._streak = 0
        self._rejected = 0
        self._accepted = 0

    def _spread(self, Hs, size):
        d = 0.0
        for i in range(len(Hs)):
            for j in range(i + 1, len(Hs)):
                d = max(d, anchor_distance(Hs[i], Hs[j], size))
        return d

    def _is_warp(self, H_candidate, H_current, anchor_m, size):
        """True when the view centre has barely moved (at most
        warp_anchor_m) yet the whole pitch area moved more than
        max_jump_m: a warp (a rotation or shear glitch around a pivot
        near the view centre), which the view-centre distance alone would
        accept as a small drift."""
        if anchor_m > self.warp_anchor_m:
            return False
        return view_distance(H_candidate, H_current, size) > self.max_jump_m

    def update(self, H_candidate, H_current, size=DEFAULT_SIZE, moving=True):
        cand = np.asarray(H_candidate, dtype=np.float64)
        if H_current is not None and not (is_valid(cand, size) and
                                          view_is_plausible(cand, size)):
            self._streak += 1
            self._rejected += 1
            return H_current, "rejected_invalid"
        if H_current is None:
            a = anchor(cand, size)
            if is_valid(cand, size) and a is not None and \
                    float(np.linalg.norm(a)) <= FIRST_MAX:
                self._pending = []
                self._streak = 0
                self._accepted += 1
                return cand, "accepted_first"
            self._streak += 1
            self._rejected += 1
            return H_current, "rejected_invalid"
        a = anchor_distance(cand, H_current, size)
        if a <= self.max_jump_m and not self._is_warp(cand, H_current, a, size):
            self._pending = []
            self._streak = 0
            self._accepted += 1
            return cand, "accepted"
        self._pending.append(cand)
        del self._pending[:-self.confirm]          # only the last `confirm` are ever read
        tail = self._pending[-self.confirm:]
        tol = self.loose_confirm_tol_m if self._streak >= 3 else self.confirm_tol_m
        if len(tail) == self.confirm and self._spread(tail, size) <= tol:
            self._pending = []
            self._streak = 0
            self._accepted += 1
            return cand, "accepted_confirmed"
        self._streak += 1
        self._rejected += 1
        return H_current, "rejected_jump"
