"""Analytics over recorded match frames.

Implements the shapes fixed by docs/WEBAPP_CONTRACT.md: load frames from a
JSONL file, per-player stats, per-team shape, heatmaps and a full summary.

Standard library only, pure functions, inputs are never modified.

Player stats can be computed over jitter-smoothed positions (the ``smooth``
keyword of ``player_stats`` and ``summarize``, off by default): the tracker's
pitch positions contain single-frame jumps, so the raw per-frame steps show
impossible top speeds.  Smoothing, per person and before any step is
computed, replaces every appearance's displacement from the previous
positioned appearance by the median (per coordinate) of the displacements of
the up to 2 * SMOOTH_HALF + 1 consecutive appearances centred on it, and
rebuilds the positions as the running sum of those median displacements from
the first appearance of each tracked segment (segments are runs of positioned
appearances at most MAX_STEP_GAP frames apart; the segment's first and the
track's last positioned appearance keep their recorded positions).  Steady
motion (constant displacement per appearance) is preserved exactly, a
single-frame position jump is removed, and no track is ever mixed with
another person's.
"""

import json
import math

PITCH_X_RANGE = (-52.5, 52.5)
PITCH_Y_RANGE = (-34.0, 34.0)
MAX_STEP_GAP = 5
PLAYER_SPEED_CAP = 12.0
BALL_SPEED_CAP = 40.0
SPRINT_SPEED = 7.0
SPRINT_MIN_SECONDS = 1.0
SMOOTH_HALF = 6  # appearances on each side of the smoothing median window


def load_frames(path):
    """Read frames from a JSONL file.

    path: file path (str); returns the parsed frame dicts as a list in file
    order, one JSON object per line. Blank lines are skipped; a line that is
    not valid JSON, or is JSON that is not an object with a "frame" key,
    raises ValueError naming the line.
    """
    frames = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                raise ValueError("line %d: invalid JSON (%s)" % (lineno, exc)) from None
            if not isinstance(obj, dict) or "frame" not in obj:
                raise ValueError(
                    'line %d: expected a JSON object with a "frame" key, got %r' % (lineno, obj)
                )
            frames.append(obj)
    return frames


def _valid_xy(value):
    """Return (x, y) if value is a valid position, else None."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return None
    return (value[0], value[1])


def _clean_sequence(frames):
    """Deduplicate by frame number (last one wins) and sort by frame."""
    by_frame = {}
    for frame in frames:
        by_frame[frame["frame"]] = frame
    return [by_frame[key] for key in sorted(by_frame)]


def _frame_people(frame):
    """Yield each person dict of a frame at most once (first entry per id)."""
    people = frame.get("people")
    if not isinstance(people, list):
        return
    seen = set()
    for person in people:
        if not isinstance(person, dict) or "id" not in person or person["id"] is None:
            continue
        if person["id"] in seen:
            continue
        seen.add(person["id"])
        yield person


def _most_common(values):
    """Most common non-None value; ties go to the smallest; None if all None."""
    counts = {}
    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    return min(v for v, c in counts.items() if c == best)


def _candidate_steps(points, fps, speed_cap):
    """Yield (gap, d, t, speed, counted) for every candidate step of a track.

    points is a list of (frame, x, y) in frame order; gap in frames, d in
    metres, t in seconds, speed in m/s. A step is not counted when its frame
    gap exceeds MAX_STEP_GAP or its speed exceeds speed_cap.
    """
    for (f0, x0, y0), (f1, x1, y1) in zip(points, points[1:]):
        gap = f1 - f0
        d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        t = gap / fps
        speed = d / t
        counted = gap <= MAX_STEP_GAP and speed <= speed_cap
        yield gap, d, t, speed, counted


def _step_totals(steps):
    """(distance_m, total_time_s, max_speed_ms) over the counted steps.

    steps is a sequence of (gap, d, t, speed, counted) candidate steps.
    """
    distance = 0.0
    total_time = 0.0
    max_speed = 0.0
    for _, d, t, speed, counted in steps:
        if counted:
            distance += d
            total_time += t
            max_speed = max(max_speed, speed)
    return distance, total_time, max_speed


_END_STEP = (0, 0.0, 0.0, 0.0, False)  # discarded step that ends the last run


def _sprints(steps, fps):
    """Number of sprints over a sequence of candidate steps (fps in frames/s).

    steps is a sequence of (gap, d, t, speed, counted) candidate steps. A run
    is a maximal group of neighbouring counted steps with speed at least
    SPRINT_SPEED; a discarded step, a slower counted step or the end of the
    track ends the run. A run counts as one sprint when its total gap divided
    by fps is at least SPRINT_MIN_SECONDS (seconds).
    """
    sprints = 0
    run = 0  # sum of gaps (frames) in the current fast run
    for gap, _, _, speed, counted in [*steps, _END_STEP]:
        if counted and speed >= SPRINT_SPEED:
            run += gap
        else:
            if run / fps >= SPRINT_MIN_SECONDS - 1e-9:
                sprints += 1
            run = 0
    return sprints


def _median(values):
    """Median of a non-empty list of numbers: the middle of the sorted
    values, the mean of the two middle ones when the count is even."""
    order = sorted(values)
    mid = len(order) // 2
    if len(order) % 2:
        return order[mid]
    return (order[mid - 1] + order[mid]) / 2


def _smoothed_appearances(appearances):
    """One person's appearances with the tracker's jitter smoothed out.

    appearances is a per-person list as built by _appearances: (frame, role,
    team, jersey, xy) in frame order, xy None when the position is not
    valid.  Returns a new list; the input is not modified.

    Rule (applied per person, never across ids):

    * The positioned appearances are grouped into segments: consecutive
      appearances of a segment are at most MAX_STEP_GAP frames apart (a
      larger gap starts a new segment, the same gap at which a step would
      be discarded).
    * The first positioned appearance of each segment keeps its recorded
      position; it anchors the segment.
    * Every other appearance of a segment: its displacement from the
      previous positioned appearance (x and y separately) is replaced by the
      median of the displacements of the up to 2 * SMOOTH_HALF + 1
      consecutive appearances centred on it (clipped at the segment ends),
      and its smoothed position is the anchor position plus the accumulated
      median displacements up to that appearance.
    * The last positioned appearance of the track keeps its recorded
      position, and appearances without a valid position are left as they
      are.

    A steady track (constant displacement per appearance) comes out exactly
    as it went in, and a single-frame position jump is removed, because the
    jump's two large displacements are the median outliers of their
    centred windows.
    """
    n = len(appearances)
    out = list(appearances)
    if n < 3:
        return out
    pos = [a[4] for a in appearances]
    last_pos_index = None
    for i in range(n):
        if pos[i] is not None:
            last_pos_index = i
    segments = []
    current = []
    for i in range(n):
        if pos[i] is None:
            continue
        if current and appearances[i][0] - appearances[current[-1]][0] > MAX_STEP_GAP:
            segments.append(current)
            current = []
        current.append(i)
    if current:
        segments.append(current)
    for segment in segments:
        count = len(segment)
        if count < 2:
            continue
        disp = [None] * count
        for j in range(1, count):
            p0, p1 = pos[segment[j - 1]], pos[segment[j]]
            disp[j] = (p1[0] - p0[0], p1[1] - p0[1])
        acc = [pos[segment[0]][0], pos[segment[0]][1]]
        for j in range(1, count):
            if segment[j] == last_pos_index:
                continue  # the track's last appearance is not smoothed
            lo = max(1, j - SMOOTH_HALF)
            hi = min(count, j + SMOOTH_HALF + 1)
            xs = [disp[k][0] for k in range(lo, hi)]
            ys = [disp[k][1] for k in range(lo, hi)]
            acc = [acc[0] + _median(xs), acc[1] + _median(ys)]
            f, role, team, jersey, _ = appearances[segment[j]]
            out[segment[j]] = (f, role, team, jersey, (acc[0], acc[1]))
    return out


def _appearances(seq):
    """Person appearances of a clean sequence, keyed by person id.

    Each value is a list of (frame, role, team, jersey, xy) tuples in frame
    order, one per appearance (first entry per id per frame); xy is None when
    the position is not valid.
    """
    apps = {}
    for frame in seq:
        for person in _frame_people(frame):
            apps.setdefault(person["id"], []).append(
                (frame["frame"], person.get("role"), person.get("team"),
                 person.get("jersey"), _valid_xy(person.get("xy")))
            )
    return apps


def _player_entry(pid, apps, fps):
    """Stats dict for one person id (fps in frames/s), unrounded.

    Keys: id, role, team, jersey, frames (count), seconds (s), distance_m (m),
    avg_speed_ms (m/s), max_speed_ms (m/s), sprints (count).
    """
    points = [(f, pos[0], pos[1]) for f, _, _, _, pos in apps if pos is not None]
    steps = list(_candidate_steps(points, fps, PLAYER_SPEED_CAP))
    distance, total_time, max_speed = _step_totals(steps)
    return {
        "id": pid,
        "role": _most_common(a[1] for a in apps),
        "team": _most_common(a[2] for a in apps),
        "jersey": _most_common(a[3] for a in apps),
        "frames": len(apps),
        "seconds": len(apps) / fps,
        "distance_m": distance,
        "avg_speed_ms": distance / total_time if total_time else 0.0,
        "max_speed_ms": max_speed,
        "sprints": _sprints(steps, fps),
    }


def player_stats(frames, fps=25, smooth=False):
    """Per-person stats over the clean sequence of frames, unrounded.

    frames: list of frame dicts (see load_frames); fps: frames/s (default
    25). Returns one dict per distinct person id (referees too), sorted by
    id ascending: id, role, team, jersey, frames (count), seconds (s),
    distance_m (m), avg_speed_ms (m/s), max_speed_ms (m/s), sprints
    (count).

    smooth: when True, each person's positions are first jitter-smoothed
    (see _smoothed_appearances) and the steps are computed from the
    smoothed positions; when False (the default) the recorded positions
    are used, exactly as before.
    """
    apps = _appearances(_clean_sequence(frames))
    if smooth:
        apps = {pid: _smoothed_appearances(appearances)
                for pid, appearances in apps.items()}
    return [_player_entry(pid, apps[pid], fps) for pid in sorted(apps)]


def team_shape(frames, team):
    """Average shape over the qualifying frames of one team, unrounded.

    frames: list of frame dicts (see load_frames); team: "left" or "right".
    Returns {"frames": count, "avg_centroid": [x, y] in metres or None,
    "avg_width": metres or None, "avg_length": metres or None}; a frame
    qualifies when it has at least 3 positioned players (role "player") of
    the team.
    """
    seq = _clean_sequence(frames)
    qualifying = 0
    cx = cy = width = length = 0.0
    for frame in seq:
        points = []
        for person in _frame_people(frame):
            if person.get("team") != team or person.get("role") != "player":
                continue
            pos = _valid_xy(person.get("xy"))
            if pos is not None:
                points.append(pos)
        if len(points) < 3:
            continue
        qualifying += 1
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx += sum(xs) / len(xs)
        cy += sum(ys) / len(ys)
        width += max(ys) - min(ys)
        length += max(xs) - min(xs)
    if qualifying == 0:
        return {"frames": 0, "avg_centroid": None, "avg_width": None, "avg_length": None}
    return {
        "frames": qualifying,
        "avg_centroid": [cx / qualifying, cy / qualifying],
        "avg_width": width / qualifying,
        "avg_length": length / qualifying,
    }


def _ball_points(seq):
    """(frame, x, y) in metres for each clean frame whose ball has a valid
    position."""
    points = []
    for frame in seq:
        ball = frame.get("ball")
        if not isinstance(ball, dict):
            continue
        pos = _valid_xy(ball.get("xy"))
        if pos is not None:
            points.append((frame["frame"], pos[0], pos[1]))
    return points


def _ball_distance(points, fps):
    """Total ball distance in metres over counted steps (cap BALL_SPEED_CAP).

    points is a list of (frame, x, y) in metres in frame order; fps in
    frames/s.
    """
    distance, _, _ = _step_totals(_candidate_steps(points, fps, BALL_SPEED_CAP))
    return distance


def heatmap(frames, which, nx=21, ny=14):
    """Occupancy grid for one team or the ball.

    frames: list of frame dicts (see load_frames); which: "left", "right" or
    "ball" (anything else raises ValueError); nx, ny: grid size in columns and
    rows (default 21 x 14). Returns ny rows of nx ints: occurrences per cell
    over the clean sequence, positions in metres on the fixed pitch ranges.
    """
    if which not in ("left", "right", "ball"):
        raise ValueError("unknown heatmap target: %r" % (which,))
    grid = [[0] * nx for _ in range(ny)]
    seq = _clean_sequence(frames)
    x_lo, x_hi = PITCH_X_RANGE
    y_lo, y_hi = PITCH_Y_RANGE
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo

    def add(x, y):
        if x < x_lo or x > x_hi or y < y_lo or y > y_hi:
            return
        col = min(int(math.floor((x - x_lo) * nx / x_span)), nx - 1)
        row = min(int(math.floor((y - y_lo) * ny / y_span)), ny - 1)
        grid[row][col] += 1

    if which == "ball":
        for _, x, y in _ball_points(seq):
            add(x, y)
    else:
        for frame in seq:
            for person in _frame_people(frame):
                if person.get("team") != which:
                    continue
                if person.get("role") not in ("player", "goalkeeper"):
                    continue
                pos = _valid_xy(person.get("xy"))
                if pos is not None:
                    add(pos[0], pos[1])
    return grid


def _rounded_player(entry):
    return {
        "id": entry["id"],
        "role": entry["role"],
        "team": entry["team"],
        "jersey": entry["jersey"],
        "frames": entry["frames"],
        "seconds": round(entry["seconds"], 2),
        "distance_m": round(entry["distance_m"], 2),
        "avg_speed_ms": round(entry["avg_speed_ms"], 2),
        "max_speed_ms": round(entry["max_speed_ms"], 2),
        "sprints": entry["sprints"],
    }


def _rounded_shape(shape):
    centroid = shape["avg_centroid"]
    return {
        "frames": shape["frames"],
        "avg_centroid": None if centroid is None else [round(centroid[0], 2), round(centroid[1], 2)],
        "avg_width": None if shape["avg_width"] is None else round(shape["avg_width"], 2),
        "avg_length": None if shape["avg_length"] is None else round(shape["avg_length"], 2),
    }


def summarize(frames, fps=25, smooth=False):
    """Full analytics summary of the clean sequence of frames.

    frames: list of frame dicts (see load_frames); fps: frames/s (default
    25). Returns a JSON-safe dict: fps (frames/s), frame_count, first_frame,
    last_frame, duration_s (s), players (player_stats with seconds in s,
    distance_m in m and speeds in m/s rounded to 2 decimals), teams
    (team_shape in metres rounded to 2 decimals), heatmaps (21 x 14 grids),
    ball (frames_with_ball count, distance_m in metres rounded to 2 decimals).

    smooth: passed on to player_stats (see there); the teams, heatmaps and
    ball sections always use the recorded positions.
    """
    seq = _clean_sequence(frames)
    count = len(seq)
    ball_points = _ball_points(seq)
    return {
        "fps": fps,
        "frame_count": count,
        "first_frame": seq[0]["frame"] if seq else None,
        "last_frame": seq[-1]["frame"] if seq else None,
        "duration_s": round(count / fps, 2),
        "players": [_rounded_player(p)
                    for p in player_stats(frames, fps, smooth=smooth)],
        "teams": {
            "left": _rounded_shape(team_shape(frames, "left")),
            "right": _rounded_shape(team_shape(frames, "right")),
        },
        "heatmaps": {
            "grid": {"nx": 21, "ny": 14,
                     "x_range": [PITCH_X_RANGE[0], PITCH_X_RANGE[1]],
                     "y_range": [PITCH_Y_RANGE[0], PITCH_Y_RANGE[1]]},
            "left": heatmap(frames, "left"),
            "right": heatmap(frames, "right"),
            "ball": heatmap(frames, "ball"),
        },
        "ball": {
            "frames_with_ball": len(ball_points),
            "distance_m": round(_ball_distance(ball_points, fps), 2),
        },
    }
