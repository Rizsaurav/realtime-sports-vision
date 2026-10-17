# Feature: formation

## Purpose

Report, per team and per 60-second window, the shape the team was playing
(e.g. "4-4-2"). The shape is derived from outfield player positions only:
each player's window-average position is projected onto the team's
attacking axis (the line from its own goal to the opponent's goal), the
players are split into lines by gaps along that axis, and the line sizes,
read from the back line forward, form the formation string. The route also
returns a confidence in [0, 1] and the average position of every role slot
(line + seat within the line), so a client can draw the shape on a pitch.

## Interface

Module to build: `src/sideline/features/formation.py`, a member of the
`sideline.features` package (contract: `tasks/feature_registry.md`).
Standard library plus `sideline.features._common` only; no other imports,
Python 3.9 compatible, no global state, inputs never mutated.

- `build_formation(frames, fps=25, window_s=60.0) -> dict`
  Pure function over the parsed frame list (format: `docs/WEBAPP_CONTRACT.md`,
  frames in increasing `frame` order). Returns exactly
  `{"window_s": <float>, "windows": [<window>, ...]}` where every float is
  rounded to 2 decimals. No I/O, no mutation of `frames`.

- `register(router, ctx) -> None`
  Declares one route on the router mounted under `/api/matches/{match_id}`:
  `GET /formation`, i.e. `GET /api/matches/{match_id}/formation`.
  Handler: `def formation(match_id: str, window_s: Optional[str] = None)`.
  It must first call `ctx.require(match_id)` (404 with an `error` body for
  unknown or malformed ids - the 404 is answered before any query
  validation), then validate `window_s`, compute the payload with
  `build_formation(ctx.frames(match_id), fps=_common.FPS, window_s=w)` and
  serve it through `ctx.cache_json(match_id, "formation-w" + str(w), compute)`.

- `GET /api/matches/{match_id}/formation?window_s=<number>`
  - 200: `{"match_id": <str>, "window_s": <float>, "windows": [...]}` -
    the cached payload with `match_id` added. A window object has exactly
    the keys `index` (int, 0-based), `start_frame` (int), `end_frame` (int),
    `duration_s` (float), `left`, `right`. Each of `left`/`right` has
    exactly the keys `formation` (string or null), `confidence` (float),
    `players` (int), `slots` (list). Each slot has exactly the keys
    `line` (int), `seat` (int), `x` (float), `y` (float), `n` (int).
    Slots are ordered by line ascending, then seat ascending; windows by
    index ascending.
  - 400: `window_s` is present but not a finite number in [5, 600].
    Body exactly `{"error": "window_s must be a number between 5 and 600"}`.
  - 404: unknown or malformed match id (raised by `ctx.require`).

`window_s` parses with `float(...)`; the default is 60.0. Example:
`?window_s=30` parses to 30.0 and is cached as
`<match>/features/formation-w30.0.json` (the default 60.0 uses
`formation-w60.0.json`), so different `window_s` values never share a
cache file; a cached payload is served only while it is newer than
`frames.jsonl`.

## Rules

Constants: `fps = 25`, pitch half-length 52.5 m, line gap threshold
`GAP = 10.0` m, minimum outfield players per frame 3.

1. **Window length.** `W = int(round(window_s * fps))` frames (Python
   `round`, halves to even). Example: window_s 60.0 -> 1500; 30 -> 750;
   12.5 -> int(round(312.5)) = 312.
2. **Windowing.** For k = 0, 1, 2, ... the window covers
   `[first + k*W, min(first + (k+1)*W - 1, last)]` where first/last are
   the first/last recorded frame numbers; stop when the start exceeds
   last; a window is emitted only if at least one recorded frame falls in
   its range. `duration_s = round((end_frame - start_frame + 1) / fps, 2)`.
   Example: first 0, last 1499, W 1500 -> one window [0, 1499],
   duration 60.0. 25,000 frames (0..24999) -> 17 windows, the last
   [24000, 24999] with duration round(1000/25, 2) = 40.0.
3. **Outfield appearances.** A person entry is an appearance for team T in
   one frame iff `role == "player"`, `team == T`, and `xy` is a length-2
   list of finite numbers. If one id appears several times in the same
   frame, only the first occurrence counts. Goalkeepers, referees (even
   ones mislabelled with a team) and null `xy` never count. Example: a
   frame holding id 9 at [-10.0, -6.0] and again at [99.0, 99.0] yields
   exactly one appearance, at [-10.0, -6.0].
4. **Player window average.** Per id, over its appearances in the window:
   `x = sum_x / n`, `y = sum_y / n`, `n` = number of appearances.
   Example: id 1 at x = -50.0 in 1200 of 1500 frames -> x = -50.0,
   n = 1200.
5. **Attacking coordinate.** Distance from the team's own goal:
   `d = x + 52.5` for the left team (own goal at x = -52.5) and
   `d = 52.5 - x` for the right team (own goal at x = +52.5). Example:
   a left defender at x = -50 -> d = 2.5; a right defender at x = 50 ->
   d = 2.5; the right team's striker at x = 10 -> d = 42.5.
6. **Lines.** Sort the window-average players by `(d, id)` ascending and
   start a new line whenever `d_i - d_prev >= GAP` (10.0). Example:
   d values 2.5 (x4), 22.5 (x4), 42.5 (x2): gaps 20 and 20, both >= 10,
   so three lines of sizes 4, 4, 2. A gap of exactly 10.0 starts a new
   line (d = 10.0 then 20.0 -> two lines).
7. **Formation string.** Line sizes joined with "-", in ascending d
   (the back line, smallest d, first - for both teams d is the distance
   from the team's own goal); null when fewer than 3 players have a
   window average. Example: sizes 4, 4, 2 -> "4-4-2"; the right team at
   d = 2.5 (4 players), 22.5 (3), 42.5 (3) -> "4-3-3".
8. **Seats.** Within a line, sort by `(-y, id)` for the left team and
   `(y, id)` for the right team (seat 0 is the team's own left side when
   facing the opponent's goal); seat = 0-based position. Example: a left
   line with y = 15, 5, -5, -15 gets seats 0..3 in that order; a right
   line with y = -15, -5, 5, 15 gets seats 0..3 in that order.
9. **Slots.** Every seat is one slot; its `x`, `y` are the occupant's
   window averages rounded to 2 decimals and `n` the occupant's
   appearance count. Example: slot (line 0, seat 0) of the left team in
   the worked recording below is x = -50.0, y = 15.0, n = 1500.
10. **Coverage.** `coverage = ok_frames / total_frames` where total is
    the number of recorded frames in the window and ok the frames with at
    least 3 outfield appearances of the team (0 if total is 0).
    Example: present in 1200 of 1500 recorded frames -> 0.8.
11. **Tightness.** Per line, `spread = max(d) - min(d)` over its players'
    window averages; factor `max(0.0, 1.0 - spread / GAP)`;
    `tightness` is the product over lines (a lone player has spread 0,
    factor 1.0). Example: a line spanning d 30.5..32.0 ->
    1 - 1.5/10 = 0.85; a line spanning 5.0 m -> 0.5.
12. **Confidence.** `confidence = round(coverage * tightness, 2)`.
    Examples: 1.0 x 1.0 -> 1.0; 0.8 x 1.0 -> 0.8; 1.0 x 0.85 -> 0.85;
    1.0 x 0.5 -> 0.5. A team with fewer than 3 window-average players
    reports `formation: null`, `confidence: 0.0`, `players: <0..2>`,
    `slots: []`.
13. **players** is the number of ids with at least one appearance in the
    window (10 for a full classic side).
14. **Worked recording.** 1500 frames (0..1499), static positions,
    25 fps. Left team (ids 1..10, back-to-front): 4 at x = -50 with
    y = 15, 5, -5, -15; 4 at x = -30 with y = 15, 5, -5, -15; 2 at
    x = -10 with y = 6, -6. Right team (ids 21..30): 4 at x = 50 with
    y = -15, -5, 5, 15; 3 at x = 30 with y = -15, 0, 15; 3 at x = 10
    with y = -10, 0, 10. One goalkeeper per team plus a referee, all
    excluded. Result, window [0, 1499], duration 60.0: left
    formation "4-4-2", confidence 1.0 (coverage 1500/1500, all spreads
    0), players 10, slot (0, 0) = (-50.0, 15.0, 1500), slot (2, 0) =
    (-10.0, 6.0, 1500); right formation "4-3-3" (d 2.5, 22.5, 42.5),
    confidence 1.0, slot (0, 0) = (50.0, -15.0, 1500) (the right team's
    left side is y = -15), slot (2, 2) = (10.0, 10.0, 1500). Rounding
    example: a striker at x = -10.006, y = -6.126 reports x = -10.01,
    y = -6.13.

## Edge cases

- Empty input (pure function): `build_formation([])` returns
  `{"window_s": 60.0, "windows": []}`.
- One frame: a recording of a single frame 100 yields one window
  [100, 100] with duration round(1/25, 2) = 0.04 and slot n = 1.
- Missing ball: `ball` null or `ball.xy` null in every frame changes
  nothing (the ball is not used at all).
- Missing positions: a player with `xy` null in every frame of the window
  is absent; with one of two strikers missing, "4-4-2" becomes "4-4-1"
  with players 9.
- Duplicate ids: first occurrence in a frame wins (see rule 3), so a
  duplicated id never creates an 11th slot.
- Gaps in frames: coverage counts recorded frames only, and a window
  range containing no recorded frame is omitted. Example: frames
  0..999 and 2000..2499 give windows [0, 1499] (n = 1000 per slot) and
  [1500, 2499] (n = 500, duration 40.0), both confidence 1.0.
- Huge input / performance budget: a 25,000-frame recording with 22
  positioned people per frame (about 63 MB) must answer a cold
  `GET /api/matches/{id}/formation` in under 10 s wall clock; the
  algorithm is a single linear pass per window and team, O(people
  appearances).
- Input validation: `window_s` of "abc", "0", "4.9", "601", "-3" or
  "inf" -> 400; a malformed match id -> 404 before the 400 check.

## Non-goals

- No goalkeeper in the formation, no per-frame (sub-window) shapes, no
  tracking of shape changes inside a window.
- No use of the ball position, no pitch geometry re-estimation, no
  smoothing of tracker jitter (recorded `xy` is used as-is).
- No named positions (no "LB", "CM", ...): a slot is identified only by
  line and seat.
- No cross-match comparison and no UI in this feature.
