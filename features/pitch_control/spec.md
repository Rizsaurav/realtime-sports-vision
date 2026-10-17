# Feature: pitch_control — grid-based pitch control

## Purpose

For a stored match, compute which team "controls" each cell of a 21 × 14 grid laid over the
105 m × 68 m pitch: a team controls a cell when the nearest of its positioned players would
reach the cell centre first if both teams' nearest players ran to it at a constant 5 m/s.
The control is evaluated for every recorded frame, aggregated per 1-second bucket
(a territory time series) and averaged over the requested window (an average control map per
team, plus a share per team). The result is served by the match library app as a backend
feature plugged in through the registry described in `tasks/feature_registry.md`.

## Interface

- Module to build: `src/sideline/features/pitch_control.py`, a feature of the
  `src/sideline/features/` package (`tasks/feature_registry.md`). It defines
  `register(router, ctx)` and adds the route below to the `router` that is mounted at
  `/api/matches/{match_id}`. Standard library only (plus FastAPI for the route), Python 3.9
  compatible, no global state, no new dependencies; importing it must not import torch, cv2
  or numpy. The feature never mutates the frame dicts it is handed.
- Route: `GET /api/matches/{match_id}/pitch-control` (declared as `GET /pitch-control`).
- Query parameters (both optional): `start`, `end` — inclusive frame numbers, decimal strings
  matching `^[0-9]+$`. Defaults: `start = meta["first_frame"]`, `end = meta["last_frame"]`.
- Status codes:
  - `200` — the JSON body below.
  - `400` — `{"error": "start must be a non-negative integer"}` when `start` is not a
    non-negative integer, `{"error": "end must be a non-negative integer"}` for `end`, and
    `{"error": "start must not be greater than end"}` when `start > end`.
  - `404` — via `ctx.require(match_id)`, for a malformed id (not `m_[0-9a-f]{10}`) or an
    unknown match, with no file access for malformed ids.
- Body on `200` (every fraction rounded to 3 decimal places, each independently):

```json
{
  "match_id": "m_0123456789",
  "start": 0,
  "end": 24,
  "fps": 25,
  "seconds": 1,
  "speed_ms": 5.0,
  "grid": {"nx": 21, "ny": 14, "x_range": [-52.5, 52.5], "y_range": [-34.0, 34.0]},
  "share": {"left": 0.524, "right": 0.429, "neutral": 0.048},
  "series": [{"second": 0, "left": 0.524, "right": 0.429, "neutral": 0.048}],
  "control": {"left": [[1.0, "21 values per row, 14 rows (index = row r, column c)"]],
              "right": [[0.0, "..."]]}
}
```

- Field meanings:
  - `start`, `end`: the window actually served (ints); `fps` is 25; `speed_ms` is 5.0.
  - `seconds`: `S = ceil((end - start + 1) / 25)` — the number of 1-second buckets.
  - `grid`: the fixed grid description (21 columns × 14 rows over the pitch ranges).
  - `share`: `{"left", "right", "neutral"}` — for each side, the fraction of all
    (cell, second) pairs of the window it owns: `round(total / (294 * S), 3)`.
  - `series`: exactly `S` objects `{"second": i, "left": ..., "right": ..., "neutral": ...}`
    with `i` = 0-based bucket index; `left` = `round(n_left(i) / 294, 3)` where `n_left(i)`
    is the number of cells the team owns in second `i` (out of the 294), etc.
  - `control`: the average control map, one 14 × 21 grid per team (index = row `r`,
    column `c`); `control[T][r][c]` = `round(s_T(r,c) / S, 3)` where `s_T(r,c)` is the
    number of seconds team `T` owned cell `(r,c)`. There is no third (neutral) map.
- Caching: the computation runs through
  `ctx.cache_json(match_id, "pitch-control-{start}-{end}", compute)`, so the result is
  stored in `<match>/features/pitch-control-<start>-<end>.json` and the response is exactly
  that JSON; it is recomputed automatically when `frames.jsonl` becomes newer.
- `GET /api/features` (the registry) must list the feature with `name` `"pitch_control"`
  and `ok` `true` once built.

## Rules

1. **Grid geometry.** The pitch is `x` in [-52.5, 52.5] and `y` in [-34, 34] metres. The
   grid has 21 columns and 14 rows, so every cell is exactly 105/21 = 5 m wide and
   68/14 = 5 m high. Cell (row `r`, column `c`) with `r` in 0..13, `c` in 0..20 has centre
   `(x, y) = (-50 + 5c, -30 + 5r)`. Example: (0,0) → (-50, -30); (8,10) → (0, 10);
   (13,20) → (50, 30); column 11 centres on x = 5, row 8 centres on y = 10.
2. **Counted people.** In each frame, a person counts for team `T` when role is
   `"player"` or `"goalkeeper"`, team is `T` (`"left"` or `"right"`), and `xy` is a 2-element
   list of finite numbers (the platform helper `people_xy(frame, team=T)` gives exactly
   these as `(id, x, y)`). Referees, `team: null` and missing/invalid `xy` never count.
   If the same id appears twice in one frame, the first entry wins and the duplicate is
   ignored. Positions outside the pitch ranges are still used as-is. Example: a frame with
   left id 1 at (0, 0), left id 1 again at (100, 0) and right id 2 at (10, 0) gives left one
   player at (0, 0) and right one player at (10, 0).
3. **Reach time.** A player `d` metres from a cell centre reaches it in `d / 5` seconds
   (constant 5 m/s). A team's reach time to a cell is the minimum over its counted players
   in the frame; a team with no counted player has reach time infinity. Example:
   d = 10 m → 2.0 s; d = 0.5 m → 0.1 s.
4. **Per-frame owner.** The team with the strictly smaller reach time owns the cell in that
   frame; a tie (including infinity = infinity) is neutral. Because both teams move at 5 m/s,
   comparing reach times is exactly comparing the squared distances to each team's nearest
   counted player, so implementations may skip the square roots; ties are unaffected.
   Example: left 8.05 m vs right 8.05 m → neutral; left 3 m vs right 12 m → left.
5. **Second buckets.** The window is split into buckets of 25 frames (1 s at 25 fps) aligned
   to `start`: bucket `k` covers frames `start + 25k` .. `start + 25k + 24`, clipped at
   `end`. `S = ceil((end - start + 1) / 25)`. Example: start=0, end=249 → S = 10;
   start=0, end=23 → S = 1 (a 24-frame partial second); start=5, end=29 → S = 1
   (bucket covers frames 5..29, not 0..4 and 5..29).
6. **Per-second owner.** For one cell in one second containing `B` recorded frames, a team
   owns the cell in that second when it owned it in strictly more than half of those frames
   (count > B/2); otherwise the cell is neutral for that second. `B = 0` (no recorded frame
   in the bucket) → neutral. Example: 13 of 25 frames → owner; 12 of 24 frames → not an
   owner, since 12 is not > 12.
7. **Territory series.** For second `i`, `left = round(n_left/294, 3)`,
   `right = round(n_right/294, 3)`, `neutral = round(n_neutral/294, 3)`, where the `n_*` are
   cell counts out of the 294 cells. Example: 154 left cells → round(154/294, 3) = 0.524.
8. **Average control map.** `control[T][r][c] = round(s_T(r,c) / S, 3)`. Example: a cell
   owned in 1 of 2 seconds → 0.5; in 0 of 2 → 0.0.
9. **Share.** `share[T] = round(Σ_s n_T(s) / (294 * S), 3)` — the fraction of all
   cell-seconds, one exact rational rounded once (do not average the rounded series values).
   Example: 154 left cells in second 0 and 126 in second 1 → 280 of 588 cell-seconds →
   round(280/588, 3) = 0.476.
10. **Worked example (the canonical fixture).** 25 identical frames, one left player at
    (0, 0), one right player at (10, 0). The tie line is x = 5, which is exactly the centre
    of column 11: column 11 is neutral (14 cells), columns 0..10 are left (11 × 14 = 154
    cells), columns 12..20 are right (9 × 14 = 126 cells). With the default window
    (start=0, end=24, S=1): `series = [{"second": 0, "left": 0.524, "right": 0.429,
    "neutral": 0.048}]` (154/294 = 0.523809..., 126/294 = 0.428571..., 14/294 =
    0.047619...), `share` the same, `control["left"]` rows = `[1.0]×11 + [0.0] +
    [0.0]×9`, `control["right"]` rows = `[0.0]×11 + [0.0] + [1.0]×9`.
11. **Window validation and defaults.** `start`/`end` default to the match's first/last
    frame. Non-integer or negative values → 400 (exact messages in the Interface). The
    window may extend beyond the recorded frames: buckets without recorded frames are simply
    all-neutral; there is no 400 for out-of-range windows.
12. **Data access.** Use `ctx.frames(match_id)` (the LRU-cached parsed frames) and keep only
    the frames whose number lies in [start, end]; use `ctx.meta` / `ctx.require` for the
    bounds and the 404. Do not re-read `frames.jsonl` yourself.

## Edge cases

- **Empty input / nobody on pitch:** a frame with no counted people → all 294 cells neutral
  in that frame; a match whose frames never contain counted people → `share` =
  {left 0.0, right 0.0, neutral 1.0}, control maps all 0.0. Referees and `team: null`
  people do not change anything (e.g. a "left" referee at (8, 0) next to a right player at
  (10, 0) leaves column 11 of the canonical fixture neutral).
- **One frame:** a 1-frame match (start=end=first frame) → S = 1, the single bucket has
  B = 1, so a team that owns a cell in that frame owns it for the second (1 > 0.5).
- **Missing ball:** the ball is never used by this feature; `ball` null, present or
  positioned changes nothing (a ball at (0, 0) with no people → all neutral).
- **Missing positions:** a person with `xy` null or invalid in a frame is simply not counted
  for that frame; if that empties a team, the other team wins every cell in that frame, but
  one such frame among 25 cannot flip a second (1 is not more than 12.5, see rule 6).
- **Duplicate ids:** first entry per id per frame wins (rule 2 example).
- **Gaps in frames:** only recorded frames count toward B; a bucket with a gap still uses
  its recorded frames (20 of 25 frames all agreeing → the majority side owns).
- **Window beyond the match:** start=0, end=99 over a 25-frame match → S = 4; seconds 1..3
  are all-neutral and dilute the share (left = round(154/1176, 3) = 0.131 in the canonical
  fixture). A window fully beyond the match (start=1000, end=1024) → S = 1, all neutral,
  share neutral 1.0.
- **Split seconds:** 24 frames with the two teams swapped halfway (12 vs 12) → no team has
  a strict majority anywhere → every cell neutral for the second (share neutral 1.0);
  25 frames 13 vs 12 → the 13-frame side wins the contested cells.
- **Validation:** `start` > `end` → 400; `start=abc`, `end=-1` → 400; unknown or malformed
  match id (e.g. `m_0000000000`, `m_123`) → 404.
- **Huge input and performance budget:** a 25,000-frame match (S = 1000, 22 counted people
  per frame) must be served by a single GET (default window) in under 30 s of wall clock.
  Keep the inner loops lean: one squared-distance minimum per cell per team, no per-cell
  function calls, no per-frame allocation of the 294-entry fields per player.
- **No mutation / idempotence:** the feature must not modify stored frames or its inputs;
  repeating the same request returns exactly the same JSON (served from the cache file).

## Non-goals

- No ball-based or three-way control: "neutral" means neither team, nothing else.
- No per-player attribution, no per-cell reach-time output, no sub-second resolution.
- No interpolation or smoothing between frames; each recorded frame is used as-is.
- No app-level routes (`register_app`), no changes to shared modules, no state other than
  the `features/pitch-control-*.json` cache files.
- No other pitch size or grid; no persistence of per-second maps beyond the cached window
  result.
