# Feature spec: team_shape — per-second team shape time series

## Purpose

`analytics.summarize` collapses a whole match into one static average per team
(`avg_centroid`, `avg_width`, `avg_length`), which hides when a team pressed or
sat deep. This feature exposes the shape of each team **per second** so a UI can
animate and inspect how the block stretched, compressed and re-positioned during
the match: centroid, width, length, convex hull area, defensive line height and
compactness per second, plus a match-wide average per team. It plugs into the
library app as a backend feature following `tasks/feature_registry.md`: one new
module, no edits to shared files.

## Interface

Module to build: `src/sideline/features/team_shape.py` (Python 3.9, standard
library only — no torch, cv2 or numpy; it uses the
`sideline.features._common` helpers plus FastAPI's `HTTPException`).

### HTTP route

`GET /api/matches/{match_id}/team-shape` — declared inside `register(router,
ctx)` as `@router.get("/team-shape")` (the registry mounts the router under
`/api/matches/{match_id}`).

Optional query parameter `team`, exactly `"left"` or `"right"`.

| Case | Status | Body |
|---|---|---|
| success (no `team`) | 200 | the document below, both teams |
| success `?team=left` / `?team=right` | 200 | same document, `teams` holds only that one team |
| `?team=<anything else>`, including `""` and wrong case | 400 | `{"detail": {"error": "team must be 'left' or 'right'"}}` |
| unknown or malformed `match_id` | 404 | `{"detail": {"error": "match not found"}}` (raised by `ctx.require`; malformed ids such as `m_short` or `nope` never touch the filesystem) |

The match is resolved first: a bad `team` on an unknown match answers 404, not 400.

Document (all floats rounded to 2 decimals, `t` an int):

```json
{
  "match_id": "m_ab12cd34ef",
  "fps": 25,
  "first_frame": 0,
  "last_frame": 129,
  "teams": {
    "left": {
      "seconds": [
        {"t": 0, "centroid": [5.0, 0.0], "width": 10.0, "length": 10.0,
         "hull_area": 100.0, "defensive_line": 57.5, "compactness": 7.07}
      ],
      "average": {"centroid": [5.0, 0.0], "width": 10.0, "length": 10.0,
                  "hull_area": 100.0, "defensive_line": 57.5, "compactness": 7.07}
    },
    "right": {"seconds": [], "average": null}
  }
}
```

* `seconds`: only the *qualifying* seconds of the team, ascending `t`. A team
  with no qualifying second answers `"seconds": []` and `"average": null`.
* The **full** document (both teams, including `match_id`) is what the route
  caches via `ctx.cache_json(match_id, "team-shape", ...)` — stored at
  `<match_dir>/features/team-shape.json`, served only while it is newer than
  `frames.jsonl`, written atomically (temp file + rename). The `team` filter is
  applied *after* the cache read, so the cache never stores a filtered copy.

### Python interface (public names of the module)

```python
FPS = 25
MIN_PLAYERS = 4
OWN_GOAL_X = {"left": -52.5, "right": 52.5}

def team_second(positions, team):
    """positions: sequence of (x, y) tuples in metres, one per distinct player
    (already averaged per second by the caller); team: "left" | "right"
    (anything else raises ValueError). Returns the 6-metric dict below, or None
    when len(positions) < MIN_PLAYERS. Unrounded floats."""

def shape_series(frames, team, fps=25):
    """frames: list of frame dicts (docs/WEBAPP_CONTRACT.md). Returns
    [{"t": s, "centroid": ..., "width": ..., "length": ..., "hull_area": ...,
      "defensive_line": ..., "compactness": ...}, ...] for the qualifying
    seconds only, ascending t, unrounded."""

def shape_match(frames, fps=25):
    """Returns {"fps": fps, "first_frame": int|None, "last_frame": int|None,
    "teams": {"left": {"seconds": [...], "average": {...}|None},
              "right": {...}}}, unrounded. Empty input: first_frame/last_frame
    None, both teams {"seconds": [], "average": None}."""

def register(router, ctx):
    """Defines GET /team-shape on the given APIRouter."""
```

`team_second`, `shape_series` and `shape_match` never mutate their inputs.

## Rules

1. **Cleaning.** Frames are deduplicated by `frame` number (the last occurrence
   wins) and sorted ascending before use; the upload already rejects
   non-increasing frames, so this guards direct callers only. `ball`, `colors`,
   `box`, `delay` and `jersey` are ignored entirely.
   Example: two lines both with `"frame": 5` — only the second line's people
   are used.
2. **Second binning.** With `first` the cleaned first frame number, the second
   index is `t = (frame - first) // 25` (25 fps).
   Example: `first = 7`: frame 7 → `t = 0`, frame 31 → `t = 0`, frame 32 →
   `t = 1`. With `first = 0`, frame 250 → `t = 10`.
3. **Counted people.** A person counts for team `T` in a frame when
   `team == T`, `role == "player"` (goalkeepers, referees and `team: null`
   people never count) and `xy` is a valid position: a 2-element list/tuple of
   finite numbers. `xy: null`, one-element lists and non-finite values are
   invalid. If one `id` appears twice in a frame, only the first entry counts.
   Example: a frame holding a left goalkeeper at (-50, 0) and a referee at
   (40, 20) contributes 0 points to either team.
4. **Per-second player position.** For one second and one team, each distinct
   counted `id` gets the arithmetic mean of its x observations and of its y
   observations in that second.
   Example: a player at x = 0 in frames 0–19 (20 frames) and x = 10 in frames
   20–24 (5 frames) has second-0 position x = (20·0 + 5·10) / 25 = 2.0.
5. **Qualification.** A (team, second) pair is in the series only when it has
   at least `MIN_PLAYERS` = 4 distinct counted players.
   Example: 3 players in a second → no entry; 4 → an entry.
6. **centroid** = [mean x, mean y] of the n per-player positions.
   Example: the square (0,−5), (0,5), (10,−5), (10,5) → [5.0, 0.0].
7. **width** = max y − min y. Example: that square → 10.0.
8. **length** = max x − min x. Example: that square → 10.0.
9. **hull_area** = area in m² of the convex hull of the n points (monotone
   chain or any equivalent hull; duplicate and collinear points removed from
   the hull). A degenerate hull (fewer than 3 non-collinear points) has area
   0.0; interior points change nothing.
   Examples: the 10×10 square → 100.0. The diamond (20,0), (30,0), (25,8),
   (25,−8) has diagonals 10 and 16 → 10·16/2 = 80.0. Four collinear points
   (0,0), (1,0), (2,0), (3,0) → 0.0. The square plus the interior point (5,5)
   → still 100.0.
10. **defensive_line** = the distance in metres from the team's own goal line
    to the mean x of its **4 deepest outfield players** — the 4 counted
    players nearest the own goal: the 4 smallest x for `left`, the 4 largest x
    for `right` (ties keep input order, which is the order the ids were first
    seen in that second).
    `defensive_line = |x̄_d4 − OWN_GOAL_X[team]|` with `OWN_GOAL_X` = left −52.5,
    right +52.5, so 0.0 means "on the goal line" and 52.5 means "at the
    halfway line".
    Examples: the left square has x̄ = 5 → |5 − (−52.5)| = 57.5. Five left
    players at x = 0, 10, 12, 14, 16 → the forward at 16 is excluded, x̄ =
    (0+10+12+14)/4 = 9 → 61.5. Five right players at x = 10, 12, 14, 16, 40 →
    the 4 deepest are 40, 16, 14, 12, x̄ = 20.5 → |20.5 − 52.5| = 32.0.
11. **compactness** = mean over the n players of the Euclidean distance from
    the player's position to the centroid, in metres.
    Examples: the square — every corner sits √(5²+5²) = 5√2 ≈ 7.0711 from the
    centroid → 7.0711 (7.07 rounded). The diamond — distances 5, 5, 8, 8 →
    (5+5+8+8)/4 = 6.5.
12. **Match average.** For each metric, the plain mean of the *unrounded*
    per-second values over the team's qualifying seconds, then rounded for
    JSON; the centroid's average is the mean of the centroid x's and the mean
    of the centroid y's. `average` is null when the team has no qualifying
    second.
    Example: a team with centroid [5, 0] (defensive_line 57.5) in second 0 and
    [15, 0] (defensive_line 67.5) in second 1 answers average centroid
    [10.0, 0.0] and average defensive_line (57.5+67.5)/2 = 62.5.
13. **Rounding.** Every float in the HTTP document is rounded to 2 decimals
    (as `round(x, 2)` does). The pure functions return unrounded values.
    Examples: 7.0710678… → 7.07; 5.2179163… → 5.22; 12.30981… → 12.31.
14. **Cache.** The route builds the full document with
    `ctx.cache_json(match_id, "team-shape", compute)`; a cached document is
    served only while it is newer than `frames.jsonl`, otherwise it is
    recomputed and written atomically. `?team=` never changes what is cached.
15. **Performance budget.** One `GET /api/matches/{id}/team-shape` over the
    25,000-frame recording of the performance test (22 people per frame — 10
    outfielders and 1 goalkeeper per team; 59,693,095 bytes of JSONL; 1,000
    seconds) completes in **less than 30 s**. Machine assumption: one core,
    Python 3.9, laptop CPU. On such a machine the observed cost of the
    measured path (read the 25,000 lines, compute both teams, build the
    ~239,593-char document) is about 3 s, so the budget is 10× the measured
    cost.

## Edge cases

* **Empty input.** `shape_match([])` → `first_frame`/`last_frame` null, both
  teams `"seconds": []`, `"average": null`; `shape_series([], "left")` → `[]`.
  The route can never see an empty match (the store rejects 0-frame uploads).
* **One frame.** A single frame at `frame: 7` with 4 left and 4 right players
  → `first_frame = last_frame = 7`, one second at `t = 0` per team.
* **Missing ball.** The ball is ignored: a recording with `ball: {"xy":
  [0, 0]}` in every frame answers the identical document as the same recording
  with `ball: null` everywhere.
* **Missing positions.** People with invalid `xy` are dropped; a frame whose 4
  players include 2 with `xy: null` has only 2 counted players → the second is
  omitted (below MIN_PLAYERS = 4).
* **Duplicate ids.** Two entries with the same `id` in one frame → the first
  entry's position is used, the second is ignored.
* **Duplicate frame numbers** (direct calls only): the last occurrence wins.
* **Gaps in frames.** Frames 0–24 and 50–74 (25–49 missing) → seconds
  `t = 0` and `t = 2` only; `t = 1` is absent with no placeholder, and
  `last_frame = 74`.
* **Below-minimum team.** A team with 3 counted players in a second contributes
  no entry for it; a team with no qualifying second has `"seconds": []` and
  `"average": null` in the route answer.
* **Huge input.** The 25,000-frame / 22-people recording (59,693,095 bytes) is
  processed within the 30 s budget, and the answer carries 1,000 `t` entries
  (0–999) per team.

## Non-goals

* No ball-based or possession-based shape metrics (those belong to the heatmap
  and possession features).
* No per-player time series, no goalkeeper metrics, no frame-level (sub-second)
  resolution.
* No cross-match or cross-team comparison endpoint (`/api/compare` is another
  feature's route).
* No rendering: this is a pure backend route; drawing the shape is the web
  layer's job.
* No query parameters besides `team`, no pagination or windowing of the series.
