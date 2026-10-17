# Feature: intensity — distance in speed zones per player and per team

## Purpose

For a stored match, compute how far each player — and each team — moved in each of
five speed zones: walk (under 2 m/s), jog (2–4), run (4–6), high-speed (6–7) and
sprint (7 and above), both as totals over the whole match and as per-minute time
series, together with a work rate in m/min for every player and team and two
top-10 lists. The steps are taken from the recorded positions with the same step
rules as `analytics` (a step over 12 m/s is discarded, as is a step spanning more
than 5 frames). The result is served by the match library app as a backend feature
plugged in through the registry described in `tasks/feature_registry.md`.

## Interface

**Cross-task dependency (hard precondition).** This feature assumes the feature
registry from `tasks/feature_registry.md` is already built: the
`src/sideline/features/` package (`__init__.py`, `_common.py` with `FeatureContext`
and its helpers), the `register_all` hook in `create_library_app`, and
`GET /api/features`. This spec neither builds nor modifies any of that; without it
the feature cannot be loaded and cannot be tested. Building the registry is a
separate task.

- Module to build: `src/sideline/features/intensity.py`, a feature of the
  `src/sideline/features/` package (`tasks/feature_registry.md`). It defines
  `register(router, ctx)` and adds the route below to the `router` that is mounted
  at `/api/matches/{match_id}`. Standard library only (plus FastAPI for the
  route), Python 3.9 compatible, no global state, no new dependencies; importing it
  must not import torch, cv2 or numpy. The feature never mutates the frame dicts
  it is handed.
- Route: `GET /api/matches/{match_id}/intensity` (declared as `GET /intensity`).
  It takes **no query parameters**: the whole recorded match is always processed;
  extra query parameters are accepted and ignored.
- Status codes:
  - `200` — the JSON body below.
  - `404` — via `ctx.require(match_id)`, for a malformed id (not
    `^m_[0-9a-f]{10}$`) or an unknown match, with no file access for malformed ids.
  - The feature never answers `400`: nothing in the request can be invalid.
- Body on `200` (every distance in metres and every work rate in m/min, rounded
  to 2 decimal places; the canonical worked example at the end of the Rules
  section fills in real numbers):

```json
{
  "match_id": "m_0123456789",
  "fps": 25,
  "first_frame": 0,
  "last_frame": 1499,
  "duration_s": 60.0,
  "minutes": 1,
  "zones": ["walk", "jog", "run", "high", "sprint"],
  "players": [
    {"id": 1, "team": "left", "jersey": "1",
     "distance_m": 59.8,
     "zones_m": {"walk": 59.8, "jog": 0.0, "run": 0.0, "high": 0.0, "sprint": 0.0},
     "per_minute": [[59.8, 0.0, 0.0, 0.0, 0.0]],
     "work_rate_m_per_min": 59.8}
  ],
  "teams": {
    "left": {"distance_m": 239.2,
             "zones_m": {"walk": 59.8, "jog": 179.4, "run": 0.0, "high": 0.0, "sprint": 0.0},
             "per_minute": [[59.8, 179.4, 0.0, 0.0, 0.0]],
             "work_rate_m_per_min": 239.2},
    "right": {"distance_m": 1136.2,
              "zones_m": {"walk": 0.0, "jog": 0.0, "run": 299.0, "high": 388.7, "sprint": 448.5},
              "per_minute": [[0.0, 0.0, 299.0, 388.7, 448.5]],
              "work_rate_m_per_min": 1136.2}
  },
  "top_distance": [
    {"id": 4, "team": "right", "jersey": "4", "distance_m": 448.5}
  ],
  "top_sprint": [
    {"id": 4, "team": "right", "jersey": "4", "sprint_m": 448.5}
  ]
}
```

- Field meanings:
  - `fps` is 25. `first_frame` / `last_frame` / `duration_s` describe the match
    span: `duration_s = (last_frame - first_frame + 1) / fps`, rounded to 2
    decimals.
  - `minutes` = ceil((last_frame - first_frame + 1) / 1500), always at least 1
    (1500 frames = 1 minute at 25 fps).
  - `zones` names the five zones in the row order used by every `per_minute`
    value.
  - `players`: one entry per counted person (Rules 1–2), sorted by id ascending.
    `distance_m` is the total over all zones; `zones_m` is a dict with exactly
    the five zone keys; `per_minute` is a list of exactly `minutes` rows, row
    `m` holding the five metres moved in minute `m`, one per zone in `zones`
    order; `work_rate_m_per_min` is the work rate over the whole match (m/min).
  - `teams`: exactly the keys `"left"` and `"right"`, always present, each with
    the same shape as a player entry without `id` and `jersey`.
  - `top_distance`: at most 10 players by unrounded total distance, descending,
    ties by id ascending; `top_sprint`: at most 10 players by unrounded
    sprint-zone distance, descending, ties by id ascending. Fewer than 10
    counted players means fewer entries (zero-distance players are included).
- Caching: the computation runs through
  `ctx.cache_json(match_id, "intensity", compute)`, so the result is stored in
  `<match>/features/intensity.json` and the response is exactly that JSON; it is
  recomputed automatically when `frames.jsonl` becomes newer than the cache file.
- `GET /api/features` (the registry) must list the feature with name `"intensity"`
  and `ok` `true` once built.

## Rules

1. **Counted people.** A person counts when the most common non-null `role` over
   all their appearances is `"player"` or `"goalkeeper"`, and the most common
   non-null `team` is `"left"` or `"right"` (attribute resolution: most common
   non-null value, ties to the lexicographically smallest — the same rule the
   `players` list of `analytics.summarize` uses). Referees and unassigned people
   are excluded from everything. The person's `jersey` is the most common
   non-null jersey and may stay null. Example: a person seen as left in 3 of 5
   appearances is left; a person whose role is "referee" in every appearance is
   excluded even when their team is "left".
2. **Appearances and steps.** In each frame the first entry per id wins and later
   duplicates are ignored. `xy` is a valid position only when it is a 2-element
   list of finite numbers (booleans are not numbers). A step connects two
   consecutive *positioned* appearances of the same person: `gap` is the frame
   difference, `d` the Euclidean distance in metres, `t = gap / fps`,
   `v = d / t` with fps = 25 and ordinary float arithmetic — the step formula of
   `analytics`. A step is counted when `gap <= 5` and `v <= 12` (the same step
   rules as `analytics`: steps over 12 m/s are discarded, as are steps spanning
   more than 5 frames); a discarded step contributes nothing. Example: 1.0 m in
   5 frames → t = 0.2, v = 5.0, counted; 0.5 m in 6 frames → v = 2.0833 but
   gap 6 > 5, discarded; 2.5 m in 5 frames → v = 12.5 > 12, discarded.
3. **Speed zones.** A counted step's zone is decided by its `v`: walk
   `v < 2`, jog `2 <= v < 4`, run `4 <= v < 6`, high `6 <= v < 7`, sprint
   `v >= 7` (the sprint side is inclusive, like the `>= 7.0` sprint threshold of
   `analytics`). The whole distance `d` of the step goes to that one zone.
   Boundary cases under the Rule 2 formula (gap 5, so t = 0.2): d = 0.4 →
   v = 2.0 → jog; d = 0.8 → v = 4.0 → run; d = 1.2 → v = 5.999999999999999 →
   run; d = 1.21 → v = 6.05 → high; d = 1.4 → v = 6.999999999999999 → high;
   d = 1.41 → v = 7.049999999999999 → sprint; d = 2.4 → v = 11.999999999999998
   → counted (v <= 12), sprint; d = 2.5 → v = 12.5 → discarded.
4. **Minute buckets.** Minute `m` covers frames
   `[first_frame + 1500m, first_frame + 1500(m+1))`. A step is attributed to the
   minute of its **start** frame: `m = (f0 - first_frame) // 1500`; a step that
   crosses a boundary stays in the earlier minute. Example: with first_frame 0 a
   step starting at frame 1495 (ending at 1500) is minute 0, a step starting at
   frame 1500 is minute 1.
5. **Totals and per-minute rows.** A person's zone distance is the unrounded sum
   of the `d` of its counted steps in that zone; `distance_m` is the unrounded
   sum over all five zones. `per_minute` has exactly `minutes` rows (one per
   minute 0..minutes-1), each row the five per-minute zone sums, unrounded, zero
   where the person moved nothing in that minute. Example: 599 counted steps of
   1.0 m, 300 of them starting in minute 0 and 299 in minute 1 → per-minute run
   values 300.0 and 299.0, total 599.0.
6. **Work rate.** `work_rate_m_per_min = distance / (duration_s / 60)` with the
   unrounded total distance and `duration_s = (last_frame - first_frame + 1) /
   fps` — metres per minute of match real time, per player and per team.
   Example: 599.0 m over a 120.0 s match → 599.0 / 2.0 = 299.5 m/min; 0.8 m
   over a 0.24 s match → 0.8 / 0.004 = 200.0 m/min.
7. **Teams.** A team's zone distances, per-minute rows, total and work rate are
   computed over the exact (unrounded) sum of the steps of its players and
   rounded once — never over the already-rounded player values. Example: two
   left-side players each moving 0.004 m (each displays as 0.0 m) → the team's
   distance is 0.01 m, not 0.0 m.
8. **Rounding.** Every distance and work rate in the body is its unrounded value
   rounded to 2 decimal places — each rounded once, at the end, independently.
   The same unrounded totals feed `distance_m`, `zones_m`, `per_minute`,
   `work_rate_m_per_min` and both top lists (the lists sort on the unrounded
   values). Example: an exact total of 0.008 m displays as `distance_m` 0.01
   while its two per-minute rows (0.004 each) display 0.0; summing the displayed
   rows would give the wrong total.
9. **Top lists.** `top_distance` lists all counted players sorted by unrounded
   total distance, descending, ties by id ascending, truncated to 10 entries;
   `top_sprint` by unrounded sprint-zone distance, descending, ties by id
   ascending, truncated to 10. Example: totals 448.5, 388.7, 299.0, 179.4, 59.8
   and five players at 0.0 → `top_distance` starts [448.5, 388.7, 299.0, 179.4,
   59.8]; `top_sprint` puts the 448.5 m sprinter first and lists the 0.0 m
   players after, in id order.
10. **Ball.** The ball is never used by this feature; a positioned ball, a null
    ball or a missing `ball` key all give the same result.
11. **Data access.** Frames come from `ctx.frames(match_id)` (the LRU-cached
    parsed frames); the 404 and the span come from `ctx.require` / `ctx.meta`.
    The feature never reads `frames.jsonl` itself and never calls `analytics`.
    The route takes no query parameters.
12. **Canonical worked example.** 1500 frames (0..1499, 60.0 s, 1 minute). Every
    5 frames (0, 5, ..., 1495) these people appear, moving in +x with a constant
    5-frame displacement: left player 1 (0.2 m per step, v = 1.0), left player 2
    (0.6 m, v = 3.0), right player 3 (1.0 m, v = 5.0), right goalkeeper 4
    (1.5 m, v = 7.5), right player 5 (1.3 m, v = 6.5), plus a left-side referee 9
    (2.0 m per step, v = 10.0) and an unassigned player 10 (1.0 m per step).
    Each track yields 299 counted steps:
    - id 1: walk 59.8 m, work rate 59.8; id 2: jog 179.4; id 3: run 299.0;
      id 4: sprint 448.5; id 5: high 388.7 — in metres; the work rate equals the
      distance because the duration is exactly 1.0 minute.
    - left: 239.2 m (walk 59.8, jog 179.4), work rate 239.2; right: 1136.2 m
      (run 299.0, high 388.7, sprint 448.5), work rate 1136.2.
    - `top_distance`: [4: 448.5, 5: 388.7, 3: 299.0, 2: 179.4, 1: 59.8];
      `top_sprint`: [4: 448.5, then 1, 2, 3, 5 at 0.0 in id order].
    - the `players` list contains exactly the ids 1..5 (9 and 10 are excluded by
      Rule 1); every `per_minute` is a single row equal to the total.

## Edge cases

- **Empty input / nobody on pitch:** a match whose frames contain no counted
  people (empty `people` lists) → `players` is `[]`, both teams all zero, work
  rates 0.0, top lists empty; `minutes` and `duration_s` still come from the
  frame span (25 empty frames → duration_s 1.0, minutes 1).
- **One frame:** a 1-frame match → duration_s 0.04, minutes 1, every distance
  and work rate 0.0, one zero entry per counted person.
- **Missing ball:** never used (Rule 10); a ball positioned in every frame
  changes nothing.
- **Missing positions:** an appearance with `xy` null (or not a 2-element list of
  finite numbers) is simply not a positioned appearance; the step spans over it
  and the frame gap of that step decides (Rule 2). A person whose appearances
  are all unpositioned still gets a zero entry (a null jersey stays null).
- **Duplicate ids:** first entry per id per frame wins (Rule 2): a second entry
  with the same id at a far position is ignored.
- **Gaps in frames:** only recorded frames exist; the duration and the minute
  count use the span from the first to the last frame, not the frame count; a
  person-level absence of more than 5 frames discards the step spanning it.
- **Off-pitch positions:** positions outside the pitch ranges (e.g. x = 60)
  count as-is.
- **Ties and truncation:** equal unrounded distances are ordered by id
  ascending; with more than 10 counted players the top lists carry exactly 10
  entries.
- **Huge input and performance budget:** a 25,000-frame match (60,708,202 bytes
  as recorded, 22 counted people per frame, 17 minutes) must be served by a
  single GET in under 15 s of wall clock. Keep the inner loop lean: one
  distance, one speed and a few comparisons per step, no per-frame allocations
  beyond small lists.
- **No mutation / idempotence:** the feature never modifies the frame dicts or
  any file it does not own; repeating the same request returns exactly the same
  JSON (served from the cache file).

## Non-goals

- No smoothing of positions (unlike the `analytics` endpoint, which smooths);
  the recorded positions are used as-is.
- No ball distance, no speed-over-time, no sub-minute resolution, no speed
  zones other than the five listed.
- No query parameters or windows: the route always covers the whole match.
- No app-level routes (`register_app`), no changes to shared modules, no state
  other than the per-match `features/intensity.json` cache file.
- No building of the feature registry itself: `tasks/feature_registry.md` is a
  separate task and a hard precondition (see the Interface section).
