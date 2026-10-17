# Feature: events — detected event timeline for a stored match

## Purpose

Detect notable moments in a stored recording and expose them as one sorted,
time-stamped timeline: player sprints, fast ball peaks, possession changes
between the two teams, and long balls. The feature is a read-only analytics
plug-in for the library app: it uses the feature registry
(`tasks/feature_registry.md`) and never modifies the stored match. The
timeline answers "what happened, when, and who was involved", with one event
object per detected occurrence and a stable, deterministic order.

## Interface

- Module: `src/sideline/features/events.py` (importable as
  `sideline.features.events`), a feature of the registry: it defines
  `register(router, ctx)` which adds routes to the `APIRouter` mounted under
  `/api/matches/{match_id}`.
- Pure function: `detect_events(frames, fps=25) -> list`. `frames` is the
  list of parsed frame dicts in file order (format in
  `docs/WEBAPP_CONTRACT.md`). Returns the full event list as JSON-ready
  dicts (rounded exactly as in the Rules), sorted as in Rule R10. Must not
  mutate `frames` or any frame dict.
- Route: `GET /api/matches/{match_id}/events`.
  - `200` with exactly `{"match_id": <match id str>, "events": [<event>, ...]}`.
  - `404` with `{"error": <str>}` for an unknown match id or an id that does
    not match `m_[0-9a-f]{10}` (via `ctx.require(match_id)`). No other
    status codes; the route accepts no query parameters and no body.
- Caching: the handler computes the events list once per fresh recording via
  `ctx.cache_json(match_id, "events", compute)`, so the list is stored at
  `<match dir>/features/events.json` and served from that file while it is
  newer than `frames.jsonl`; otherwise it is recomputed and rewritten
  atomically. The stored JSON is the events list itself (not the full
  response). Repeated GETs are idempotent.
- Event dict, exact keys (all events): `frame` (int, the frame number),
  `time_s` (float, seconds since the first frame of the recording), `type`
  (one of `"sprint"`, `"ball_speed_peak"`, `"long_ball"`,
  `"possession_change"`), `team` (`"left"`, `"right"` or `null`), `player`
  (int track id or `null`). Type-specific extra keys:
  - `sprint`: `speed_m_s`, `distance_m`, `duration_s` (floats, 2 decimals).
  - `ball_speed_peak`: `speed_m_s` (float, 2 decimals).
  - `long_ball`: `distance_m` (float, 2 decimals).
  - `possession_change`: `from_team` (`"left"` or `"right"`).

## Rules

Named constants (module level; the four sprint/step constants are reused
from `sideline.analytics`, not redefined): `FPS = 25` (frames/s, from
`_common`), `BALL_PEAK_SPEED = 15.0` m/s, `LONG_BALL_DISTANCE = 25.0` m,
`LONG_BALL_WINDOW = 50` frames (= 2.0 s), `SPRINT_SPEED = 7.0` m/s,
`SPRINT_MIN_SECONDS = 1.0` s, `MAX_STEP_GAP = 5` frames,
`PLAYER_SPEED_CAP = 12.0` m/s.

- R1 Time. `time_s = round((frame - first_frame) / FPS, 3)` where
  `first_frame` is the frame number of the first frame of the recording.
  Example: first frame 120, event at frame 245 → (245-120)/25 = 5.0.
- R2 People. Within one frame the first entry per id wins; later duplicate
  entries for the same id are ignored. A person has a position when `xy` is
  a list of 2 finite numbers. Sprint candidates have role `player` or
  `goalkeeper`; possession owners additionally need team `"left"` or
  `"right"`.
- R3 Step. A person's track is the list of their positioned appearances in
  frame order. A step from appearance `(f0, p0)` to `(f1, p1)`: gap
  `g = f1 - f0` (frames), `d = sqrt(dx^2 + dy^2)` (m), `t = g / FPS` (s),
  `v = d / t` (m/s). A step is *counted* when `g <= MAX_STEP_GAP` and
  `v <= PLAYER_SPEED_CAP`. Example: (0,0) → (0.32,0) in 1 frame: v =
  0.32/0.04 = 8.0 m/s, counted.
- R4 Sprint. A run is a maximal consecutive group of counted steps with
  `v >= SPRINT_SPEED`. The run is a sprint when its total gap (sum of the
  steps' gaps) divided by FPS is at least `SPRINT_MIN_SECONDS` (with a
  1e-9 s tolerance). The event's `frame` is the run's last appearance
  frame; `speed_m_s` is the run's highest step speed (earliest on ties),
  `distance_m` the sum of the run's step distances, `duration_s` the total
  gap / FPS; `team` is the person's team at the event frame (null if not
  left/right); `player` is the id. Example: 25 steps of 0.32 m at 8.0 m/s
  from frame 200: total gap 25 frames = 1.0 s, distance 25 x 0.32 = 8.0 m
  → sprint event at frame 225 with speed_m_s 8.0, distance_m 8.0,
  duration_s 1.0.
- R5 Ball observations. A frame is a ball observation when `ball` is an
  object with `xy` a list of 2 finite numbers. Observations are used in
  frame order.
- R6 Ball speed sample. For two consecutive observations `(f0, a)` and
  `(f1, b)`, one sample is attributed to frame `f1` with speed
  `v = dist(a, b) / ((f1 - f0) / FPS)` (m/s). Example: (0,0) → (10,0) from
  frame 0 to frame 25: v = 10/1.0 = 10.0 m/s at frame 25.
- R7 Ball speed peak. Samples with `v > BALL_PEAK_SPEED` are grouped into
  runs of consecutive samples (adjacent in the sample list, i.e. adjacent
  observations). One event per run, at the frame of the highest sample of
  the run (earliest on ties), with `speed_m_s = round(max v, 2)`,
  `team` null, `player` null. Example: a 2.0 m/frame flight (50.0 m/s) for
  12 frames produces exactly 1 event, at the first sample frame, speed_m_s
  50.0.
- R8 Long ball. For each observation at frame `t`, its anchor is the first
  observation with frame `>= t - LONG_BALL_WINDOW`. The observation is
  *long* when the anchor is strictly before `t` and the anchor-to-`t`
  distance is `> LONG_BALL_DISTANCE` (m). Long observations form runs of
  consecutive observations (adjacent in the observation list); one event
  per run, at the run's first observation frame, with
  `distance_m = round(max anchor distance in the run, 2)`, `team` null,
  `player` null. Exactly 25.0 m is not a long ball. Example: the ball rests
  at (0,0) until frame 99, then moves 1.0 m/frame until frame 149 (50.0 m
  total). First long observation: frame 125, where the 2 s window
  displacement first reaches 26.0 m; the maximum window displacement is
  50.0 m (frame 149) → one event at frame 125 with distance_m 50.0.
- R9 Possession. A frame has an owner when it is a ball observation and at
  least one person has a position: the owner is the eligible person (R2)
  closest to the ball (Euclidean distance); ties go to the smaller id. The
  first owned frame is not an event. A frame whose owner's team differs
  from the team of the most recent earlier owned frame is a possession
  change: event at that frame with `team` = the new team, `player` = the
  new owner's id, `from_team` = the previous team. A frame without an
  owner (no ball, or no eligible positioned person) keeps the previous
  owner. Example: ball at (10.5,0), id 1 (left) at (1,0) and id 2 (right)
  at (20,0): both 9.5 m away → the tie goes to id 1, team left.
- R10 Order. The timeline is sorted by `(frame, type)` ascending; `type`
  compares alphabetically, so at one frame the order is
  `ball_speed_peak`, `long_ball`, `possession_change`, `sprint`.
- R11 Rounding. `time_s` to 3 decimals; `speed_m_s`, `distance_m`,
  `duration_s` to 2 decimals (Python `round`).
- R12 Independence. The detectors run over the same frames and do not
  exclude each other: one kick can produce both a `ball_speed_peak` and a
  `long_ball`; a `sprint` and a `possession_change` can share a frame.

## Edge cases

- One frame: the smallest storable recording. No pairs exist, so the
  answer is 200 with `"events": []`. A zero-frame recording cannot be
  stored at all (the upload answers 400), so the route never sees an empty
  list.
- Missing ball: `ball` null or `ball.xy` null means no ball observation
  there — no speed sample, no long ball, no owner at that frame; sprints
  are still detected. A ball that vanishes for 25 frames and reappears 10 m
  away 26 frames later gives v = 10/(26/25) = 9.615... ≈ 9.6 m/s: below
  both thresholds, no events.
- Missing positions: a person without `xy` has no appearance there; the
  next step simply spans the gap and is counted when `g <= 5`. Example: a
  player at 0.32 m/frame missing frames 13-15 has the step 12 → 16 with
  g = 4, d = 1.28 m, v = 8.0 m/s, so the sprint run continues (total
  duration 29/25 = 1.16 s, distance 9.28 m).
- Duplicate ids: the first entry per id per frame wins everywhere (R2).
  Example: id 7 listed twice in every frame, first at (0,0), second at
  (60,0) for frames 0-14 and (5,0) for frames 15-29, plus id 8 (right) at
  (20,0) and the ball at (0,0): the owner is id 7 (left) on every frame →
  no possession change; reading the last entry instead would fabricate a
  change at frame 15.
- Gaps in frame numbers: all times and speeds come from frame numbers, so
  a missing stretch of frames changes durations, not positions. Example:
  frames 0-4 with the ball at (0,0), next frame 30 with the ball at
  (26,0): the sample is v = 26/(26/25) = 25.0 m/s at frame 30 (a peak) and
  the anchor distance is 26.0 m (a long ball), both at frame 30.
- Huge input: a 25,000-frame recording. Budget: the first (uncached) GET
  completes in under 15.0 s, the second GET (served from the feature
  cache) in under 5.0 s, and the result is exactly the 2 events the rules
  predict for the fixture.

## Non-goals

- No mapping to video time or wall-clock timestamps (frame-relative
  seconds only); no filtering or query parameters on the route.
- No cross-type deduplication (a kick may legitimately be both a fast ball
  and a long ball), no per-team possession share or other aggregate stats,
  no referee/team assignment, no machine learning.
- The feature writes nothing except its own `features/events.json` cache
  file; it never touches `frames.jsonl`, `meta.json` or other features'
  caches.
