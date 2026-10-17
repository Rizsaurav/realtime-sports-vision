# Feature: compare — side-by-side headline numbers for two matches

## Purpose

Answer "how do these two matches differ at a glance?" with one read-only
route: `GET /api/compare?a=<match id>&b=<match id>` returns the two
matches' headline numbers side by side — duration, number of tracked
people, team width and length (per team), average distance per tracked
person and ball found % — as a fixed list of 8 metric rows, each row
carrying the value of `a`, the value of `b` and the difference
(`a - b`). The numbers are the same ones the library app already shows
per match: this feature reuses the app's cached analytics summary for
each match, so a compare row can never disagree with the match's own
analytics page. It is an app-level backend feature plugged in through
the feature registry; it never modifies a stored match.

## Interface

**Cross-task dependency (hard precondition).** This feature assumes the
feature registry from `tasks/feature_registry.md` is already built: the
`src/sideline/features/` package (`__init__.py`, `_common.py` with
`FeatureContext` and its helpers), the `register_all` hook in
`create_library_app`, `GET /api/features`, and — used by this feature —
the **app-level routes addendum**, i.e. `register_all` also calls a
feature's `register_app(app, ctx)` after `register`. This spec neither
builds nor modifies any of that.

- Module to build: `src/sideline/features/compare.py`, a feature of the
  `src/sideline/features/` package. It defines
  - `register(router, ctx)`: **a no-op** (this feature has no routes
    under `/api/matches/{match_id}`); it must exist and must not raise.
  - `register_app(app, ctx)`: adds the route below to `app`.
  Standard library only (plus FastAPI for the route), Python 3.9
  compatible, no global state, no new dependencies; importing it must
  not import torch, cv2 or numpy. The feature never mutates the frame
  dicts or any file, and writes no state of its own.
- Route: `GET /api/compare` with two query parameters, `a` and `b`.
  Both are declared as **optional** string parameters (default `None`),
  so a missing parameter is answered by the feature with `400`, never
  by FastAPI validation with `422`. Extra query parameters are accepted
  and ignored.
- Status codes, checked in this exact order:
  1. `400` — `a` or `b` is missing or an empty string, body
     `{"error": "a and b are both required"}`. No file access.
  2. `400` — `a == b` (plain string equality), body
     `{"error": "a and b must be different matches"}`. Checked before
     any lookup, so the same unknown id twice is `400`, not `404`.
  3. `404` — either id is unknown or malformed (does not match
     `^m_[0-9a-f]{10}$`), raised by `ctx.require(match_id)`, with no
     file access for malformed ids.
  4. `200` — the JSON body below.
- Body on `200`:

```json
{
  "a": "m_0123456789",
  "b": "m_fedcba0123",
  "metrics": [
    {"name": "duration_s", "a": 4.0, "b": 4.0, "diff": 0.0},
    {"name": "tracked_people", "a": 5, "b": 6, "diff": -1},
    {"name": "team_width_left_m", "a": 10.0, "b": 20.0, "diff": -10.0},
    {"name": "team_width_right_m", "a": null, "b": 4.0, "diff": null},
    {"name": "team_length_left_m", "a": 4.0, "b": 2.0, "diff": 2.0},
    {"name": "team_length_right_m", "a": null, "b": 2.0, "diff": null},
    {"name": "avg_distance_m", "a": 6.65, "b": 7.12, "diff": -0.47},
    {"name": "ball_found_pct", "a": 80.0, "b": 20.0, "diff": 60.0}
  ]
}
```

- `a` and `b` echo the requested ids. `metrics` has exactly the 8 rows
  above, in exactly this order; every row has exactly the keys
  `name`, `a`, `b`, `diff`.
- Data access: for each side the feature takes `S = ctx.analytics(mid)`
  (the match's cached analytics summary, the same object the app's
  `GET /api/matches/{mid}/analytics` endpoint serves — in the library
  app it is computed with jitter smoothing on) and `ctx.require(mid)`
  for the 404. The feature never calls `analytics.summarize` itself,
  never reads `frames.jsonl` directly and keeps no cache file of its
  own.
- `GET /api/features` must list the feature with name `"compare"` and
  `ok` `true` once built (both `register` and `register_app` succeeded).

## Rules

1. **Check order.** Missing/empty parameter → `400`; then `a == b` →
   `400`; then `ctx.require(a)` and `ctx.require(b)` → `404` when
   either fails. Examples: `?a=&b=m_0123456789` → `400` (empty
   parameter); `?a=m_ffffffffff&b=m_ffffffffff` → `400` (the two ids
   are equal, even though unknown); `?a=m_123&b=m_0123456789` → `404`
   (`m_123` is malformed and fails `require` with no file access);
   `?a=m_0123456789&b=m_9876543210` → `404` (well-formed but never
   uploaded).
2. **`duration_s`.** The summary's `duration_s`, passed through
   unchanged: `round(frame_count / 25, 2)`, where `frame_count` is the
   number of recorded frames — **not** the span from first to last
   frame. Example: 100 recorded frames → `4.0`; 5 recorded frames
   numbered 0, 1, 2, 10, 11 → `5 / 25 = 0.2` (the 12-frame span would
   give `0.48`).
3. **`tracked_people`.** The number of entries in the summary's
   `players` list, i.e. the count of distinct person ids tracked in the
   recording — referees, unassigned people and people without a single
   valid position included. It is an integer. Example: 5 players plus a
   referee, an unassigned person and an unpositioned person → `7`.
4. **`team_width_{left,right}_m` and `team_length_{left,right}_m`.**
   The summary's `teams[side].avg_width` and `teams[side].avg_length`,
   passed through unchanged (metres, already rounded to 2 decimals by
   the summary, or `null` when the team has no frame with at least 3
   positioned players). Example: a left side with players at
   y = 0, 5, 10 and x spread 0..4 in every qualifying frame → width
   `10.0`, length `4.0`; a right side with only 2 positioned players
   in every frame → width `null`, length `null`.
5. **`avg_distance_m`.** The mean of the summary's per-person
   `distance_m` values (the already-2-decimal values, over the same
   population as `tracked_people`), rounded to 2 decimals with Python's
   `round`; `0.0` when the summary has no players. Example: distances
   4.75, 4.75, 4.75, 9.5, 9.5 → total `33.25`, mean `33.25 / 5 =
   6.65`; distances 9.5, 9.5, 9.5, 4.75, 4.75, 4.75 → total `42.75`,
   mean `42.75 / 6 = 7.125` → `7.12` (round half to even).
6. **`ball_found_pct`.** `round(100 * frames_with_ball / frame_count,
   2)` from the summary's `ball.frames_with_ball` (frames with a valid
   ball position; a ball found in the image but not placed on the pitch
   does not count) and the summary's `frame_count`. A stored match
   always has at least 1 frame (the upload rejects empty files), so the
   division is defined. Examples: 80 of 100 frames → `80.0`; 12 of 25
   → `48.0`; 4 of 10 → `40.0`.
7. **`diff`.** `a - b` computed from the values the row reports (that
   is, after the 2-decimal rounding of rules 2–6), then rounded to 2
   decimals with Python's `round`; for `tracked_people` it is the plain
   integer difference. It is `null` whenever either side is `null`.
   Examples: `6.65 - 7.12 = -0.47`; `7.12 - 6.65 = 0.47`;
   `5 - 6 = -1`; `null - 4.0 = null`; `10.0 - 4.0 = 6.0`.
8. **Determinism and idempotence.** The result depends only on the two
   stored summaries; the feature adds no timestamp, random value or
   request-order state. Repeating the same request returns exactly the
   same JSON.
9. **No mutation.** The feature only reads through `ctx` (meta/summary
   are the store's own caches); it does not modify frame dicts, match
   files or the summaries, and creates no files.
10. **Canonical worked example.** Two 100-frame matches (4.0 s each).
    Match A: at frames 0, 5, ..., 95 (20 appearances, 19 steps of
    gap 5, t = 0.2 s) left players 1, 2, 3 move 0.25 m per step
    (4.75 m each, v = 1.25) at y = 0, 5, 10 with x bases 0, 2, 4;
    right players 4, 5 move 0.5 m per step (9.5 m each, v = 2.5).
    Ball placed in the 80 frames with `f % 5 != 2`. Match B: same
    layout, but left players 1, 2, 3 move 0.5 m per step (9.5 m each)
    at y = 0, 10, 20 with x bases 0, 1, 2, and right players 4, 5, 6
    move 0.25 m per step (4.75 m each) at y = 0, 2, 4 with x bases
    40, 41, 42; ball placed in the 20 frames with `f % 5 == 0`.
    `GET /api/compare?a=<A>&b=<B>` then returns exactly the body in
    the Interface section: duration `4.0 / 4.0 / 0.0`; tracked
    `5 / 6 / -1`; width left `10.0 / 20.0 / -10.0`, width right
    `null / 4.0 / null`; length left `4.0 / 2.0 / 2.0`, length right
    `null / 2.0 / null`; average `6.65 / 7.12 / -0.47`; ball
    `80.0 / 20.0 / 60.0`. Reversed (`a=<B>&b=<A>`) swaps the two value
    columns and flips every non-null diff's sign.
11. **Performance budget.** One `GET /api/compare` comparing two
    25,000-frame matches (11 left + 11 right people in every frame,
    about 62 MB per recording, both summaries cold) must complete in
    under 45 s of wall clock. The heavy work is the summaries' first
    computation (player stats with smoothing, team shapes and the three
    21 x 14 heatmaps per match), which `ctx.analytics` caches in
    `analytics.json`; the route itself must only read the two summaries
    and build 8 rows — no frame-level loop of its own. A second request
    for the same pair is served from the caches and returns quickly.

## Edge cases

- **Missing / empty / identical parameters:** `a` or `b` absent or `""`
  → `400` (rule 1); `a == b` → `400` even when the id is unknown or
  malformed (equality precedes lookup).
- **Unknown or malformed ids:** any id not matching
  `^m_[0-9a-f]{10}$` (e.g. `m_123`, `M_0123456789`, `compare`,
  `m_012345678901`) or a well-formed id that was never uploaded →
  `404`, with no file access for malformed ids.
- **No tracked people:** `tracked_people` `0`, `avg_distance_m` `0.0`
  (not an error and not `null`), all four team shape values `null`.
- **A team without 3 positioned players in any frame:** that team's
  width and length are `null`, and the row's `diff` is `null` even when
  the other side has a value (`null - 4.0 = null`).
- **Ball never found:** `ball_found_pct` `0.0`; a ball found but not
  placed (`ball.xy` null) does not count, a valid position does.
- **Gaps in frames:** `duration_s` and `ball_found_pct` are based on
  the number of recorded frames, not on the first..last span (rule 2).
- **One-frame match:** duration `0.04`; one tracked stationary person
  gives `tracked_people` `1` and `avg_distance_m` `0.0`.
- **Reversed arguments:** values swap columns, every non-null diff
  changes sign; nothing else changes.
- **Huge input:** covered by rule 11's budget; the exact pinned values
  for the 25,000-frame recordings are: duration `1000.0`, tracked
  `22`, width `20.0` and length `10.0` for both teams on both sides,
  average `1562.44` (24,999 steps of 1/16 m = 1562.4375 m) vs
  `781.22` (24,999 steps of 1/32 m = 781.21875 m), diff `781.22`,
  ball found `50.0` on both sides.
- **Idempotence / no mutation:** repeating the request returns exactly
  the same JSON; the matches' frames, files and summaries are
  unmodified (see `test_frames_not_mutated`).

## Non-goals

- No winner/loser flags, percentages of the difference, or any
  "which is better" judgement beyond the numeric `diff` (a minus b).
- No comparison of more than 2 matches, no per-player or per-minute
  breakdowns (the per-match analytics endpoint covers those), no
  windows or time ranges.
- No UI for the comparison: this is the backend route only.
- No cache file of its own (the summaries already are cached), no
  changes to shared modules, and no building of the feature registry
  itself (`tasks/feature_registry.md` is a separate task and a hard
  precondition).
