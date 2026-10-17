# Possession feature

## Purpose

Estimate which team has the ball at every moment of a recorded match, and
summarize it as a segment timeline plus percentages per team and per 5-minute
window. The estimate is deliberately simple and fully reproducible: a team
"has" the ball on a frame when one of its players is the nearest player to
the ball and is within 3 m; a team counts the ball as possessed only when it
keeps it that way for at least 0.5 s. The feature plugs into the library app
through the feature registry (`tasks/feature_registry.md`): one module, no
edits to shared files.

## Interface

Module to build: `src/sideline/features/possession.py` (package
`sideline.features`, discovered by the registry; standard library only, no
new dependencies, no global state).

1. Pure function, importable without the app:

   ```python
   compute_possession(frames, fps=25) -> dict
   ```

   `frames` is a list of frame dicts exactly as written to `frames.jsonl`
   (`docs/WEBAPP_CONTRACT.md` section 1), in strictly increasing `frame`
   order. The function must not mutate `frames` (not even nested dicts). It
   returns a JSON-serializable dict with exactly these keys: `fps`,
   `frame_count`, `duration_s`, `timeline`, `percentages`, `windows`.

2. `register(router, ctx)` defines one route on the match-level router:
   `@router.get("/possession")`, i.e. the full path
   `GET /api/matches/{match_id}/possession`. The handler:

   - calls `ctx.require(match_id)` - an id that does not match
     `^m_[0-9a-f]{10}$` or is not in the store answers **404** (error body
     from `ctx.require`);
   - gets `frames = ctx.frames(match_id)`;
   - returns, **200**, the dict
     `{"match_id": match_id, **compute_possession(frames, fps=25)}`
     computed through `ctx.cache_json(match_id, "possession", compute)`, so
     the result is cached in `<match>/features/possession.json` and
     recomputed only when `frames.jsonl` becomes newer than the cache.
   - accepts no query parameters; the route itself answers no 400/422.

Response JSON (200) - exactly these keys, owner values are the strings
`"left"`, `"right"` or `"none"`:

```json
{
  "match_id": "m_0123456789",
  "fps": 25,
  "frame_count": 40,
  "duration_s": 1.6,
  "timeline": [
    {"start_frame": 0, "end_frame": 19, "owner": "left", "seconds": 0.8},
    {"start_frame": 20, "end_frame": 39, "owner": "right", "seconds": 0.8}
  ],
  "percentages": {"left": 50.0, "right": 50.0, "none": 0.0},
  "windows": [
    {"window": 0, "from_s": 0, "to_s": 300, "frames": 40,
     "left": 50.0, "right": 50.0, "none": 0.0}
  ]
}
```

`compute_possession` returns the same document without `match_id`.

## Rules

1. **Time base.** `fps = 25`, so one frame is `1/25 s = 0.04 s`.
   `duration_s = round((last_frame - first_frame + 1) / fps, 2)`.
   Example: frames 0..39 -> `40/25 = 1.6`.
2. **Ball present on a frame** iff `frame["ball"]` is a dict whose `xy` is a
   2-element list of non-null numbers. Otherwise the ball is absent on that
   frame and the frame's raw owner is `"none"`. Example:
   `{"box": [900, 500, 12, 12], "xy": [3.1, -2.0]}` is present;
   `ball: null` and `ball.xy: null` are absent.
3. **Players.** An entry of `frame["people"]` counts iff it is a dict with
   `role` in `{"player", "goalkeeper"}`, `team` in `{"left", "right"}` and an
   `xy` 2-element list of non-null numbers. Referees, `team: null`,
   `xy: null` and non-dict entries are ignored. A missing `people` key is
   treated as an empty list.
4. **Distance.** Euclidean, in metres:
   `d = sqrt((x_ball - x_player)^2 + (y_ball - y_player)^2)`.
   Example: ball at `(3.1, -2.0)`, player at `(1.0, -2.0)` ->
   `d = sqrt(2.1^2 + 0^2) = 2.1`.
5. **Reach threshold: 3.0 m.** A player is in reach iff `d <= 3.0` (the
   boundary is included). Example: ball at `(0, 0)`; player at `(3, 0)` ->
   `d = 3.0` -> in reach; player at `(3.1, 0)` -> `d = 3.1` -> out of
   reach.
6. **Raw owner of a frame.** Let `d_left`/`d_right` be the minimum distance
   from the ball to the team's counting players (`None` when the team has
   no counting player on the frame).
   - ball absent -> `"none"`;
   - both teams present with exactly equal minimum distances -> `"none"`
     (a perfect tie is unresolvable). Example: ball `(0, 0)`, left `(-1, 0)`,
     right `(1, 0)` -> `1.0 == 1.0` -> `"none"`;
   - the smallest existing distance is `> 3.0` -> `"none"`. Example: ball
     `(0, 0)`, left `(10, 0)`, right `(30, 0)` -> `10 > 3.0` -> `"none"`;
   - otherwise the team with the smaller distance. Example: ball `(0, 0)`,
     left `(1, 0)` (`d = 1.0`), right `(50, 0)` (`d = 50.0`) -> `"left"`.
   - only one team has players -> that team wins when in reach (no tie is
     possible); example: only left `(1, 0)` around ball `(0, 0)` ->
     `"left"`.
7. **Runs.** Frames in the file order form a *run* when their `frame`
   numbers are consecutive (`n, n+1, ...`) and their raw owner is the same;
   a run is maximal. A missing frame number ends the run even if the next
   present frame has the same raw owner. Example: raw-`left` frames
   0..6 and 10..16 (7..9 missing) form two runs of 7 frames, not one of 14.
8. **Hold requirement: 0.5 s.** A run of length `L` frames spans
   `(L - 1)` frame intervals of `0.04 s`; a team run counts as possession
   iff `(L - 1) / fps >= 0.5`. At `fps = 25` that is `L >= 14`:
   `13 x 0.04 = 0.52 s >= 0.5` (accepted), `12 x 0.04 = 0.48 s < 0.5`
   (rejected). A confirmed run keeps its team for **all** of its frames; an
   unconfirmed team run is re-labelled `"none"` on every frame. Runs never
   accumulate across gaps or other stretches: left for 6 frames, ball away
   for 6, left for 6 frames yields no possession.
9. **Timeline.** The maximal runs of consecutive frames with the same final
   owner (after rule 8), in frame order. Entry:
   `{"start_frame": s, "end_frame": e, "owner": o, "seconds":
   round((e - s + 1) / fps, 2)}` (inclusive). Example: 30 left frames,
   0..29 -> `{"start_frame": 0, "end_frame": 29, "owner": "left",
   "seconds": 1.2}` because `30/25 = 1.2`.
10. **Percentages.** Over the `frame_count` frames of the recording,
    `pct = round(100.0 * n_owner / frame_count, 1)` for each of
    `left`, `right`, `none`. Example: 48 frames with 14 left, 14 right,
    20 none -> `29.2`, `29.2`, `41.7` (rounded parts need not sum to
    100.0).
11. **5-minute windows.** Match time of frame `n` is
    `t = (n - first_frame) / fps` seconds; window index
    `w = floor(t / 300)` (at `fps = 25`: `floor((n - first_frame) / 7500)`;
    a frame at exactly `t = 300.0`, i.e. 7500 frames after the first, is in
    window 1). One entry per window that contains at least one frame:
    `{"window": w, "from_s": 300*w, "to_s": 300*(w+1), "frames": k,
    "left": ..., "right": ..., "none": ...}` with the rule-10 formula over
    the window's `k` frames; windows without frames are omitted. Example:
    first frame 0, frames 0..24 and 15000..15024 -> window 0 with 25 frames
    and window 2 with 25 frames; windows 1 and 3 are omitted.
12. **Rounding.** Python `round()`: 1 decimal for percentages, 2 decimals
    for `seconds` and `duration_s`. Distances are compared in metres
    (comparing squared distances is equivalent and acceptable).
13. **Complexity.** One pass over the frames, a constant-size scan of each
    frame's `people` (O(frames x players)); no sorting, no per-frame
    allocation beyond the output document.

## Edge cases

- **Empty input** (pure function only - the API rejects empty uploads with
  400): `compute_possession([])` returns `fps 25`, `frame_count 0`,
  `duration_s 0.0`, `timeline []`, `percentages` all `0.0`, `windows []`.
- **One frame.** A single frame with no ball: `timeline` is
  `[{start 0, end 0, "none", 0.04}]`, percentages `none 100.0`, one window
  with `frames: 1`, `duration_s 0.04`.
- **Missing ball.** `ball: null` or `ball.xy: null` on a frame -> `"none"`
  there; a 15-frame left stretch followed by 15 ball-less frames gives
  timeline `left 0..14` then `none 15..29` and percentages `left 50.0`,
  `none 50.0`.
- **Missing positions.** No people, only referees, only one team, `xy:
  null`, non-dict entries, or a missing `people` key all count as "no
  player" for that team/frame; e.g. ball at `(0, 0)`, only a referee at
  `(0.5, 0)` and the nearest player at `(10, 0)` -> `"none"`.
- **Duplicate ids.** Each `people` entry is considered independently; no
  deduplication by `id`. Ball `(0, 0)`, two left entries with id 5 at
  `(50, 0)` and `(1, 0)` -> `d_left = 1.0` -> `"left"`.
- **Gaps in frames.** Frame numbers strictly increase but may skip; a gap
  breaks runs (7 + 7 left frames around a 3-frame gap give no possession)
  and window indices are computed from absolute frame numbers.
- **First frame not 0.** Durations, times and windows are measured from
  `first_frame`: 14 frames starting at 100 with a left hold give timeline
  `start_frame 100`, window 0 with `from_s 0`.
- **Huge input / performance budget.** A recording of 25,000 consecutive
  frames (0..24999) with 22 players and a ball per frame:
  `compute_possession` completes in **under 5 s**; the same recording via
  the HTTP route (uploaded, then `GET` possession) answers within **20 s**
  of the request. Expected result: one segment 0..24999 `left` of
  `1000.0 s`, percentages `left 100.0`, four windows with 7500, 7500, 7500
  and 2500 frames, each `left 100.0`.

## Non-goals

- No per-player, per-zone or per-half possession; no comparison between
  matches.
- No ball-speed, shot, pass or dead-ball (throw-in, goal kick) detection;
  the 3 m / 0.5 s rule is the whole model.
- No interpolation between frames, no smoothing, no clamping of
  coordinates: positions are trusted as recorded, even outside the pitch
  bounds.
- No query parameters or options on the route; `fps` is fixed at 25 by the
  recording contract (the pure function accepts an `fps` argument for
  testability).
- No changes to shared modules (`library.py`, `features/_common.py`,
  `matches.py`), no new dependencies, no mutation of the stored
  `frames.jsonl` or match meta.
