# Feature spec: data_quality — a quality report for one recording

## Purpose

A recorded match (`frames.jsonl`, format in `docs/WEBAPP_CONTRACT.md`) may have tracking
drops, id switches, uncalibrated camera stretches and unreadable jerseys, and none of the
existing endpoints say so. `analytics.summarize` and the per-feature routes silently
produce numbers out of whatever survived. This feature computes a **quality report** for
one stored match: the share of frames with at least 8 tracked people, the frame ranges
with no tracked people at all (the signature of an uncalibrated or turned-away camera),
the average tracked people per frame, an estimate of fragment tracks (a track that starts
within 1.5 s and 3 m of a track that just ended — the typical id switch), the share of
tracked people assigned to a team, the share with a readable jersey, the share of frames
with the ball found, plain-language warnings for every failing aspect, and an overall
grade from A to D. It plugs into the library app as a backend feature: one new module,
no edits to shared files.

## Dependencies (hard cross-task dependency)

This feature is a plug-in for the feature registry of `tasks/feature_registry.md`:
`FeatureContext` (`require`, `frames`, `cache_json`), the registry's mount of every
feature router under `/api/matches/{match_id}`, and the `GET /api/features` listing.
**The registry is not present in this branch.** Building it is explicitly a non-goal of
this spec (another task owns it), so until that task is merged the tests in this
directory fail at import time. Everything this feature computes is defined by this spec
alone; only the HTTP wiring assumes the registry as documented there.

## Interface

Module to build: `src/sideline/features/data_quality.py` (Python 3.9, standard library
only — no torch, cv2 or numpy; FastAPI only through what `ctx` provides; no global
state; importing it must not import torch, cv2 or numpy). The feature never mutates its
input frame dicts.

Public names:

```python
FPS = 25
MIN_TRACKED = 8
FRAGMENT_WINDOW_S = 1.5
FRAGMENT_DISTANCE_M = 3.0
LONG_RANGE_FRAMES = 25

def tracked_people(frame):
    """The (id, x, y) tuples of the tracked people of one frame dict, in input
    order. A tracked person is an entry whose "id" is an int (a bool does not
    count) and whose "xy" is a 2-element list or tuple of finite numbers (int
    or float; bool, None, inf and nan do not count). role, team, jersey and
    box are irrelevant here. If one id appears twice in the frame, the first
    entry wins and the duplicate is ignored."""

def quality_report(frames, fps=25):
    """The full quality document (without "match_id") for the given frame
    dicts, every number rounded exactly as the route serves it (Rules 3-11).
    frames: list of frame dicts in the recorded format. Never mutates frames."""

def register(router, ctx):
    """Defines GET /quality on the given APIRouter (the registry mounts it
    under /api/matches/{match_id})."""
```

### HTTP route

`GET /api/matches/{match_id}/quality` (declared as `@router.get("/quality")`).

| Case | Status | Body |
|---|---|---|
| success | 200 | the document below |
| unknown or malformed `match_id` (not `^m_[0-9a-f]{10}$`) | 404 | `{"detail": {"error": "match not found"}}`, raised by `ctx.require`; malformed ids never touch the filesystem |

The route takes **no** query parameters; any query parameters sent are ignored (200 with
the same document). There is therefore no 400 for this route.

Document (the response; every float already rounded as stated):

```json
{"match_id": "m_ab12cd34ef", "fps": 25, "first_frame": 0, "last_frame": 49,
 "frame_count": 50, "tracked_per_frame": 25.0, "good_frames_share": 1.0,
 "no_people_ranges": [], "fragment_tracks": 0, "team_assigned_pct": 96.0,
 "jersey_read_pct": 96.0, "ball_found_pct": 100.0, "warnings": [], "grade": "A"}
```

* `fps`: the fps passed to `quality_report` (25 unless the caller says otherwise).
* `first_frame`, `last_frame`: int, or `null` only for empty input.
* `frame_count`: number of frames after cleaning (Rule 1).
* `tracked_per_frame`: Rule 3. `good_frames_share`: Rule 4.
* `no_people_ranges`: Rule 5, ascending, each `{"start": int, "end": int, "frames": int}`
  with `frames = end - start + 1`.
* `fragment_tracks`: int, Rule 6.
* `team_assigned_pct`, `jersey_read_pct`, `ball_found_pct`: Rules 7-9 (percents).
* `warnings`: Rule 10, exact strings, fixed order.
* `grade`: Rule 11, one of `"A"`, `"B"`, `"C"`, `"D"`.

The route builds the full response (including `match_id`) through
`ctx.cache_json(match_id, "quality", compute)`: the complete response document is stored
in `<match>/features/quality.json`, served only while it is newer than `frames.jsonl`,
written atomically (temp file + rename). Repeating the request returns byte-identical
JSON.

## Rules

1. **Cleaning.** Frames are deduplicated by `frame` number (the last occurrence wins)
   and sorted ascending before any rule is applied; the upload already rejects
   non-increasing frames, so this guards direct calls only. `delay`, `colors`, `box` are
   ignored. Example: two lines both with `"frame": 5` — only the second line's people
   are used.
2. **Tracked person.** As in `tracked_people`: int id (bool excluded) + `xy` a
   2-element list/tuple of finite numbers. First entry per id per frame wins.
   Examples: an entry with `id: 7` twice counts once (the first position); `xy: null`,
   `xy: [1]` and `xy: [Infinity, 0]` are not tracked; an `id` of `"7"` (a string) is not
   tracked. A person with `xy: null` (found in the image, not placed on the pitch) is
   **not** tracked: such a frame is part of a no-people range, which is exactly how an
   uncalibrated camera shows up.
3. **tracked_per_frame** = total tracked people over all frames, divided by the number
   of frames, the exact quotient rounded to 2 decimals (`round(x, 2)`); 0.0 for empty
   input. Example: 246 tracked over 75 frames → 3.28.
4. **Good frames.** A frame is *good* when it has at least `MIN_TRACKED` = 8 tracked
   people. `good_frames_share` = (good frames / frames), the exact fraction rounded to
   3 decimals; 0.0 for empty input. Examples: 100 good of 150 → 0.667; 85 of 100 → 0.85.
5. **No-people ranges.** A *no-people frame* has 0 tracked people. Every maximal run of
   consecutive recorded frame numbers of no-people frames is reported as
   `{"start", "end", "frames"}`; ranges are ascending in `start`; `frames` =
   `end - start + 1`. A gap in the recorded frame numbers ends the run even when the
   next recorded frame is also empty. Example: recorded frames 0..149 with 25..49 empty
   → `{"start": 25, "end": 49, "frames": 25}`; recorded 0..10 empty, 11 missing, 12
   empty → two ranges, 0..10 and 12..12.
6. **Fragment tracks.** A *track* is the set of tracked entries with one id across the
   recording; its *start* is its first recorded frame (position = its `xy` there) and
   its *end* its last recorded frame (position = its `xy` there). Track A is a
   **fragment** when there exists a different track B with (a) A starts in B's last
   frame or later — `A.start >= B.end`; a track that started while B was still alive is
   an overlap, not a fragment; (b) `(A.start - B.end) / fps <= 1.5`; and (c) the
   Euclidean distance between A's start position and B's end position is at most 3.0 m
   (both coordinates matter: `sqrt(dx^2 + dy^2)`, not `|dx| + |dy|`). `fragment_tracks`
   counts the starting tracks A that have at least one such B; each A is counted once,
   and one B may justify several A's. Examples: B ends at frame 24 at (10.0, 3.0); A
   starts at frame 25 at (11.6, 5.2): 1 frame = 0.04 s and
   `sqrt(1.6^2 + 2.2^2) = sqrt(7.4) ≈ 2.72` → fragment; A' at frame 25 at (13.2, 5.4):
   `sqrt(3.2^2 + 2.4^2) = 4.0` → not a fragment. B ends at frame 0 at (-10, -2); A
   starts at frame 37 at (-10.6, -2.8): 37/25 = 1.48 s and distance 1.0 → fragment;
   the same A starting at frame 38: 38/25 = 1.52 s → not a fragment.
7. **team_assigned_pct** = (tracked people with `team` "left" or "right" / all tracked
   people), the exact fraction times 100 rounded to 1 decimal; 0.0 when no one is
   tracked. Referees and unassigned people count against it. Example: 24 of 25 → 96.0.
8. **jersey_read_pct** = (tracked people whose `jersey` is a non-empty string — `str`
   with a non-blank `strip()`; `null`, `""` and non-strings do not count) over all
   tracked people, times 100 rounded to 1 decimal; 0.0 when no one is tracked.
   Example: 4 of 8 → 50.0.
9. **ball_found_pct** = (frames whose `ball` is not `null` / frames), times 100 rounded
   to 1 decimal; 0.0 for empty input. `ball` with `xy: null` **is** found (found but
   not placed); only `ball: null` is not found. Example: 1 of 3 frames → 33.3.
10. **Warnings.** Each warning is computed from the *rounded* document values (the same
    numbers the report shows), with the percent formatted as `%.1f`, and emitted in
    this fixed order; for a recording with at least one tracked person, a warning
    appears exactly when the matching grade check (Rule 11) fails:
    1. when the good-frames percent `round(100 * n_good / n_frames, 1)` is below 80.0:
       `"Only {p}% of frames have 8 or more tracked people."`
       Example: 100 of 150 → `"Only 66.7% of frames have 8 or more tracked people."`
    2. for each no-people range with at least `LONG_RANGE_FRAMES` = 25 frames, ascending:
       `"No tracked people in frames {start}..{end} ({n} frames) - the camera may be
       uncalibrated."` A 15-frame range is reported in `no_people_ranges` but produces
       no warning.
    3. when `fragment_tracks` >= 1:
       `"{k} tracks start within 1.5 s and 3 m of a track that just ended (possible id
       switches)."`
    4. when `team_assigned_pct` is below 90.0:
       `"Only {p}% of tracked people are assigned to a team."`
    5. when `jersey_read_pct` is below 50.0:
       `"Jersey numbers were read on only {p}% of tracked people."`
    6. when `ball_found_pct` is below 50.0:
       `"The ball was found in only {p}% of frames."`
    Warnings 4 and 5 are skipped when no one is tracked (their 0.0 would quote a
    nonexistent population); their percentages are still reported as 0.0.
11. **Grade.** Six checks, each evaluated on the *rounded* document values:
    (1) good-frames percent `round(100 * n_good / n_frames, 1)` >= 80.0;
    (2) no no-people range with at least 25 frames;
    (3) `fragment_tracks` == 0;
    (4) `team_assigned_pct` >= 90.0;
    (5) `jersey_read_pct` >= 50.0;
    (6) `ball_found_pct` >= 50.0.
    Score = number of passing checks: 6 → `"A"`, 5 → `"B"`, 4 → `"C"`, 3 or fewer →
    `"D"`. Empty input (0 frames) → `"D"` with no warnings. Example: the canonical
    fixture (below) passes all six → `"A"`; a recording whose only failing check is the
    ball (33.3%) passes five → `"B"`.
12. **Rounding.** Every metric is one ratio over the whole recording, rounded once at
    the end — never an average of per-frame or per-second rounded values. Example: 4
    jersey reads over 8 tracked people is exactly 50.0% even though the per-frame
    rounded percents (100.0, 33.3, 33.3) would average to 55.5.
13. **Caching.** As in the Interface: the cached file holds the complete response
    document (including `match_id`); the response equals it; it is recomputed and
    rewritten atomically when `frames.jsonl` is newer.
14. **Performance budget.** One `GET /api/matches/{id}/quality` over the 25,000-frame,
    22-people-per-frame recording of the performance test (1,000 s, ~58 MB of JSONL)
    completes in **less than 30 s** wall clock. The fragment rule must not compare every
    pair of tracks: a track starting at frame `f` can only match tracks that ended in
    `[f - 37, f]` (1.5 s at 25 fps is 37.5 frames), so bucket end events by frame and
    scan at most 38 frames per start. A reference implementation serves this request in
    a few seconds; the budget is one order of magnitude above that.

## Edge cases

* **Empty input (direct calls only).** `quality_report([])` →
  `{"fps": 25, "first_frame": null, "last_frame": null, "frame_count": 0,
  "tracked_per_frame": 0.0, "good_frames_share": 0.0, "no_people_ranges": [],
  "fragment_tracks": 0, "team_assigned_pct": 0.0, "jersey_read_pct": 0.0,
  "ball_found_pct": 0.0, "warnings": [], "grade": "D"}`. The route can never see an
  empty match (the store rejects 0-frame uploads).
* **One frame.** A single frame at `frame: 7` → `first_frame = last_frame = 7`;
  a 1-frame track starts and ends in that frame and can justify a fragment when
  another track ended in the same frame within 3 m (Rule 6 allows
  `A.start == B.end`).
* **Missing ball.** `ball: null` in a frame is simply not found; `ball` present with
  `xy: null` counts as found (Rule 9). Two recordings identical except that one has
  `ball: {"box": [...], "xy": [0, 0]}` and the other `ball: {"box": [...], "xy": null}`
  in a frame answer the identical `ball_found_pct`; a `ball: null` frame lowers it.
* **Missing positions.** People with invalid `xy` are not tracked; a frame whose only
  people have `xy: null` is a no-people frame (Rule 2) and joins a range.
* **Duplicate ids in one frame.** The first entry wins (Rule 2 example).
* **Duplicate frame numbers (direct calls only).** The last occurrence wins (Rule 1).
* **Gaps in frames.** Recorded 0..10 empty, 11 missing, 12..14 empty → ranges 0..10
  and 12..14 (Rule 5); frame gaps never merge ranges.
* **Zero tracked people, non-empty recording.** All three per-person percentages are
  0.0, warnings 4 and 5 are skipped, and the grade is `"D"` (checks 1, 2, 4, 5, 6
  fail; only check 3 passes when there are no fragments).
* **Huge input.** The 25,000-frame / 22-people recording is processed within the 30 s
  budget of Rule 14; the answer stays one number per metric plus the (empty) ranges.
* **No mutation / idempotence.** `tracked_people` and `quality_report` never mutate
  their inputs (tests compare a deep copy); repeated GETs answer byte-identical JSON
  from the cache file.

## Canonical fixture (used by the tests)

50 frames (0..49), 25 tracked people in every frame, ball found in every frame:
11 left players, id `i` at `(-38 + 2(i-1), (i-1) - 5)` for `i` = 1..11 (jersey `str(i)`);
11 right players, id `11 + j` at `(20 + 2(j-1), 5 - (j-1))` for `j` = 1..11 (jersey
`str(j)`); the left goalkeeper id 120 at (-48, 0) and the right goalkeeper id 130 at
(48, 0), both with jersey "1"; one referee id 900 at (0, 8), `team: null`, `jersey:
null`. All 24 team members carry a jersey, so of 25 tracked people per frame: 24 are
team-assigned and 24 have a readable jersey. Every track spans the whole recording. The
exact document (besides `match_id`) is the one in the Interface example:
`tracked_per_frame` 25.0, `good_frames_share` 1.0, no ranges, 0 fragments,
`team_assigned_pct` 96.0, `jersey_read_pct` 96.0, `ball_found_pct` 100.0, no warnings,
grade `"A"`.

## Non-goals

* No per-player, per-team or per-second breakdowns; the report is one flat document.
* No speed, accuracy or calibration (homography) analysis; "uncalibrated" is reported
  only as no-people ranges, never diagnosed.
* No query parameters, no app-level routes (`register_app`), no other endpoints.
* No changes to shared modules and no building of the feature registry
  (`tasks/feature_registry.md` is a dependency, not this feature).
* No persistence beyond the `quality.json` cache file; no client-side rendering.
