# Sideline web app: shared contract (v1)

Everything in the match library and analytics work builds on the two things below. Do not change them; if a task
needs something that is missing, say so in the task notes instead of inventing it.

## 1. Recorded match file: `frames.jsonl`

One JSON object per line, one line per processed frame, exactly as written by `python -m sideline run`:

```json
{"frame": 120, "delay": 0,
 "people": [{"id": 7, "role": "player", "team": "left", "jersey": "10", "box": [812, 301, 44, 120], "xy": [-12.34, 5.6]}],
 "ball": {"box": [900, 500, 12, 12], "xy": [3.1, -2.0]},
 "colors": {"left": "#d03030", "right": "#2050c0"}}
```

- `frame`: integer frame index; 25 frames per second. Lines are in increasing `frame` order; some frames can be missing.
- `delay`: output delay in frames the file was produced with (same for the whole file).
- `people[].id`: integer track id (stable for one person while tracked). `role` is `"player"`, `"goalkeeper"` or
  `"referee"`. `team` is `"left"`, `"right"` or `null` (referees and unknown). `jersey` is a string such as `"10"` or `null`.
- `people[].xy` and `ball.xy`: pitch coordinates in metres. Origin at the centre spot, x along the pitch from -52.5
  to 52.5, y across from -34 to 34. `ball` is `null` when the ball was not found; `ball.xy` is `null` when found in
  the image but not placed on the pitch. `colors` is `null` until team colours are known.

## 2. Analytics summary: `analytics.summarize(frames, fps=25) -> dict`

`frames` is the list of parsed frame objects from section 1. The result is JSON-serialisable and has exactly this shape:

```json
{"fps": 25, "frame_count": 750, "first_frame": 0, "last_frame": 749, "duration_s": 30.0,
 "players": [{"id": 7, "role": "player", "team": "left", "jersey": "10", "frames": 700, "seconds": 28.0,
              "distance_m": 105.2, "avg_speed_ms": 3.76, "max_speed_ms": 7.9, "sprints": 1}],
 "teams": {"left": {"frames": 700, "avg_centroid": [-10.2, 1.1], "avg_width": 31.0, "avg_length": 38.4},
           "right": {"frames": 690, "avg_centroid": [8.0, -0.5], "avg_width": 29.5, "avg_length": 41.0}},
 "heatmaps": {"grid": {"nx": 21, "ny": 14, "x_range": [-52.5, 52.5], "y_range": [-34, 34]},
              "left": [[0, 0, "... 21 integers per row, 14 rows (index = row for y, column for x)"]],
              "right": [[0]], "ball": [[0]]},
 "ball": {"frames_with_ball": 600, "distance_m": 140.3}}
```

## 3. Modules (all under `src/sideline/`)

| Module | Public interface |
|---|---|
| `analytics.py` | `load_frames(path)`, `player_stats(frames, fps=25)`, `team_shape(frames, team)`, `heatmap(...)`, `summarize(frames, fps=25)` |
| `matches.py` | `MatchStore(root)` and `create_router(store)` (a FastAPI `APIRouter`); calls only `analytics.summarize` |
| `web/lib/replay.js` | ES module with pure functions and a `Timeline` class (no DOM access) |
| `web/lib/render.js` | ES module whose functions render data into a DOM container (works in a plain DOM such as jsdom) |
| `web/library.html`, `web/match.html` | pages that use the two libraries and the `/api/matches` endpoints |

Rules for every module: standard library only on the Python side (plus FastAPI for `matches.py`), no network access,
no global state, and nothing in code, comments, docs or commit messages about who or what produced the work.
