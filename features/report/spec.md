# Feature: report — printable single-page HTML match report

## Purpose

For a stored match, render a printable single-page HTML report: a header with the
match name and span, a headline table (player count, distances, top speed, sprints,
ball figures), the average team shape, a top-5 players table, the three occupancy
heatmaps (left team, right team, ball) as inline SVG, and — for each of the four
optional features possession, formation, events and quality — a section embedding
that feature's data, but only when the app actually serves that feature's route, so
the report works with any subset of the four present (including none). The document
is fully self-contained: no external resources of any kind (no scripts, stylesheets,
fonts or images, not even an SVG namespace URL), and it carries A4 print CSS so it
prints one page per match. It plugs into the library app as a backend feature via
the registry described in `tasks/feature_registry.md`.

## Interface

**Cross-task dependency (hard precondition).** This feature assumes the feature
registry from `tasks/feature_registry.md` is already built: the
`src/sideline/features/` package (`__init__.py`, `_common.py` with
`FeatureContext`), the `register_all` hook in `create_library_app`, and
`GET /api/features`. This spec neither builds nor modifies any of that; without it
the feature cannot be loaded and cannot be tested.

- Module to build: `src/sideline/features/report.py`, a feature of the
  `src/sideline/features/` package. It defines `register(router, ctx)` and adds
  the route below to the `router` mounted at `/api/matches/{match_id}`. Python 3.9,
  standard library only (plus FastAPI for the route), no global state, no new
  dependencies; importing it must not import torch, cv2 or numpy. The feature never
  mutates the meta dict, the summary or the frame dicts it is handed.
- Route: `GET /api/matches/{match_id}/report.html` (declared as
  `@router.get("/report.html")`).
- Status codes:
  - `200` — the HTML document below, with media type `text/html` and
    `charset=utf-8`.
  - `404` — via `ctx.require(match_id)`, for a malformed id (not
    `^m_[0-9a-f]{10}$`, no file access) or an unknown match.
  - No other status: nothing in the request can be invalid. The route reads no
    query parameters; extra query parameters are accepted and ignored.
- Public names of the module: `FEATURE_NAMES = ("possession", "formation",
  "events", "quality")`; `heatmap_svg(cells, fill) -> str`;
  `render_report(meta, summary, sections) -> str`; `register(router, ctx)`.
  `render_report` is a pure function of its three arguments (the document is
  deterministic — no clock, no randomness), so identical inputs always produce
  byte-identical documents.
- Caching: the report writes no cache file of its own. All data comes from the
  context's caches: `ctx.frames` (LRU) and the cached analytics summary; the
  per-request cost is one summary read plus linear string building over the 3
  heatmaps (294 cells each).
- `GET /api/features` (the registry) must list the feature with name `"report"`
  and `ok` `true` once built.

### The document (exact template)

One element per line, no indentation. Placeholders in braces; every number is
rendered by the display rule in Rule 7; `R` is one line per heatmap cell in Rule 6.

```
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{CSS, exactly the block of Rule 10, one line per rule}
</style>
</head>
<body>
<main class="report">
<header class="header">
<h1>{name}</h1>
<p class="meta">{frame_count} frames | {duration} s | 25 fps</p>
</header>
<section class="headline" id="headline">
<h2>Headline</h2>
<table>
<tr><th>players</th><td data-stat="players">{players}</td></tr>
<tr><th>left distance (m)</th><td data-stat="left-distance">{left}</td></tr>
<tr><th>right distance (m)</th><td data-stat="right-distance">{right}</td></tr>
<tr><th>top speed (m/s)</th><td data-stat="top-speed">{top}</td></tr>
<tr><th>sprints</th><td data-stat="sprints">{sprints}</td></tr>
<tr><th>ball distance (m)</th><td data-stat="ball-distance">{ball_dist}</td></tr>
<tr><th>ball frames</th><td data-stat="ball-frames">{ball_frames}</td></tr>
</table>
</section>
<section class="shape" id="shape">
<h2>Team shape</h2>
<table>
<tr><th>team</th><th>players</th><th>frames</th><th>centroid x</th><th>centroid y</th><th>width</th><th>length</th></tr>
<tr><td>left</td><td data-cell="left-players">{..}</td><td data-cell="left-frames">{..}</td><td data-cell="left-cx">{..}</td><td data-cell="left-cy">{..}</td><td data-cell="left-width">{..}</td><td data-cell="left-length">{..}</td></tr>
<tr><td>right</td><td data-cell="right-players">{..}</td><td data-cell="right-frames">{..}</td><td data-cell="right-cx">{..}</td><td data-cell="right-cy">{..}</td><td data-cell="right-width">{..}</td><td data-cell="right-length">{..}</td></tr>
</table>
</section>
<section class="top-players" id="top-players">
<h2>Top players</h2>
<table>
<tr><th>#</th><th>id</th><th>team</th><th>jersey</th><th>distance (m)</th><th>avg speed (m/s)</th><th>top speed (m/s)</th><th>sprints</th></tr>
{one data row per listed player, in list order}
</table>
</section>
<section class="heatmaps" id="heatmaps">
<h2>Heatmaps</h2>
<figure class="heat" id="heat-left">
<figcaption>left</figcaption>
{heatmap_svg(summary heatmaps "left", "#d03030")}
</figure>
<figure class="heat" id="heat-right">
<figcaption>right</figcaption>
{heatmap_svg(summary heatmaps "right", "#2050c0")}
</figure>
<figure class="heat" id="heat-ball">
<figcaption>ball</figcaption>
{heatmap_svg(summary heatmaps "ball", "#404040")}
</figure>
</section>
<section class="feature-sections" id="feature-sections">
{one block per present optional feature, in FEATURE_NAMES order}
</section>
</main>
</body>
</html>
```

The top-players data row (all cells in this order):

```
<tr data-player="{id}"><td>{rank}</td><td>{id}</td><td>{team}</td><td>{jersey}</td><td>{distance}</td><td>{avg}</td><td>{max}</td><td>{sprints}</td></tr>
```

The feature block (one per present optional feature):

```
<section class="feature" id="section-{name}">
<h2>{name}</h2>
<pre class="data">{JSON, html-escaped}</pre>
</section>
```

The document always contains the `feature-sections` wrapper, even when zero
optional features are present (the wrapper is then empty).

## Rules

1. **Data access.** The handler resolves the match with `ctx.require(match_id)`
   (the meta dict, or 404) and reads `ctx.analytics(match_id)` — the cached
   analytics summary with exactly the shape of section 2 of
   `docs/WEBAPP_CONTRACT.md`. The feature never calls `analytics.summarize`
   itself, never reads `frames.jsonl` itself and reads no query parameters.
   Example: on the canonical match of Rule 12, `ctx.require` gives
   `frame_count` 50 and `duration_s` 2.0; the summary carries 5 players.
2. **Counted players.** Every report figure that concerns players uses only the
   summary's `players` entries whose `role` is `"player"` or `"goalkeeper"`
   **and** whose `team` is `"left"` or `"right"`; referees (role `"referee"`)
   and unassigned people (team or role null) are excluded from the headline,
   the top table and the shape table's player counts. A left-team goalkeeper
   counts as a left player. Example: the canonical match has 5 counted players;
   adding a left-team referee (role `"referee"`, team `"left"`, 7.2 m moved at a
   7.5 m/s top speed) keeps the count at 5 and the left distance at 0.0.
3. **Headline.** `players` = number of counted players. `left-distance` =
   `round(0.0 + sum(p["distance_m"] of the left counted players), 2)`
   (0.0 when the list is empty; likewise right, always a 2-decimal float). `top-speed` = `round(max(p["max_speed_ms"]), 2)` over
   counted players (0.0 when there are none). `sprints` = `sum(p["sprints"])`.
   `ball-distance` = `summary["ball"]["distance_m"]`, `ball-frames` =
   `summary["ball"]["frames_with_ball"]` (both as-is). Example (canonical match):
   players 5, left 0.0, right `round(12.25 + 0.0, 2)` = 12.25, top speed
   `round(6.25, 2)` = 6.25, sprints 0, ball 6.12 over 50 frames.
4. **Team shape.** One table row per team, always both rows. Cells, in order:
   `players` = counted players of that team; `frames`, `centroid x`, `centroid y`,
   `width`, `length` from `summary["teams"][team]` (`avg_centroid[0]`,
   `avg_centroid[1]`, `avg_width`, `avg_length`). When a value is null (the team
   had no qualifying frame, e.g. fewer than 3 positioned players) the cell is `-`.
   Example (canonical match): left = players 3, frames 50, centroid (-10.0, 0.0)
   (each frame's 3 left players sit at x = -10, y = -10/0/10, so the per-frame
   centroid is exactly (-10, 0) and the per-frame width is 20), width 20.0,
   length 0.0; right = players 2, frames 0, all four metric cells `-`.
5. **Top players.** The counted players sorted by `distance_m` descending, ties by
   `id` ascending, truncated to 5 entries (fewer when fewer counted players).
   `rank` is 1-based; `distance`, `avg`, `max`, `sprints` are the summary values
   as-is; `jersey` is the summary jersey or `-` when null. Example (canonical
   match, distances 12.25, 0.0, 0.0, 0.0, 0.0): the rows are id 10 (rank 1,
   12.25 m, 6.25, 6.25, 0), then ids 1, 2, 3, 11 (ranks 2-5, all 0.0) in id order.
6. **Heatmap SVG.** For each of `left`, `right`, `ball` the summary grid (14 rows
   x 21 columns, row = y band, column = x band) is drawn by `heatmap_svg(cells,
   fill)` as one `<svg width="{10*nx}" height="{10*ny}" viewBox="0 0 {nx} {ny}">`
   line (210 x 140 for the standard 21 x 14 grid) with exactly `nx * ny`
   `<rect>` lines between it and `</svg>`: one per cell, row-major (row 0 first;
   within a row, column 0 to nx-1), in the exact form
   `<rect x="{col}" y="{row}" width="1" height="1" fill="{fill}" opacity="{op}"/>`
   with no `xmlns` attribute. `op` = `str(round(min(count / scale, 1.0), 2))`
   where `scale` = max(1, the largest cell count of that one map) — so the busiest
   cell of a map is always at full strength and an all-zero map renders every cell
   at `opacity="0.0"`. Fills: left `#d03030`, right `#2050c0`, ball `#404040`.
   Example: the canonical right map holds (row, col, count) (7,12,10), (7,13,20),
   (7,14,20), (9,14,50) — scale 50, opacities 0.2, 0.4, 0.4, 1.0, e.g.
   `<rect x="12" y="7" width="1" height="1" fill="#2050c0" opacity="0.2"/>`; the
   ball map holds (7,10,20) and (7,11,30) — scale 30, opacities 0.67 and 1.0.
7. **Header, title, display rule, escaping.** `{name}` is `meta["name"]`;
   `{frame_count}` and `{duration}` are `meta["frame_count"]` and
   `meta["duration_s"]` (the span-based seconds, e.g. 2.0 for 50 consecutive
   frames). `{title}` = `{name} - match report`. Numbers are rendered with `str`
   of the exact value they carry (summary values are already 2-decimal; counts and
   ranks are ints) — e.g. 20.0 displays `20.0`, 12.25 displays `12.25` — and
   `null` displays `-`. The name and every jersey are HTML-escaped
   (`&`, `<`, `>`, quotes); example: a match named `A & B <C>` renders
   `<h1>A &amp; B &lt;C&gt;</h1>` and title `A &amp; B &lt;C&gt; - match report`.
8. **Optional feature sections.** At request time, for each name in
   `FEATURE_NAMES` (fixed order possession, formation, events, quality — the
   document order, regardless of route registration order), scan
   `request.app.router.routes` in list order and take the first route whose
   `path` is exactly `/api/matches/{match_id}/{name}` and whose `methods` contain
   GET. If no such route exists, the section is absent. If one exists, call
   `route.endpoint(match_id=match_id)`; the section is included only when the
   call succeeds and returns a dict or a list — an exception (including an
   HTTPException or a TypeError from a foreign signature) or any other return
   value omits the section without affecting the rest of the report (still 200).
   The section's `<pre class="data">` holds `json.dumps(data, indent=2,
   sort_keys=True)`, HTML-escaped. Example: a possession route returning
   `{"match_id": "m_ab", "share": 0.6}` renders, unescaped,
   `{\n  "match_id": "m_ab",\n  "share": 0.6\n}`.
9. **Self-contained, single page, deterministic.** The document must contain no
   `<script`, no `<link`, no `<img`, no `<iframe`, no `<object`, no `<embed`, no
   `src=` attribute, no `@import`, no `url(`, and no occurrence of `http://` or
   `https://` anywhere (so no SVG `xmlns`); exactly one `<!DOCTYPE html>` line,
   the first one. It carries no wall-clock value, so two GETs for the same state
   return byte-identical bodies.
10. **Print CSS.** The `<style>` block is exactly:

```css
* { box-sizing: border-box; }
body { margin: 0; color: #111; font-family: Arial, Helvetica, sans-serif; font-size: 10pt; }
.report { max-width: 190mm; margin: 0 auto; padding: 8mm; }
h1 { font-size: 20pt; margin: 0 0 2mm 0; }
.meta { color: #444; margin: 0 0 6mm 0; }
h2 { font-size: 12pt; border-bottom: 1px solid #999; margin: 6mm 0 2mm 0; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 1mm 2mm; text-align: right; }
th { background: #eee; }
th:first-child, td:first-child { text-align: left; }
.heatmaps { display: flex; flex-wrap: wrap; gap: 6mm; }
.heat { margin: 0; text-align: center; }
.heat svg { border: 1px solid #999; }
pre.data { font-size: 7pt; max-height: 60mm; overflow: hidden; }
@page { size: A4; margin: 12mm; }
@media print {
  .report { max-width: none; padding: 0; }
  section, figure, table { break-inside: avoid; }
}
```

(`@page` with `size: A4` and the `@media print` block are mandatory: the report
is an A4 printout, sections keep together on paper.)
11. **Idempotence and mutation.** Repeated GETs for an unchanged match return
    byte-identical bodies (with or without optional feature routes); the feature
    modifies nothing on disk except through the context's own caches and never
    mutates the meta, the summary or any frame dict.
12. **Canonical worked example.** 50 frames (0..49, 2.0 s) with a match named
    `Canonical match`. Every frame carries: left players 1, 2, 3 standing at
    (-10, -10), (-10, 0), (-10, 10) (jerseys 1-3); right player 10 at
    (10 + 0.25f, 0) (jersey 10) — 49 steps of 0.25 m at 6.25 m/s, 12.25 m total;
    right player 11 at (20, 10) (jersey 11), standing; the ball at
    (0.125f, 0) — 49 steps of 0.125 m, 6.125 m, which the summary rounds to
    6.12. The report then shows: header `50 frames | 2.0 s | 25 fps`, title
    `Canonical match - match report`; headline players 5, left 0.0, right 12.25,
    top speed 6.25, sprints 0, ball 6.12 / 50; shape left (3, 50, -10.0, 0.0,
    20.0, 0.0), right (2, 0, -, -, -, -); top rows [10, 1, 2, 3, 11] as in Rule 5;
    heatmaps as in Rule 6 (left map: rows 4, 7, 9 all in column 8 with count 50
    and opacity 1.0; right and ball maps exactly as in the Rule 6 example); no
    feature sections.
13. **Performance budget.** A 25,000-frame match (60,708,202 bytes as recorded,
    22 players per frame) must be served by a single GET in under 20 s of wall
    clock, including the one-time analytics summary computation (measured at
    about 8 s plus parsing on a laptop); a second GET, with the summary cached,
    in under 5 s. The first response must still carry the exact values
    `25000 frames | 1000.0 s | 25 fps`, players 22, left 61486.92, right 61486.92,
    top speed 5.59, sprints 0, ball 0.0 over 12500 frames, and its first top row
    `id 1, left, jersey 1, 5589.72 m, 5.59, 5.59, 0` (ids 1-5 tie at 5589.72 m).
    Keep the render linear: one rect line per cell, one pass per table.

## Edge cases

- **Empty input / nobody on pitch:** 25 frames with empty `people` lists and no
  ball -> 200; headline players 0, both distances 0.0, top speed 0.0, sprints 0,
  ball 0.0 / 0; both shape rows show players 0, frames 0 and dashes; the
  top-players table has no data rows; all three maps are all-zero (294 rects at
  `opacity="0.0"` each); header `25 frames | 1.0 s | 25 fps`.
- **One frame:** duration 0.04 s; every distance, speed and sprint 0.0/0; one top
  row per counted player with 0.0 values; both shape rows dash.
- **Missing ball:** a ball absent in every frame -> ball 0.0 / 0 and an all-zero
  ball map; everything else unaffected.
- **Missing positions:** an appearance with `xy` null (or malformed) simply does
  not add a heatmap cell and no step; a person whose appearances are all
  unpositioned still gets a top row of 0.0 when their team is left/right.
- **Duplicate ids:** first entry per id per frame wins; a second entry of the
  same id at a far position is ignored (if it won, the 99.8 m jump would be
  discarded and the distance would read 0.0 instead of the true 0.2 m).
- **Gaps in frames:** only recorded frames exist; the header duration is the
  span, e.g. frames 0, 10, 20 -> (20 - 0 + 1) / 25 = 0.84 s; a person moving
  1.0 m per 10-frame gap makes no counted step (gap 10 > 5) -> distance 0.0,
  while the heatmap still holds the 3 observed positions (1 cell, count 3).
- **Referees:** a fast left-team referee (7.2 m at 7.5 m/s top speed) never
  appears: headline players/top speed/left distance come from the 1 standing
  left player only, and no `data-player="50"` row exists.
- **Null jersey:** a counted player whose most common jersey is null shows `-`
  in the jersey column.
- **Name escaping:** a match named `A & B <C>` renders escaped in the `<h1>` and
  the `<title>` (Rule 7 example).
- **Optional features, any subset:** 0 of the 4 -> empty wrapper; 1 (e.g.
  possession) -> exactly one block with the route's JSON; all 4 -> the four
  blocks in possession, formation, events, quality order no matter the route
  registration order; a failing or non-dict-list endpoint is silently omitted
  while its siblings still render and the status stays 200.
- **Huge input:** see Rule 13 (25,000 frames, budgets 20 s / 5 s with exact pins).
- **No mutation / idempotence:** see Rule 11 (byte-identical bodies; the frames
  endpoint answer is unchanged by a report GET).

## Non-goals

- No PDF, no raster images, no JavaScript, no sub-page navigation: exactly one
  self-contained HTML document per GET.
- No query parameters or windows: the whole match is always reported.
- No report-specific cache file: only the context's frames LRU and the analytics
  cache are used; the render itself is cheap and stateless.
- No changes to shared modules, no app-level routes (`register_app`), no building
  of the feature registry itself (`tasks/feature_registry.md` is a separate task
  and a hard precondition), no re-computation of analytics the summary already
  provides, and no i18n.
