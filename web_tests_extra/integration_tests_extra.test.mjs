// Extra integration-oriented tests for the match library web pages and their
// libraries. They push the shapes the /api/matches endpoints actually return
// (docs/WEBAPP_CONTRACT.md) through the page code and the web/lib modules,
// plus the boundary values the pages can produce. No DOM timers, no network.

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { JSDOM, VirtualConsole } from "jsdom";

const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(here, "../src/sideline/web");
const render = await import(pathToFileURL(path.join(webDir, "lib", "render.js")).href);
const replay = await import(pathToFileURL(path.join(webDir, "lib", "replay.js")).href);
const pitch = await import(pathToFileURL(path.join(webDir, "lib", "pitch.js")).href);

const DASH = "\u2013";

function container() {
  const dom = new JSDOM("<!doctype html><body><div id='c'></div></body>", {
    virtualConsole: new VirtualConsole(),
  });
  return dom.window.document.getElementById("c");
}

// One frame in the shape the frames endpoint returns (contract section 1).
function apiFrame(frame) {
  return {
    frame,
    delay: 125,
    people: [
      { id: 1, role: "player", team: "left", jersey: "7", box: [10, 20, 40, 80], xy: [0, 1] },
      { id: 2, role: "player", team: "right", jersey: "8", box: [50, 60, 40, 80], xy: [0, -2] },
    ],
    ball: { box: [5, 5, 10, 10], xy: [1, 0.2] },
    colors: { left: "#d03030", right: "#2050c0" },
  };
}
function apiFrames(n, first = 0) {
  return Array.from({ length: n }, (_, i) => apiFrame(first + i));
}

// ------------------------------------------------------- page code reads

test("the library page reads items from the list response and links rows to the match page", () => {
  const libraryHtml = fs.readFileSync(path.join(webDir, "library.html"), "utf8");
  const renderJs = fs.readFileSync(path.join(webDir, "lib", "render.js"), "utf8");
  assert.match(libraryHtml, /fetch\("\/api\/matches"/);
  // The list endpoint answers {"items": [...]}: the page must read items.
  assert.match(libraryHtml, /matches\.items/);
  // Rows open the match page with the id as a query parameter.
  assert.match(renderJs, /"\/match\.html\?id=" \+ encodeURIComponent/);
});

test("the match page requests the contract endpoints for meta, analytics, frames and exports", () => {
  const matchHtml = fs.readFileSync(path.join(webDir, "match.html"), "utf8");
  assert.match(matchHtml, /URLSearchParams\(location\.search\)\.get\("id"\)/);
  assert.match(matchHtml, /"\/api\/matches\/" \+ enc\(id\)/);
  assert.match(matchHtml, /fetchJson\(base \+ "\/analytics"\)/);
  assert.match(matchHtml, /"\/frames\?start=" \+ start/);
  assert.match(matchHtml, /"\/export\.csv\?kind=positions"/);
  assert.match(matchHtml, /"\/export\.csv\?kind=players"/);
  assert.match(matchHtml, /summary\.players/);
  assert.match(matchHtml, /summary\.teams/);
  assert.match(matchHtml, /heat\.left/);
});

// ------------------------------------------------------------- renderLibrary

test("renderLibrary renders one row per match from the list response shape", () => {
  const c = container();
  const matches = [
    { id: "m_ab12cd34ef", name: "First half", created_at: "2026-01-01T10:00:00Z",
      frame_count: 750, duration_s: 30, size_bytes: 262144 },
    { id: "m_0000000009", name: "", created_at: "2026-01-02T10:00:00Z",
      frame_count: 1500, duration_s: 60, size_bytes: 1048576 },
  ];
  render.renderLibrary(c, matches, {});
  const rows = [...c.querySelectorAll(".match-row")];
  assert.equal(rows.length, 2);
  assert.equal(rows[0].getAttribute("data-match-id"), "m_ab12cd34ef");
  assert.equal(rows[0].querySelector(".name").textContent, "First half");
  assert.equal(rows[0].querySelector(".duration").textContent, "0:30");
  assert.equal(rows[0].querySelector(".size").textContent, "256.0 KB");
  assert.equal(rows[0].querySelector(".open").getAttribute("href"),
    "/match.html?id=" + encodeURIComponent("m_ab12cd34ef"));
  // A match without a name falls back to its id.
  assert.equal(rows[1].querySelector(".name").textContent, "m_0000000009");
  assert.equal(rows[1].querySelector(".size").textContent, "1.0 MB");
});

test("renderLibrary shows the empty note for empty or non-array match lists", () => {
  for (const bad of [[], {}]) {
    const c = container();
    render.renderLibrary(c, bad, {});
    assert.equal(c.querySelector(".match-row"), null);
    assert.ok(c.querySelector("p.empty"), JSON.stringify(bad));
  }
});

// --------------------------------------------------------- renderPlayerTable

test("renderPlayerTable applies the page's default distance sort and formats units", () => {
  const c = container();
  const players = [
    { id: 1, team: "left", jersey: "7", role: "player",
      distance_m: 100.4, avg_speed_ms: 3.2, max_speed_ms: 6.7, sprints: 2 },
    { id: 2, team: "right", jersey: "8", role: "player",
      distance_m: 55, avg_speed_ms: 2.1, max_speed_ms: 5.9, sprints: 0 },
    { id: 3, team: "left", jersey: "9", role: "player",
      distance_m: 100.4, avg_speed_ms: 1, max_speed_ms: 4, sprints: 1 },
  ];
  render.renderPlayerTable(c, players, { sortKey: "distance_m", descending: true });
  const rows = [...c.querySelectorAll("tbody tr")];
  // Ties keep input order (stable sort): 1 then 3, then 55 m.
  assert.deepEqual(rows.map((r) => r.getAttribute("data-player-id")), ["1", "3", "2"]);
  assert.equal(rows[0].querySelector('td[data-key="distance_m"]').textContent, "100");
  assert.equal(rows[0].querySelector('td[data-key="avg_speed_ms"]').textContent, "3.2");
  assert.equal(rows[0].querySelector('td[data-key="max_speed_ms"]').textContent, "6.7");
  assert.equal(rows[0].querySelector('td[data-key="sprints"]').textContent, "2");
  const th = c.querySelector('th[data-key="distance_m"]');
  assert.ok(th.classList.contains("sorted"));
  assert.ok(th.classList.contains("desc"));
});

test("renderPlayerTable shows dashes for missing values and keeps them last in both directions", () => {
  const c = container();
  const players = [
    { id: 1, team: "left", jersey: "1", role: "player",
      distance_m: 5, avg_speed_ms: 1, max_speed_ms: 2, sprints: 0 },
    { id: 2, team: null, jersey: null, role: "player" },
    { id: 3, team: "right", jersey: "3", role: "player",
      distance_m: 10, avg_speed_ms: 1, max_speed_ms: 2, sprints: 0 },
  ];
  render.renderPlayerTable(c, players, { sortKey: "distance_m", descending: false });
  assert.deepEqual([...c.querySelectorAll("tbody tr")].map((r) => r.getAttribute("data-player-id")),
    ["1", "3", "2"]);
  const missing = c.querySelectorAll("tbody tr")[2];
  assert.equal(missing.querySelector('td[data-key="distance_m"]').textContent, DASH);
  assert.equal(missing.querySelector('td[data-key="team"]').textContent, DASH);
  render.renderPlayerTable(c, players, { sortKey: "distance_m", descending: true });
  assert.deepEqual([...c.querySelectorAll("tbody tr")].map((r) => r.getAttribute("data-player-id")),
    ["3", "1", "2"]);
});

// ------------------------------------------------------------ renderTeamCards

test("renderTeamCards renders both contract teams in left, right order with one-decimal metres", () => {
  const c = container();
  const teams = {
    left: { frames: 724, avg_centroid: [30.23, 0.13], avg_width: 28.99, avg_length: 17.23 },
    right: { frames: 690, avg_centroid: [8.0, -0.5], avg_width: 29.5, avg_length: 41.0 },
  };
  render.renderTeamCards(c, teams, { left: "#d03030", right: "#2050c0" });
  const cards = [...c.querySelectorAll(".team-card")];
  assert.deepEqual(cards.map((el) => el.getAttribute("data-team")), ["left", "right"]);
  const left = cards[0];
  assert.equal(left.querySelector(".stat.centroid .value").textContent, "30.2, 0.1");
  assert.equal(left.querySelector(".stat.width .value").textContent, "29.0 m");
  assert.equal(left.querySelector(".stat.length .value").textContent, "17.2 m");
  assert.equal(left.querySelector(".swatch").style.backgroundColor, "rgb(208, 48, 48)");
});

test("renderTeamCards shows the empty note when the summary has no team data", () => {
  for (const bad of [{}, { left: null, right: null }, null]) {
    const c = container();
    render.renderTeamCards(c, bad, null);
    assert.equal(c.querySelector(".team-card"), null);
    assert.ok(c.querySelector("p.empty"), JSON.stringify(bad));
  }
});

// --------------------------------------------------------- normalizeGrid + renderHeatmap

test("normalizeGrid keeps the 14x21 shape, max 0 for an all-zero heatmap, input untouched", () => {
  const zero = Array.from({ length: 14 }, () => Array(21).fill(0));
  const allZero = render.normalizeGrid(zero);
  assert.equal(allZero.max, 0);
  assert.equal(allZero.cells.length, 14);
  assert.equal(allZero.cells[0].length, 21);
  assert.ok(allZero.cells.every((row) => row.every((v) => v === 0)));

  const grid = zero.map((row) => row.slice());
  grid[7][10] = 42;
  grid[0][0] = -5; // negative counts as 0
  grid[1][1] = NaN; // non-finite counts as 0
  const result = render.normalizeGrid(grid);
  assert.equal(result.max, 42);
  assert.equal(result.cells[7][10], 1);
  assert.equal(result.cells[0][0], 0);
  assert.equal(result.cells[1][1], 0);
  assert.equal(grid[7][10], 42); // input not modified
});

test("renderHeatmap sizes the canvas from a real 14x21 grid and records the max", () => {
  const c = container();
  const grid = Array.from({ length: 14 }, () => Array(21).fill(0));
  grid[3][5] = 42;
  render.renderHeatmap(c, grid);
  const canvas = c.querySelector("canvas.heatmap");
  assert.ok(canvas);
  assert.equal(canvas.width, 21 * 20);
  assert.equal(canvas.height, 14 * 20);
  assert.equal(canvas.getAttribute("data-max"), "42");

  render.renderHeatmap(c, []);
  assert.equal(c.querySelector("canvas"), null);
  assert.ok(c.querySelector("p.empty"));
  render.renderHeatmap(c, null);
  assert.ok(c.querySelector("p.empty"));
  assert.equal(c.querySelector("canvas"), null);
});

// ------------------------------------------------------------------- Timeline

test("Timeline over a full 750-frame match clamps seeks and ends on the last frame", () => {
  const tl = new replay.Timeline({ frames: apiFrames(750), fps: 25 });
  assert.equal(tl.duration, 29.96);
  const atStart = tl.current();
  assert.equal(atStart.index, 0);
  assert.equal(atStart.t, 0);
  assert.equal(atStart.a.frame, 0);
  assert.equal(atStart.b.frame, 1);
  tl.seek(1e9); // seeks past the end clamp to the end
  assert.equal(tl.position, tl.duration);
  const atEnd = tl.current();
  assert.equal(atEnd.index, 749);
  assert.equal(atEnd.a, atEnd.b); // at the end, b is a itself
  assert.equal(atEnd.t, 0);
});

test("Timeline handles empty and single-frame lists", () => {
  const empty = new replay.Timeline({ frames: [], fps: 25 });
  assert.equal(empty.duration, 0);
  assert.equal(empty.current(), null);
  const single = new replay.Timeline({ frames: [apiFrame(0)], fps: 25 });
  assert.equal(single.duration, 0);
  const view = single.current();
  assert.equal(view.index, 0);
  assert.equal(view.a, view.b);
  assert.equal(view.t, 0);
});

test("Timeline interpolates across a gap of missing frames", () => {
  const f100 = { ...apiFrame(0), frame: 100 };
  const tl = new replay.Timeline({ frames: [apiFrame(0), f100], fps: 25 });
  assert.equal(tl.duration, 4); // (100 - 0) / 25
  tl.seek(2);
  const view = tl.current();
  assert.equal(view.a.frame, 0);
  assert.equal(view.b.frame, 100);
  assert.equal(view.t, 0.5);
});

test("Timeline rejects invalid fps, seeks and speeds", () => {
  assert.throws(() => new replay.Timeline({ frames: [], fps: 0 }), RangeError);
  assert.throws(() => new replay.Timeline({ frames: [], fps: -5 }), RangeError);
  assert.throws(() => new replay.Timeline({ frames: "nope" }), RangeError);
  const tl = new replay.Timeline({ frames: apiFrames(10), fps: 25 });
  assert.throws(() => tl.seek(NaN), RangeError);
  tl.seek(-1e9); // seeks before the start clamp to 0
  assert.equal(tl.position, 0);
  assert.throws(() => tl.setSpeed(3), RangeError);
  tl.setSpeed(0.25);
  assert.equal(tl.speed, 0.25);
});

test("Timeline playback is deterministic at a given speed and pauses at the end", () => {
  let nowMs = 0;
  const tl = new replay.Timeline({ frames: apiFrames(100), fps: 25, now: () => nowMs });
  assert.equal(tl.duration, 3.96);
  tl.play();
  nowMs = 1000; // one second of wall time at 1x
  const view = tl.tick();
  assert.equal(tl.position, 1.0);
  assert.equal(tl.playing, true);
  assert.equal(view.a.frame, 25); // 1 s * 25 fps
  nowMs = 10_000; // far past the end
  const end = tl.tick();
  assert.equal(tl.position, tl.duration);
  assert.equal(tl.playing, false); // the Timeline auto-pauses at the end
  assert.equal(end.a, end.b);
});

// ------------------------------------------------------- interpolation

test("interpolatePeople keeps shared ids, drops unseen ones and keeps null positions", () => {
  const a = {
    frame: 0,
    people: [
      { id: 1, xy: [0, 0], box: [0, 0, 10, 10] },
      { id: 2, xy: null, box: [1, 1, 2, 2] },
      { id: 3, xy: [5, 5], box: [3, 3, 4, 4] },
    ],
  };
  const b = {
    frame: 1,
    people: [
      { id: 1, xy: [2, 4], box: [0, 0, 12, 12] },
      { id: 2, xy: [9, 9], box: [1, 1, 2, 2] },
    ],
  };
  const out = replay.interpolatePeople(a, b, 0.5);
  assert.deepEqual(out, [
    { id: 1, xy: [1, 2], box: [0, 0, 11, 11] },
    { id: 2, xy: null, box: [1, 1, 2, 2] }, // null position is not interpolated
  ]);
});

test("interpolateBall is null without a ball and keeps the A position on null xy", () => {
  const a = { frame: 0, ball: null };
  const b = { frame: 1, ball: { box: [0, 0, 6, 6], xy: [1, 1] } };
  assert.equal(replay.interpolateBall(a, b, 0.5), null);

  const a2 = { frame: 0, ball: { box: [5, 5, 6, 6], xy: null } };
  const b2 = { frame: 1, ball: { box: [7, 7, 6, 6], xy: [3, 3] } };
  assert.deepEqual(replay.interpolateBall(a2, b2, 0.5), { box: [5, 5, 6, 6], xy: null });

  const a3 = { frame: 0, ball: { box: [0, 0, 6, 6], xy: [0, 0] } };
  const b3 = { frame: 1, ball: { box: [10, 10, 6, 6], xy: [4, 8] } };
  assert.deepEqual(replay.interpolateBall(a3, b3, 0.5), { box: [5, 5, 6, 6], xy: [2, 4] });
});

// ----------------------------------------------------------------- pitch.js

test("chunkRanges splits the 750-frame match span into the page's 250-frame chunks", () => {
  assert.deepEqual(pitch.chunkRanges(0, 749, 250), [
    { start: 0, end: 249 }, { start: 250, end: 499 }, { start: 500, end: 749 },
  ]);
  assert.deepEqual(pitch.chunkRanges(0, 750, 250), [
    { start: 0, end: 249 }, { start: 250, end: 499 }, { start: 500, end: 749 },
    { start: 750, end: 750 },
  ]);
  assert.deepEqual(pitch.chunkRanges(0, 249, 250), [{ start: 0, end: 249 }]);
  assert.deepEqual(pitch.chunkRanges(10, 9, 250), []);
  assert.deepEqual(pitch.chunkRanges(0, 0, 1), [{ start: 0, end: 0 }]);
  assert.throws(() => pitch.chunkRanges(0, 10, 0), RangeError);
  assert.throws(() => pitch.chunkRanges(0.5, 10, 250), RangeError);
  assert.throws(() => pitch.chunkRanges(0, 10.5, 250), RangeError);
});

test("chunkStartFor maps fractional frame numbers to their chunk start", () => {
  assert.equal(pitch.chunkStartFor(0, 0, 250), 0);
  assert.equal(pitch.chunkStartFor(249.9, 0, 250), 0);
  assert.equal(pitch.chunkStartFor(250.1, 0, 250), 250);
  assert.equal(pitch.chunkStartFor(749, 0, 250), 500);
  assert.equal(pitch.chunkStartFor(999.9, 500, 250), 750);
});

test("FrameCache evicts the least recently used chunk and answers frame lookups", () => {
  const cache = new pitch.FrameCache(2, 250);
  const chunk0 = [{ frame: 0 }, { frame: 100 }, { frame: 240 }];
  const chunk250 = [{ frame: 250 }, { frame: 300 }];
  const chunk500 = [{ frame: 500 }];
  assert.equal(cache.frameAt(150), null); // nothing cached yet
  cache.put(0, chunk0);
  cache.put(250, chunk250);
  assert.equal(cache.frameAt(150), chunk0[1]); // last frame <= 150
  assert.deepEqual(cache.starts(), [250, 0]); // the lookup refreshed chunk 0
  cache.put(500, chunk500); // evicts the least recently used chunk (250)
  assert.equal(cache.size, 2);
  assert.equal(cache.has(0), true);
  assert.equal(cache.has(250), false);
  assert.equal(cache.has(500), true);
  assert.deepEqual(cache.starts(), [0, 500]);
  assert.throws(() => cache.put(0.5, []), RangeError);
  assert.throws(() => cache.put(1000, "nope"), TypeError);
  assert.equal(cache.size, 2); // failed puts change nothing
});

test("pitchGeometry and the pitch/canvas mapping use the 113x76 canvas contract", () => {
  const g = pitch.pitchGeometry(1130, 760);
  assert.deepEqual(g, { k: 10, ox: 565, oy: 380 });
  assert.deepEqual(pitch.pitchToCanvas([52.5, 34], g), [1090, 720]);
  assert.deepEqual(pitch.pitchToCanvas([60, 0], g), [1165, 380]); // off-pitch is not clipped
  assert.deepEqual(pitch.canvasToPitch([1090, 720], g), [52.5, 34]);
  assert.deepEqual(pitch.pitchGeometry(0, 0), { k: 0, ox: 0, oy: 0 });
  assert.deepEqual(pitch.canvasToPitch([5, 5], { k: 0, ox: 0, oy: 0 }), [0, 0]);
  assert.deepEqual(pitch.pitchGeometry(NaN, 100), { k: 0, ox: 0, oy: 50 });
});

test("personStyle resolves frame colours, fallbacks, referees and keepers", () => {
  assert.deepEqual(
    pitch.personStyle({ role: "player", team: "left", jersey: "7" }, { left: "#112233" }, null),
    { fill: "#112233", stroke: "rgba(0,0,0,0.55)", strokeWidth: 1.5, label: "7" });
  assert.deepEqual(
    pitch.personStyle({ role: "player", team: "right", jersey: null }, null, { right: "#445566" }),
    { fill: "#445566", stroke: "rgba(0,0,0,0.55)", strokeWidth: 1.5, label: "" });
  assert.equal(pitch.personStyle({ role: "player", team: "right" }, null, null).fill, "#3e8ed0");
  assert.equal(pitch.personStyle({ role: "referee", jersey: "R" }, null, null).fill, "#f2d544");
  const keeper = pitch.personStyle({ role: "goalkeeper", team: "left", jersey: "1" }, { left: "#000000" }, null);
  assert.deepEqual([keeper.stroke, keeper.strokeWidth], ["#ffffff", 2.5]);
  assert.equal(pitch.personStyle(null, null, null).fill, "#9aa59f");
});
