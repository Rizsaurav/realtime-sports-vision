// Extra tests for the analytics summary contract (docs/WEBAPP_CONTRACT.md
// section 2) as consumed by the web render library
// (src/sideline/web/lib/render.js, spec in tasks/ui_pages.md).
//
// analytics.py is pure Python, so the JavaScript side that the analytics
// work fixes is the consumer of the summarize() result: player table rows,
// team shape values (including the all-null shape) and the 14x21 integer
// heatmap grids. Every assertion here is required by the contract shape or
// by the documented render.js behaviour on that shape.

import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { JSDOM, VirtualConsole } from "jsdom";

const here = path.dirname(fileURLToPath(import.meta.url));
const libPath = process.env.RENDER_LIB
  ? path.resolve(process.env.RENDER_LIB)
  : path.resolve(here, "../src/sideline/web/lib/render.js");
const lib = await import(pathToFileURL(libPath).href);
const { heatColor, normalizeGrid, renderPlayerTable, renderTeamCards, renderHeatmap } = lib;

const DASH = "\u2013";

function setup() {
  const dom = new JSDOM("<!doctype html><body><div id='c'></div></body>", {
    virtualConsole: new VirtualConsole(),
  });
  const doc = dom.window.document;
  return { dom, doc, c: doc.getElementById("c"), win: dom.window };
}

// ---------- analytics summary shape helpers (contract section 2) ----------

const GRID_NX = 21;
const GRID_NY = 14;

function zeroGrid() {
  return Array.from({ length: GRID_NY }, () => Array.from({ length: GRID_NX }, () => 0));
}

// The example summary from docs/WEBAPP_CONTRACT.md, with real 21x14 integer
// grids in place of the "..." ellipsis.
function contractSummary() {
  const left = zeroGrid();
  left[7][10] = 7; // centre spot hot cell
  left[3][4] = 2;
  const right = zeroGrid();
  right[10][15] = 5;
  const ball = zeroGrid();
  ball[7][0] = 3;
  return {
    fps: 25,
    frame_count: 750,
    first_frame: 0,
    last_frame: 749,
    duration_s: 30.0,
    players: [
      { id: 7, role: "player", team: "left", jersey: "10", frames: 700, seconds: 28.0,
        distance_m: 105.2, avg_speed_ms: 3.76, max_speed_ms: 7.9, sprints: 1 },
      { id: 5, role: "referee", team: null, jersey: null, frames: 600, seconds: 24.0,
        distance_m: 0.0, avg_speed_ms: 0.0, max_speed_ms: 0.0, sprints: 0 },
    ],
    teams: {
      left: { frames: 700, avg_centroid: [-10.2, 1.1], avg_width: 31.0, avg_length: 38.4 },
      right: { frames: 0, avg_centroid: null, avg_width: null, avg_length: null },
    },
    heatmaps: {
      grid: { nx: 21, ny: 14, x_range: [-52.5, 52.5], y_range: [-34, 34] },
      left, right, ball,
    },
    ball: { frames_with_ball: 600, distance_m: 140.3 },
  };
}

test("contract summary JSON round-trips with the exact shape", () => {
  const s = contractSummary();
  const parsed = JSON.parse(JSON.stringify(s));
  assert.deepEqual(Object.keys(parsed).sort(),
    ["ball", "duration_s", "first_frame", "fps", "frame_count", "heatmaps", "last_frame", "players", "teams"]);
  for (const map of ["left", "right", "ball"]) {
    const grid = parsed.heatmaps[map];
    assert.equal(grid.length, 14);
    for (const row of grid) {
      assert.equal(row.length, 21);
      for (const cell of row) {
        assert.equal(typeof cell, "number");
        assert.ok(Number.isInteger(cell) && cell >= 0);
      }
    }
  }
  assert.deepEqual(parsed.heatmaps.grid, { nx: 21, ny: 14, x_range: [-52.5, 52.5], y_range: [-34, 34] });
  for (const p of parsed.players) {
    assert.deepEqual(Object.keys(p).sort(),
      ["avg_speed_ms", "distance_m", "frames", "id", "jersey", "max_speed_ms", "role", "seconds", "sprints", "team"]);
  }
  assert.deepEqual(parsed.ball, { frames_with_ball: 600, distance_m: 140.3 });
});

// ---------- renderPlayerTable on summary players ----------

function cells(c, rowIdx) {
  const tr = c.querySelectorAll("tbody tr")[rowIdx];
  return [...tr.querySelectorAll("td")].map((td) => td.textContent);
}

test("renderPlayerTable formats a contract player row", () => {
  const { c } = setup();
  const players = contractSummary().players;
  renderPlayerTable(c, players, {});
  const rows = c.querySelectorAll("tbody tr");
  assert.equal(rows.length, 2);
  assert.equal(rows[0].getAttribute("data-player-id"), "7");
  // column order: team, jersey, id, role, distance_m, avg_speed_ms, max_speed_ms, sprints
  assert.deepEqual(cells(c, 0), ["left", "10", "7", "player", "105", "3.8", "7.9", "1"]);
});

test("renderPlayerTable shows en dashes for a null-team referee from the summary", () => {
  const { c } = setup();
  const players = contractSummary().players;
  renderPlayerTable(c, players, {});
  assert.deepEqual(cells(c, 1), [DASH, DASH, "5", "referee", "0", "0.0", "0.0", "0"]);
});

test("renderPlayerTable sorts summary players numerically by max speed", () => {
  const { c } = setup();
  const players = [
    { id: 3, role: "player", team: "left", jersey: "3", frames: 1, seconds: 0.04,
      distance_m: 10.0, avg_speed_ms: 1.0, max_speed_ms: 8.0, sprints: 0 },
    { id: 1, role: "player", team: "left", jersey: "1", frames: 1, seconds: 0.04,
      distance_m: 5.0, avg_speed_ms: 1.0, max_speed_ms: 2.0, sprints: 0 },
    { id: 2, role: "player", team: "right", jersey: "2", frames: 1, seconds: 0.04,
      distance_m: 20.0, avg_speed_ms: 2.0, max_speed_ms: 4.0, sprints: 0 },
  ];
  renderPlayerTable(c, players, { sortKey: "max_speed_ms" });
  assert.deepEqual([...c.querySelectorAll("tbody tr")].map((tr) => tr.getAttribute("data-player-id")), ["1", "2", "3"]);
  renderPlayerTable(c, players, { sortKey: "max_speed_ms", descending: true });
  assert.deepEqual([...c.querySelectorAll("tbody tr")].map((tr) => tr.getAttribute("data-player-id")), ["3", "2", "1"]);
});

// ---------- renderTeamCards on summary teams ----------

test("renderTeamCards formats contract team values and the all-null shape", () => {
  const { c } = setup();
  renderTeamCards(c, contractSummary().teams, { left: "#d03030", right: "#2050c0" });
  const left = c.querySelector('.team-card[data-team="left"]');
  assert.equal(left.querySelector(".centroid .value").textContent, "-10.2, 1.1");
  assert.equal(left.querySelector(".width .value").textContent, "31.0 m");
  assert.equal(left.querySelector(".length .value").textContent, "38.4 m");
  const right = c.querySelector('.team-card[data-team="right"]');
  assert.equal(right.querySelector(".centroid .value").textContent, DASH);
  assert.equal(right.querySelector(".width .value").textContent, DASH);
  assert.equal(right.querySelector(".length .value").textContent, DASH);
});

// ---------- heatmaps from the summary ----------

test("normalizeGrid on a summary heatmap keeps 14x21 shape and does not mutate", () => {
  const grid = contractSummary().heatmaps.left;
  const before = JSON.stringify(grid);
  const r = normalizeGrid(grid);
  assert.equal(r.max, 7);
  assert.equal(r.cells.length, 14);
  for (const row of r.cells) assert.equal(row.length, 21);
  assert.equal(r.cells[7][10], 1);
  assert.ok(Math.abs(r.cells[3][4] - 2 / 7) < 1e-12);
  assert.equal(JSON.stringify(grid), before);
});

test("renderHeatmap sizes the canvas from the summary grid and reports data-max", () => {
  const { c } = setup();
  renderHeatmap(c, contractSummary().heatmaps.left);
  const canvas = c.querySelector("canvas.heatmap");
  assert.ok(canvas);
  assert.equal(c.children.length, 1);
  assert.equal(canvas.width, 420); // 21 * 20
  assert.equal(canvas.height, 280); // 14 * 20
  assert.equal(canvas.getAttribute("data-max"), "7");
});

test("renderHeatmap of an all-zero summary heatmap gives data-max 0 and no empty note", () => {
  const { c } = setup();
  renderHeatmap(c, zeroGrid());
  const canvas = c.querySelector("canvas.heatmap");
  assert.ok(canvas);
  assert.equal(canvas.getAttribute("data-max"), "0");
  assert.equal(c.querySelector("p.empty"), null);
});

test("heatColor maps the extreme summary heatmap cells to the colour stops", () => {
  const grid = contractSummary().heatmaps.left;
  const { max } = normalizeGrid(grid);
  assert.equal(heatColor(0, max), "rgb(13, 17, 16)");
  assert.equal(heatColor(max, max), "rgb(255, 59, 48)");
});

test("repeated renders of the same summary heatmap do not leak state", () => {
  const { c } = setup();
  const summary = contractSummary();
  renderHeatmap(c, summary.heatmaps.left);
  renderHeatmap(c, zeroGrid());
  assert.equal(c.children.length, 1);
  assert.equal(c.querySelector("canvas.heatmap").getAttribute("data-max"), "0");
  // the summary grid passed in is untouched by the renders
  assert.equal(summary.heatmaps.left[7][10], 7);
});
