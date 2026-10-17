// Extra adversarial tests, second pass, for tasks/ui_pages.md (the render
// library in src/sideline/web/lib/render.js).
//
// These pin down spec rules the existing suites (web_tests/render.test.mjs,
// web_tests_extra/improve_ui_extra.test.mjs, web_tests_extra/
// improve_library_list_extra.test.mjs) do not check: the exact named-export
// surface, no-global DOM access, minute-boundary rounding and huge durations,
// literal toFixed(1) display boundaries in formatBytes, third-point heat
// interpolation and extreme finite clamping, boolean/string/ragged cells in
// normalizeGrid, large grids, null options objects, null ids, name values
// that are falsy but not "missing", listener lifecycle across repeated
// renders, cross-document containers, missing/non-numeric/float ids in the
// player table, leading-zero jerseys, code-unit string order, and the
// all-zero / single-row / row-longer-than-first heatmap edge cases.
// Every assertion is required by the spec text in tasks/ui_pages.md.

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
const {
  formatDuration, formatBytes, heatColor, normalizeGrid,
  renderLibrary, renderPlayerTable, renderTeamCards, renderHeatmap,
} = lib;

const DASH = "\u2013";

function setup() {
  const dom = new JSDOM("<!doctype html><body><div id='c'></div></body>", {
    virtualConsole: new VirtualConsole(),
  });
  const doc = dom.window.document;
  return { dom, doc, c: doc.getElementById("c"), win: dom.window };
}
function click(win, node) {
  const ev = new win.MouseEvent("click", { bubbles: true, cancelable: true });
  node.dispatchEvent(ev);
  return ev;
}
const P = (o) => ({ id: 1, role: "player", team: "left", jersey: "10", distance_m: 100, avg_speed_ms: 3, max_speed_ms: 6, sprints: 1, ...o });
const ids = (c) => [...c.querySelectorAll("tbody tr")].map((r) => r.getAttribute("data-player-id"));

// ---------- module surface ----------

test("the module exports exactly the eight documented functions and no default", () => {
  // "An ES module with named exports only (listed below)".
  assert.deepEqual(Object.keys(lib).sort(), [
    "formatBytes", "formatDuration", "heatColor", "normalizeGrid",
    "renderHeatmap", "renderLibrary", "renderPlayerTable", "renderTeamCards",
  ]);
  for (const f of Object.values(lib)) assert.equal(typeof f, "function");
});

test("the pure functions work in plain Node with no window/document globals", () => {
  // "No top-level DOM access", "never reads window or document globals".
  // The import at the top of this file already ran in plain Node; the
  // formatter must not need a DOM either.
  assert.equal(typeof globalThis.document, "undefined");
  assert.equal(typeof globalThis.window, "undefined");
  assert.equal(formatDuration(65), "1:05");
  assert.equal(formatBytes(2048), "2.0 KB");
  assert.equal(heatColor(5, 10), "rgb(255, 176, 32)");
  assert.deepEqual(normalizeGrid([[0, 5]]), { max: 5, cells: [[0, 1]] });
});

// ---------- formatDuration ----------

test("formatDuration: values that round onto a minute boundary", () => {
  // "Round to the nearest whole second (Math.round), then m:ss".
  assert.equal(formatDuration(59.9999999), "1:00");
  assert.equal(formatDuration(119.9999999), "2:00");
  assert.equal(formatDuration(3599.4), "59:59");
  assert.equal(formatDuration(3599.5), "60:00");
  // minutes unpadded, seconds two-digit
  assert.equal(formatDuration(605), "10:05");
});

test("formatDuration: minutes just grow for large values (no hours)", () => {
  assert.equal(formatDuration(1e9), "16666666:40");
  assert.equal(formatDuration(4294967296), "71582788:16");          // 2^32
  assert.equal(formatDuration(9007199254740991), "150119987579016:31"); // 2^53-1
  assert.equal(formatDuration(1e15), "16666666666666:40");
});

// ---------- formatBytes ----------

test("formatBytes: toFixed(1) rounding at the display boundaries", () => {
  // "(n / 1024).toFixed(1) + ' KB'" applied literally: 972/1024 = 0.94921...
  // rounds to 0.9, 973/1024 = 0.95039... rounds to 1.0.
  assert.equal(formatBytes(972), "0.9 KB");
  assert.equal(formatBytes(973), "1.0 KB");
  assert.equal(formatBytes(1023.5), "1.0 KB");
  assert.equal(formatBytes(1048575.9), "1024.0 KB"); // still < 1048576: KB branch
  // MB branch: 1310719.5/1048576 = 1.24999953... rounds to 1.2;
  // 1310720/1048576 = exactly 1.25 and toFixed rounds it to 1.3.
  assert.equal(formatBytes(1310719.5), "1.2 MB");
  assert.equal(formatBytes(1310720), "1.3 MB");
});

test("formatBytes: very large values stay in MB with one decimal", () => {
  assert.equal(formatBytes(5 * 1024 * 1048576), "5120.0 MB");
  assert.equal(formatBytes(1.5e15), "1430511474.6 MB");
});

// ---------- heatColor ----------

test("heatColor: exact interpolation at one-third and two-thirds", () => {
  // t = 1/3 in the first segment (f = 2/3), t = 2/3 in the second (f = 1/3),
  // and just below / just above the amber midpoint.
  assert.equal(heatColor(1 / 3, 1), "rgb(174, 123, 27)");
  assert.equal(heatColor(2 / 3, 1), "rgb(255, 137, 37)");
  assert.equal(heatColor(0.49, 1), "rgb(250, 173, 32)");
  assert.equal(heatColor(0.51, 1), "rgb(255, 174, 32)");
});

test("heatColor: extreme finite values clamp t to the stops", () => {
  // Both arguments are finite and max > 0, so t = value/max; it is then
  // clamped to 0..1, so overflow/underflow land on a stop.
  assert.equal(heatColor(1e308, 1e-308), "rgb(255, 59, 48)");  // t overflows to +Inf -> 1
  assert.equal(heatColor(-1e308, 1), "rgb(13, 17, 16)");       // t < 0 -> 0
  assert.equal(heatColor(1e-308, 1e308), "rgb(13, 17, 16)");   // t underflows to ~0 -> 0
});

// ---------- normalizeGrid ----------

test("normalizeGrid: booleans and strings are not finite numbers, so they count as 0", () => {
  assert.deepEqual(normalizeGrid([[true, "5", 2]]), { max: 2, cells: [[0, 0, 1]] });
});

test("normalizeGrid: ragged rows keep their shape after scaling", () => {
  const r = normalizeGrid([[1], [2, 3]]);
  assert.equal(r.max, 3);
  assert.deepEqual(r.cells, [[1 / 3], [2 / 3, 1]]);
});

test("normalizeGrid: a 1000x1000 grid scales by the true maximum", () => {
  const grid = Array.from({ length: 1000 }, () => Array(1000).fill(0));
  grid[37][91] = 2500;
  grid[500][0] = 7;
  const r = normalizeGrid(grid);
  assert.equal(r.max, 2500);
  assert.equal(r.cells.length, 1000);
  assert.equal(r.cells[37][91], 1);
  assert.equal(r.cells[500][0], 7 / 2500);
  assert.equal(r.cells[0][0], 0);
  assert.equal(grid[37][91], 2500); // input untouched
});

// ---------- renderLibrary ----------

test("renderLibrary: the open link and delete button carry the fixed text", () => {
  const { c } = setup();
  renderLibrary(c, [{ id: "a1", name: "n", duration_s: 1, size_bytes: 1 }], {});
  const row = c.querySelector(".match-row");
  assert.equal(row.querySelector("a.open").textContent, "Open");
  assert.equal(row.querySelector("button.delete").textContent, "Delete");
});

test("renderLibrary: a null options object behaves like no options", () => {
  // "Both callbacks and the whole options object are optional".
  const { c, win } = setup();
  renderLibrary(c, [{ id: "x", name: "x", duration_s: 1, size_bytes: 1 }], null);
  const row = c.querySelector(".match-row");
  assert.ok(row);
  const ev = click(win, row.querySelector("a.open"));
  assert.equal(ev.defaultPrevented, false);
  assert.doesNotThrow(() => click(win, row.querySelector("button.delete")));
});

test("renderLibrary: a null id is stringified for the DOM and passed through as given", () => {
  const { c, win } = setup();
  const opened = [];
  renderLibrary(c, [{ id: null, name: "n", duration_s: 1, size_bytes: 1 }], { onOpen: (v) => opened.push(v) });
  const row = c.querySelector(".match-row");
  assert.equal(row.getAttribute("data-match-id"), "null"); // String(id), not encoded
  assert.equal(row.querySelector("a.open").getAttribute("href"), "/match.html?id=null");
  click(win, row.querySelector("a.open"));
  assert.deepEqual(opened, [null]); // "the id value as given in the data"
});

test("renderLibrary: only missing, null or empty names fall back to the id", () => {
  // ".name is the name as text. If the name is missing, null or empty, show
  // String(id) instead." 0 and " " are none of those.
  const { c } = setup();
  renderLibrary(c, [
    { id: "z1", name: 0, duration_s: 1, size_bytes: 1 },
    { id: "z2", name: " ", duration_s: 1, size_bytes: 1 },
  ], {});
  assert.deepEqual([...c.querySelectorAll(".name")].map((n) => n.textContent), ["0", " "]);
});

test("renderLibrary: re-rendering detaches the previous callbacks", () => {
  // Every call "replaces all existing children", so the first render's
  // listeners must not survive the second render.
  const { c, win } = setup();
  let a = 0, b = 0;
  renderLibrary(c, [{ id: "m", name: "m", duration_s: 1, size_bytes: 1 }], { onOpen: () => a++ });
  renderLibrary(c, [{ id: "m", name: "m", duration_s: 1, size_bytes: 1 }], { onOpen: () => b++ });
  click(win, c.querySelector("a.open"));
  assert.equal(a, 0);
  assert.equal(b, 1);
});

test("renderLibrary: all render functions build elements in the container's own document", () => {
  // "builds elements with container.ownerDocument".
  const other = new JSDOM("<!doctype html><body><div class='l'></div><div class='t'></div><div class='e'></div><div class='h'></div></body>", {
    virtualConsole: new VirtualConsole(),
  });
  const doc = other.window.document;
  const l = doc.querySelector(".l");
  const t = doc.querySelector(".t");
  const e = doc.querySelector(".e");
  const h = doc.querySelector(".h");
  renderLibrary(l, [{ id: "a", name: "n", duration_s: 1, size_bytes: 1 }], {});
  renderPlayerTable(t, [P({})], {});
  renderTeamCards(e, { left: { avg_centroid: [0, 0], avg_width: 1, avg_length: 1 } }, null);
  renderHeatmap(h, [[1]]);
  for (const cont of [l, t, e, h]) {
    assert.ok(cont.children.length >= 1);
    for (const child of cont.children) assert.equal(child.ownerDocument, doc);
  }
});

// ---------- renderPlayerTable ----------

test("renderPlayerTable: a null options object keeps the input order and marks no header", () => {
  const { c, win } = setup();
  renderPlayerTable(c, [P({ id: 3 }), P({ id: 1 })], null);
  assert.deepEqual(ids(c), ["3", "1"]);
  assert.equal(c.querySelectorAll("th.sorted").length, 0);
  assert.doesNotThrow(() => click(win, c.querySelector("th")));
});

test("renderPlayerTable: id renders as text, compares as numbers, and a missing id sorts last", () => {
  const noId = { team: "left", jersey: "5", role: "player", distance_m: 10, avg_speed_ms: 1, max_speed_ms: 2, sprints: 0 };
  const players = [P({ id: 3 }), noId, P({ id: 2.5 })];
  const { c } = setup();
  renderPlayerTable(c, players, { sortKey: "id" });
  assert.deepEqual(ids(c), ["2.5", "3", "undefined"]); // String(player.id)
  const cells = [...c.querySelectorAll('td[data-key="id"]')].map((t) => t.textContent);
  assert.deepEqual(cells, ["2.5", "3", DASH]); // not a finite number -> en dash
  renderPlayerTable(c, players, { sortKey: "id", descending: true });
  assert.deepEqual(ids(c), ["3", "2.5", "undefined"]); // missing last in both directions
});

test("renderPlayerTable: non-numeric string ids render the dash and sort last in input order", () => {
  const { c } = setup();
  const players = [
    P({ id: 1 }),
    { id: "a", team: "left", jersey: "1", role: "player", distance_m: 1, avg_speed_ms: 1, max_speed_ms: 1, sprints: 0 },
    { id: "b", team: "left", jersey: "2", role: "player", distance_m: 1, avg_speed_ms: 1, max_speed_ms: 1, sprints: 0 },
  ];
  renderPlayerTable(c, players, { sortKey: "id" });
  assert.deepEqual(ids(c), ["1", "a", "b"]); // both missing: keep input order among themselves
  const cells = [...c.querySelectorAll('td[data-key="id"]')].map((t) => t.textContent);
  assert.deepEqual(cells, ["1", DASH, DASH]);
});

test("renderPlayerTable: all-digit jerseys with leading zeros compare as numbers", () => {
  // "compare as numbers when both values are all digits": 11 (from "011")
  // is greater than 9, while string order would put "011" first.
  const { c } = setup();
  const players = [P({ id: 1, jersey: "011" }), P({ id: 2, jersey: "9" })];
  renderPlayerTable(c, players, { sortKey: "jersey" });
  assert.deepEqual(ids(c), ["2", "1"]);
  renderPlayerTable(c, players, { sortKey: "jersey", descending: true });
  assert.deepEqual(ids(c), ["1", "2"]);
});

test("renderPlayerTable: string columns use plain code-unit order", () => {
  // 'a' (0x61) < '\u00ef' (0xEF) and 'Z' (0x5A) < 'a' (0x61).
  const a = setup();
  renderPlayerTable(a.c, [P({ id: 1, team: "naze" }), P({ id: 2, team: "na\u00efve" })], { sortKey: "team" });
  assert.deepEqual(ids(a.c), ["1", "2"]);
  const b = setup();
  renderPlayerTable(b.c, [P({ id: 3, role: "a" }), P({ id: 4, role: "Z" })], { sortKey: "role" });
  assert.deepEqual(ids(b.c), ["4", "3"]);
});

test("renderPlayerTable: negative numeric values sort below zero", () => {
  const { c } = setup();
  const players = [P({ id: 1, distance_m: -1 }), P({ id: 2, distance_m: -5 }), P({ id: 3, distance_m: 3 })];
  renderPlayerTable(c, players, { sortKey: "distance_m" });
  assert.deepEqual(ids(c), ["2", "1", "3"]);
  renderPlayerTable(c, players, { sortKey: "distance_m", descending: true });
  assert.deepEqual(ids(c), ["3", "1", "2"]);
});

test("renderPlayerTable: onSort fires from every header even when no valid sortKey is active", () => {
  const { c, win } = setup();
  const seen = [];
  renderPlayerTable(c, [P({ id: 1 })], { sortKey: "nonsense", onSort: (k) => seen.push(k) });
  click(win, c.querySelector('th[data-key="team"]'));
  assert.deepEqual(seen, ["team"]);
  assert.equal(c.querySelectorAll("th.sorted").length, 0);
});

test("renderPlayerTable: re-rendering replaces the onSort handler", () => {
  const { c, win } = setup();
  let a = 0, b = 0;
  renderPlayerTable(c, [P({ id: 1 })], { onSort: () => a++ });
  renderPlayerTable(c, [P({ id: 1 })], { onSort: () => b++ });
  click(win, c.querySelector("th"));
  assert.equal(a, 0);
  assert.equal(b, 1);
});

test("renderPlayerTable: repeated renders with the same options are idempotent", () => {
  const { c } = setup();
  const players = [P({ id: 2, sprints: 3 }), P({ id: 1, sprints: 1 }), P({ id: 3, sprints: 3 })];
  renderPlayerTable(c, players, { sortKey: "sprints" });
  assert.deepEqual(ids(c), ["1", "2", "3"]);
  renderPlayerTable(c, players, { sortKey: "sprints" });
  assert.deepEqual(ids(c), ["1", "2", "3"]);
  assert.equal(c.children.length, 1);
  assert.equal(c.querySelectorAll("table.players").length, 1);
  assert.equal(c.querySelectorAll("th.sorted").length, 1);
});

// ---------- renderTeamCards ----------

test("renderTeamCards: an array without sides gives the single empty note", () => {
  // "If teams is null or has neither side: one child <p class='empty'>".
  const { c } = setup();
  renderTeamCards(c, [], null);
  assert.equal(c.children.length, 1);
  assert.equal(c.querySelector("p.empty").textContent, "No team data.");
  assert.equal(c.querySelector(".team-card"), null);
});

test("renderTeamCards: a lone right team renders exactly one Right card", () => {
  const { c } = setup();
  renderTeamCards(c, { right: { avg_centroid: [8, -0.5], avg_width: 29.5, avg_length: 41 } }, null);
  const cards = [...c.querySelectorAll(".team-card")];
  assert.equal(cards.length, 1);
  assert.equal(cards[0].getAttribute("data-team"), "right");
  assert.equal(cards[0].querySelector(".team-name").textContent, "Right");
  assert.equal(cards[0].querySelector(".centroid .value").textContent, "8.0, -0.5");
});

test("renderTeamCards: toFixed is applied literally to exact halves", () => {
  // (-10.25).toFixed(1) and (10.25).toFixed(1) are exact-half cases.
  const { c } = setup();
  renderTeamCards(c, { left: { avg_centroid: [-10.25, 10.25], avg_width: 1, avg_length: 1 } }, null);
  assert.equal(c.querySelector(".centroid .value").textContent, "-10.3, 10.3");
});

test("renderTeamCards: a null colour for one side leaves only that swatch uncoloured", () => {
  const { c } = setup();
  const teams = {
    left: { avg_centroid: [0, 0], avg_width: 1, avg_length: 1 },
    right: { avg_centroid: [0, 0], avg_width: 1, avg_length: 1 },
  };
  renderTeamCards(c, teams, { left: null, right: "#2050c0" });
  assert.equal(c.querySelector('[data-team="left"] .swatch').style.backgroundColor, "");
  assert.equal(c.querySelector('[data-team="right"] .swatch').style.backgroundColor, "rgb(32, 80, 192)");
});

test("renderTeamCards: non-numeric width and length show the en dash", () => {
  const { c } = setup();
  renderTeamCards(c, { left: { avg_centroid: [0, 0], avg_width: "31", avg_length: NaN } }, null);
  const card = c.querySelector(".team-card");
  assert.equal(card.querySelector(".centroid .value").textContent, "0.0, 0.0");
  assert.equal(card.querySelector(".width .value").textContent, DASH);
  assert.equal(card.querySelector(".length .value").textContent, DASH);
});

// ---------- renderHeatmap ----------

test("renderHeatmap: an all-zero grid is not empty and gets data-max '0'", () => {
  // "Empty grid (no rows, or first row empty, or not an array)" only; an
  // all-zero grid still gets a canvas, and data-max is String(max) with
  // max = 0 from normalizeGrid.
  const { c } = setup();
  renderHeatmap(c, [[0, 0], [0, 0]]);
  assert.equal(c.children.length, 1);
  const cv = c.querySelector("canvas.heatmap");
  assert.ok(cv);
  assert.equal(c.querySelector("p.empty"), null);
  assert.equal(cv.width, 40);
  assert.equal(cv.height, 40);
  assert.equal(cv.getAttribute("data-max"), "0");
});

test("renderHeatmap: a single-row grid is 20px per cell", () => {
  const { c } = setup();
  renderHeatmap(c, [[1, 2, 3]]);
  const cv = c.querySelector("canvas.heatmap");
  assert.equal(cv.width, 60);   // nx = 3
  assert.equal(cv.height, 20);  // ny = 1
  assert.equal(cv.getAttribute("data-max"), "3");
});

test("renderHeatmap: rows longer than the first do not grow the canvas", () => {
  // nx is the length of the FIRST row; ny is the number of rows. The max
  // still comes from every cell.
  const { c } = setup();
  renderHeatmap(c, [[1], [2, 3, 4]]);
  const cv = c.querySelector("canvas.heatmap");
  assert.equal(cv.width, 20);
  assert.equal(cv.height, 40);
  assert.equal(cv.getAttribute("data-max"), "4");
});
