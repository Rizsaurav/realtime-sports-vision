// Extra adversarial tests for the render library and pages task
// (tasks/ui_pages.md, implemented in src/sideline/web/lib/render.js).
//
// These target spec rules the existing suites (web_tests/render.test.mjs,
// web_tests_extra/improve_analytics_extra.test.mjs) do not check:
// non-finite and non-numeric inputs, literal toFixed/encodeURIComponent
// boundaries, non-array inputs to the render functions, numeric vs string
// sort semantics, code-unit string order, missing-value rules for numeric
// columns, repeated-render and cross-container state effects, and large
// inputs. Every assertion is required by the spec text in tasks/ui_pages.md.

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

// ---------- formatDuration ----------

test("formatDuration: non-finite and non-number values give 0:00", () => {
  // "A value that is not a finite number, or is negative, gives '0:00'":
  // strings, booleans, objects and infinities are not finite numbers.
  assert.equal(formatDuration("65"), "0:00");
  assert.equal(formatDuration(true), "0:00");
  assert.equal(formatDuration({}), "0:00");
  assert.equal(formatDuration(Infinity), "0:00");
  assert.equal(formatDuration(-Infinity), "0:00");
});

test("formatDuration: rounding boundaries use Math.round", () => {
  assert.equal(formatDuration(0.4), "0:00");      // rounds down to 0
  assert.equal(formatDuration(0.5), "0:01");      // Math.round(0.5) = 1
  assert.equal(formatDuration(59.5), "1:00");     // Math.round(59.5) = 60
  assert.equal(formatDuration(59), "0:59");       // seconds do not overflow
  assert.equal(formatDuration(119), "1:59");
  assert.equal(formatDuration(119.5), "2:00");    // Math.round(119.5) = 120
  assert.equal(formatDuration(60), "1:00");
});

test("formatDuration: no hours part, minutes just grow", () => {
  assert.equal(formatDuration(86400), "1440:00");
  assert.equal(formatDuration(86399.4), "1439:59"); // rounds to 86399 = 1439*60 + 59
  assert.equal(formatDuration(360000), "6000:00");
});

// ---------- formatBytes ----------

test("formatBytes: non-number values give 0.0 KB", () => {
  // "A value that is not a finite number, or is negative, gives '0.0 KB'".
  assert.equal(formatBytes(null), "0.0 KB");
  assert.equal(formatBytes("1024"), "0.0 KB");
  assert.equal(formatBytes(true), "0.0 KB");
  assert.equal(formatBytes(Infinity), "0.0 KB");
});

test("formatBytes: the KB/MB rule is applied literally at the boundary", () => {
  assert.equal(formatBytes(1023), "1.0 KB");          // 0.999023... toFixed(1) rounds up
  assert.equal(formatBytes(1048575.5), "1024.0 KB");  // still < 1048576: KB branch
  assert.equal(formatBytes(0.5), "0.0 KB");           // 0.000488 toFixed(1) rounds down
  assert.equal(formatBytes(1073741824), "1024.0 MB"); // large MB value
});

// ---------- heatColor ----------

test("heatColor: max <= 0, non-finite or non-number arguments give t = 0", () => {
  assert.equal(heatColor(5, -10), "rgb(13, 17, 16)");   // max not > 0
  assert.equal(heatColor(Infinity, Infinity), "rgb(13, 17, 16)"); // not finite
  assert.equal(heatColor(5, null), "rgb(13, 17, 16)");
  assert.equal(heatColor("5", "10"), "rgb(13, 17, 16)"); // not numbers
});

test("heatColor: quarter-point interpolation and both sides of the midpoint", () => {
  // t = 0.125: r = round(13 + 242*0.25) = 74, g = round(17 + 159*0.25) = 57, b = round(20) = 20
  assert.equal(heatColor(12.5, 100), "rgb(74, 57, 20)");
  // t = 0.875: g = round(176 - 117*0.75) = 88, b = round(32 + 16*0.75) = 44
  assert.equal(heatColor(87.5, 100), "rgb(255, 88, 44)");
  // Just below and just above t = 0.5 both round back to the amber stop.
  assert.equal(heatColor(4.99, 10), "rgb(255, 176, 32)");
  assert.equal(heatColor(5.01, 10), "rgb(255, 176, 32)");
});

// ---------- normalizeGrid ----------

test("normalizeGrid: all-invalid cells give max 0 and all-zero cells", () => {
  const r = normalizeGrid([[-1, NaN], [null, -5]]);
  assert.equal(r.max, 0);
  assert.deepEqual(r.cells, [[0, 0], [0, 0]]);
});

test("normalizeGrid: large and fractional maxima", () => {
  const big = normalizeGrid([[1e15, 2e15]]);
  assert.equal(big.max, 2e15);
  assert.deepEqual(big.cells, [[0.5, 1]]);
  const frac = normalizeGrid([[0.25, 0.75]]);
  assert.equal(frac.max, 0.75);
  assert.deepEqual(frac.cells, [[0.25 / 0.75, 1]]);
});

test("normalizeGrid: cells is a new array, not a view of the input", () => {
  const g = [[1, 2], [3, 4]];
  const r1 = normalizeGrid(g);
  r1.cells[0][0] = 99;
  assert.deepEqual(g, [[1, 2], [3, 4]]);
  const r2 = normalizeGrid(g);
  assert.deepEqual(r2.cells, [[0.25, 0.5], [0.75, 1]]);
});

// ---------- renderLibrary ----------

test("renderLibrary: non-array matches give the single empty note", () => {
  // "Empty (or not an array): the container gets exactly one child, <p class='empty'>".
  for (const bad of [null, undefined, "nope", 42, {}]) {
    const { c } = setup();
    renderLibrary(c, bad, {});
    assert.equal(c.children.length, 1, `matches = ${String(bad)}`);
    const p = c.querySelector("p.empty");
    assert.ok(p, `matches = ${String(bad)}`);
    assert.equal(p.textContent, "No matches yet. Upload a recording to get started.");
    assert.equal(c.querySelector(".match-row"), null);
  }
});

test("renderLibrary: numeric ids are stringified for the DOM but passed through to callbacks", () => {
  const { c, win } = setup();
  const opened = [];
  const deleted = [];
  renderLibrary(c, [{ id: 123, name: "n", duration_s: 65, size_bytes: 1024 }], {
    onOpen: (id) => opened.push(id),
    onDelete: (id) => deleted.push(id),
  });
  const row = c.querySelector(".match-row");
  assert.equal(row.getAttribute("data-match-id"), "123"); // String(id), not encoded
  assert.equal(row.querySelector("a.open").getAttribute("href"), "/match.html?id=123");
  assert.equal(row.querySelector(".duration").textContent, "1:05");
  assert.equal(row.querySelector(".size").textContent, "1.0 KB");
  click(win, row.querySelector("a.open"));
  click(win, row.querySelector("button.delete"));
  assert.deepEqual(opened, [123]); // the id value as given in the data
  assert.deepEqual(deleted, [123]);
});

test("renderLibrary: name falls back to String(id) for null and stringifies other names", () => {
  const { c } = setup();
  renderLibrary(c, [
    { id: "q1", name: null, duration_s: 1, size_bytes: 1 },
    { id: "q2", name: 42, duration_s: 1, size_bytes: 1 },
  ], {});
  const names = [...c.querySelectorAll(".name")].map((n) => n.textContent);
  assert.deepEqual(names, ["q1", "42"]);
});

test("renderLibrary: href encodes reserved characters, data-match-id does not", () => {
  const { c } = setup();
  renderLibrary(c, [
    { id: "a/b?c=d", name: "x", duration_s: 1, size_bytes: 1 },
    { id: "fuß", name: "x", duration_s: 1, size_bytes: 1 },
  ], {});
  const [r1, r2] = [...c.querySelectorAll(".match-row")];
  assert.equal(r1.getAttribute("data-match-id"), "a/b?c=d");
  assert.equal(r1.querySelector("a.open").getAttribute("href"), "/match.html?id=a%2Fb%3Fc%3Dd");
  assert.equal(r2.getAttribute("data-match-id"), "fuß");
  assert.equal(r2.querySelector("a.open").getAttribute("href"), "/match.html?id=fu%C3%9F");
});

test("renderLibrary: missing duration_s and size_bytes format as zero", () => {
  const { c } = setup();
  renderLibrary(c, [{ id: "m", name: "no numbers" }], {});
  const row = c.querySelector(".match-row");
  assert.equal(row.querySelector(".duration").textContent, "0:00");
  assert.equal(row.querySelector(".size").textContent, "0.0 KB");
});

test("renderLibrary: non-function callbacks are ignored and never prevent the link", () => {
  const { c, win } = setup();
  renderLibrary(c, [{ id: "x", name: "x", duration_s: 1, size_bytes: 1 }], { onOpen: "yes", onDelete: 5 });
  const row = c.querySelector(".match-row");
  const evOpen = click(win, row.querySelector("a.open"));
  assert.equal(evOpen.defaultPrevented, false); // onOpen is not a function: nothing special
  assert.doesNotThrow(() => click(win, row.querySelector("button.delete")));
});

test("renderLibrary: callbacks are called with exactly one argument", () => {
  const { c, win } = setup();
  const arities = [];
  renderLibrary(c, [{ id: "z", name: "z", duration_s: 1, size_bytes: 1 }], {
    onOpen: (...a) => arities.push(["open", a.length, a[0]]),
    onDelete: (...a) => arities.push(["delete", a.length, a[0]]),
  });
  const row = c.querySelector(".match-row");
  click(win, row.querySelector("a.open"));
  click(win, row.querySelector("button.delete"));
  assert.deepEqual(arities, [["open", 1, "z"], ["delete", 1, "z"]]);
});

test("renderLibrary: 500 matches keep the given order", () => {
  const { c } = setup();
  const matches = Array.from({ length: 500 }, (_, i) => ({
    id: "id-" + String(i).padStart(3, "0"),
    name: "m" + i,
    duration_s: i,
    size_bytes: i,
  }));
  renderLibrary(c, matches, {});
  const rows = [...c.querySelectorAll(".match-row")];
  assert.equal(rows.length, 500);
  assert.equal(c.children.length, 500);
  assert.equal(rows[0].getAttribute("data-match-id"), "id-000");
  assert.equal(rows[499].getAttribute("data-match-id"), "id-499");
  assert.equal(rows[250].querySelector(".name").textContent, "m250");
});

// ---------- renderPlayerTable ----------

test("renderPlayerTable: id is compared as numbers, not strings", () => {
  const { c } = setup();
  const players = [P({ id: 10 }), P({ id: 2 }), P({ id: 9 })];
  renderPlayerTable(c, players, { sortKey: "id" });
  assert.deepEqual(ids(c), ["2", "9", "10"]);
  renderPlayerTable(c, players, { sortKey: "id", descending: true });
  assert.deepEqual(ids(c), ["10", "9", "2"]);
});

test("renderPlayerTable: missing numeric values sort last, keeping input order among themselves", () => {
  const { c } = setup();
  const players = [
    P({ id: "a", distance_m: 9 }),
    P({ id: "b", distance_m: null }),
    P({ id: "c", distance_m: 1 }),
    P({ id: "d", distance_m: undefined }),
  ];
  renderPlayerTable(c, players, { sortKey: "distance_m" });
  assert.deepEqual(ids(c), ["c", "a", "b", "d"]);
  renderPlayerTable(c, players, { sortKey: "distance_m", descending: true });
  assert.deepEqual(ids(c), ["a", "c", "b", "d"]);
});

test("renderPlayerTable: empty strings and non-finite numbers render the en dash", () => {
  const { c } = setup();
  renderPlayerTable(c, [
    P({ id: 1, team: "", role: "" }),
    P({ id: 2, avg_speed_ms: Infinity, max_speed_ms: NaN, sprints: "3" }),
    { id: 3 },
  ], {});
  const rows = c.querySelectorAll("tbody tr");
  const cell = (tr, key) => tr.querySelector(`td[data-key="${key}"]`).textContent;
  assert.equal(cell(rows[0], "team"), DASH);   // empty string
  assert.equal(cell(rows[0], "role"), DASH);   // empty string
  assert.equal(cell(rows[1], "avg_speed_ms"), DASH);  // not a finite number
  assert.equal(cell(rows[1], "max_speed_ms"), DASH);  // NaN
  assert.equal(cell(rows[1], "sprints"), DASH);       // string is not a finite number
  // A bare { id } row: every other cell is the en dash.
  assert.equal(cell(rows[2], "team"), DASH);
  assert.equal(cell(rows[2], "jersey"), DASH);
  assert.equal(cell(rows[2], "id"), "3");
  assert.equal(cell(rows[2], "role"), DASH);
  assert.equal(cell(rows[2], "distance_m"), DASH);
  assert.equal(cell(rows[2], "avg_speed_ms"), DASH);
  assert.equal(cell(rows[2], "max_speed_ms"), DASH);
  assert.equal(cell(rows[2], "sprints"), DASH);
});

test("renderPlayerTable: toFixed is applied exactly at the half values", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ distance_m: 99.5, avg_speed_ms: 3.75, max_speed_ms: 0.25 })], {});
  const row = c.querySelector("tbody tr");
  assert.equal(row.querySelector('td[data-key="distance_m"]').textContent, "100");   // (99.5).toFixed(0)
  assert.equal(row.querySelector('td[data-key="avg_speed_ms"]').textContent, "3.8");  // (3.75).toFixed(1)
  assert.equal(row.querySelector('td[data-key="max_speed_ms"]').textContent, "0.3");  // (0.25).toFixed(1)
});

test("renderPlayerTable: string columns compare in plain code-unit order", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 1, team: "left" }), P({ id: 2, team: "Left" })], { sortKey: "team" });
  assert.deepEqual(ids(c), ["2", "1"]); // 'L' (0x4C) < 'l' (0x6C)
  renderPlayerTable(c, [P({ id: 1, role: "zz" }), P({ id: 2, role: "AA" }), P({ id: 3, role: "Aa" })], { sortKey: "role" });
  assert.deepEqual(ids(c), ["2", "3", "1"]); // "AA" < "Aa" < "zz"
});

test("renderPlayerTable: jersey mixes numeric and string comparison", () => {
  const { c } = setup();
  const players = [P({ id: 1, jersey: "9" }), P({ id: 2, jersey: "10" }), P({ id: 3, jersey: "a1" }), P({ id: 4, jersey: "2" })];
  renderPlayerTable(c, players, { sortKey: "jersey" });
  // Digits order numerically (2 < 9 < 10); "a1" is not all digits, so it
  // compares as a string against each digit jersey and sorts after them.
  assert.deepEqual(ids(c), ["4", "1", "2", "3"]);
  renderPlayerTable(c, players, { sortKey: "jersey", descending: true });
  assert.deepEqual(ids(c), ["3", "2", "1", "4"]);
});

test("renderPlayerTable: re-rendering without a sort key clears the header classes", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 1 }), P({ id: 2 })], { sortKey: "sprints", descending: true });
  assert.equal(c.querySelectorAll("th.sorted").length, 1);
  assert.equal(c.querySelectorAll("th.desc").length, 1);
  renderPlayerTable(c, [P({ id: 1 }), P({ id: 2 })], {});
  assert.equal(c.querySelectorAll("th.sorted").length, 0);
  assert.equal(c.querySelectorAll("th.desc").length, 0);
});

test("renderPlayerTable: clicking every header reports the keys in column order", () => {
  const { c, win } = setup();
  const seen = [];
  renderPlayerTable(c, [P({})], { onSort: (k) => seen.push(k) });
  for (const th of c.querySelectorAll("thead th")) click(win, th);
  assert.deepEqual(seen, ["team", "jersey", "id", "role", "distance_m", "avg_speed_ms", "max_speed_ms", "sprints"]);
});

test("renderPlayerTable: 1000 distinct values sort correctly and the input is untouched", () => {
  const { c } = setup();
  const players = Array.from({ length: 1000 }, (_, i) => P({ id: i, distance_m: (i * 37) % 1000 }));
  const values = players.map((p) => p.distance_m); // a permutation of 0..999 (37 coprime to 1000)
  const asc = values.map((v, i) => i).sort((a, b) => values[a] - values[b]);
  renderPlayerTable(c, players, { sortKey: "distance_m" });
  assert.deepEqual(ids(c).map(Number), asc);
  renderPlayerTable(c, players, { sortKey: "distance_m", descending: true });
  assert.deepEqual(ids(c).map(Number), [...asc].reverse());
  assert.deepEqual(players.map((p) => p.id), Array.from({ length: 1000 }, (_, i) => i)); // input unmodified
});

test("renderPlayerTable: separate containers do not share sort state", () => {
  const a = setup();
  const b = setup();
  renderPlayerTable(a.c, [P({ id: 3 }), P({ id: 1 }), P({ id: 2 })], { sortKey: "id" });
  renderPlayerTable(b.c, [P({ id: 7 }), P({ id: 5 })], {});
  assert.deepEqual(ids(a.c), ["1", "2", "3"]);
  assert.equal(a.c.querySelectorAll("th.sorted").length, 1);
  assert.deepEqual(ids(b.c), ["7", "5"]);
  assert.equal(b.c.querySelectorAll("th.sorted").length, 0);
});

test("renderPlayerTable: player values are inserted as text, never as HTML", () => {
  const { c } = setup();
  const evil = "<img src=x onerror=alert(1)><b>x</b>";
  renderPlayerTable(c, [P({ id: 1, team: evil, role: "a<b" })], {});
  assert.equal(c.querySelector("img"), null);
  assert.equal(c.querySelector("b"), null);
  assert.equal(c.querySelector('td[data-key="team"]').textContent, evil);
  assert.equal(c.querySelector('td[data-key="role"]').textContent, "a<b");
});

// ---------- renderTeamCards ----------

test("renderTeamCards: a colour missing for one side leaves that swatch uncoloured", () => {
  const { c } = setup();
  const teams = {
    left: { avg_centroid: [0, 0], avg_width: 1, avg_length: 1 },
    right: { avg_centroid: [0, 0], avg_width: 1, avg_length: 1 },
  };
  renderTeamCards(c, teams, { left: "#d03030" });
  // The left swatch carries the colour (jsdom serializes the hex as rgb);
  // the right swatch exists but has no colour set.
  assert.equal(c.querySelector('[data-team="left"] .swatch').style.backgroundColor, "rgb(208, 48, 48)");
  assert.equal(c.querySelector('[data-team="right"] .swatch').style.backgroundColor, "");
  assert.equal(c.querySelectorAll(".team-card").length, 2);
});

test("renderTeamCards: undefined teams give the single empty note", () => {
  const { c } = setup();
  renderTeamCards(c, undefined, null);
  assert.equal(c.children.length, 1);
  assert.equal(c.querySelector("p.empty").textContent, "No team data.");
});

test("renderTeamCards: malformed centroid values give the en dash", () => {
  const { c } = setup();
  const teams = {
    left: { avg_centroid: [5], avg_width: 1, avg_length: 1 },
    right: { avg_centroid: [1, "x"], avg_width: 1, avg_length: 1 },
  };
  renderTeamCards(c, teams, null);
  assert.equal(c.querySelector('[data-team="left"] .centroid .value').textContent, DASH);
  assert.equal(c.querySelector('[data-team="right"] .centroid .value').textContent, DASH);
  const one = setup();
  renderTeamCards(one.c, { left: { avg_centroid: [NaN, 2], avg_width: 1, avg_length: 1 } }, null);
  assert.equal(one.c.querySelector(".centroid .value").textContent, DASH);
});

test("renderTeamCards: toFixed is applied literally to zero, negative and half values", () => {
  const { c } = setup();
  const teams = {
    left: { avg_centroid: [0, 0], avg_width: 31.25, avg_length: 0 },
    right: { avg_centroid: [-1000000.26, 99999.9375], avg_width: -5, avg_length: 38.44 },
  };
  renderTeamCards(c, teams, null);
  const left = c.querySelector('[data-team="left"]');
  assert.equal(left.querySelector(".centroid .value").textContent, "0.0, 0.0");
  assert.equal(left.querySelector(".width .value").textContent, "31.3 m"); // (31.25).toFixed(1)
  assert.equal(left.querySelector(".length .value").textContent, "0.0 m");
  const right = c.querySelector('[data-team="right"]');
  assert.equal(right.querySelector(".centroid .value").textContent, "-1000000.3, 99999.9");
  assert.equal(right.querySelector(".width .value").textContent, "-5.0 m");
  assert.equal(right.querySelector(".length .value").textContent, "38.4 m");
});

// ---------- renderHeatmap ----------

test("renderHeatmap: non-array grids give the single empty note", () => {
  for (const bad of [null, undefined, 42, "grid"]) {
    const { c } = setup();
    renderHeatmap(c, bad);
    assert.equal(c.children.length, 1, `grid = ${String(bad)}`);
    assert.equal(c.querySelector("canvas"), null);
    assert.equal(c.querySelector("p.empty").textContent, "No heatmap data.");
  }
});

test("renderHeatmap: an empty first row gives the empty note even if later rows are full", () => {
  const { c } = setup();
  renderHeatmap(c, [[], [1, 2]]);
  assert.equal(c.children.length, 1);
  assert.equal(c.querySelector("canvas"), null);
  assert.equal(c.querySelector("p.empty").textContent, "No heatmap data.");
});

test("renderHeatmap: data-max is String(max) from normalizeGrid, including floats and invalid cells", () => {
  const a = setup();
  renderHeatmap(a.c, [[0, 2.5]]);
  assert.equal(a.c.querySelector("canvas.heatmap").getAttribute("data-max"), "2.5");
  const b = setup();
  renderHeatmap(b.c, [[-5, NaN]]); // both cells count as 0, but the grid is not empty
  const cv = b.c.querySelector("canvas.heatmap");
  assert.ok(cv, "a grid with invalid cells still gets a canvas");
  assert.equal(cv.getAttribute("data-max"), "0");
  assert.equal(b.c.querySelector("p.empty"), null);
  assert.equal(cv.width, 40);
  assert.equal(cv.height, 20);
});

test("renderHeatmap: ragged rows and single-cell grids size from the first row", () => {
  const a = setup();
  renderHeatmap(a.c, [[1, 2], [3]]); // nx = 2 (first row), ny = 2 rows, max 3
  const cvA = a.c.querySelector("canvas.heatmap");
  assert.equal(cvA.width, 40);
  assert.equal(cvA.height, 40);
  assert.equal(cvA.getAttribute("data-max"), "3");
  const b = setup();
  renderHeatmap(b.c, [[7]]);
  const cvB = b.c.querySelector("canvas.heatmap");
  assert.equal(cvB.width, 20);
  assert.equal(cvB.height, 20);
  assert.equal(cvB.getAttribute("data-max"), "7");
});

test("renderHeatmap: a large 50x40 grid", () => {
  const { c } = setup();
  const grid = Array.from({ length: 40 }, () => Array(50).fill(0));
  grid[13][42] = 999;
  renderHeatmap(c, grid);
  const cv = c.querySelector("canvas.heatmap");
  assert.equal(c.children.length, 1);
  assert.equal(cv.width, 1000);
  assert.equal(cv.height, 800);
  assert.equal(cv.getAttribute("data-max"), "999");
});

test("renderHeatmap: re-rendering replaces the empty note with a canvas and vice versa", () => {
  const { c } = setup();
  renderHeatmap(c, null);
  assert.ok(c.querySelector("p.empty"));
  renderHeatmap(c, [[1]]);
  assert.equal(c.children.length, 1);
  assert.equal(c.querySelectorAll("canvas").length, 1);
  assert.equal(c.querySelector("p.empty"), null);
  renderHeatmap(c, []);
  assert.equal(c.children.length, 1);
  assert.equal(c.querySelector("canvas"), null);
  assert.ok(c.querySelector("p.empty"));
});
