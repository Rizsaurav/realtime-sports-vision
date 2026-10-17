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

const DASH = "–";

function setup() {
  const dom = new JSDOM("<!doctype html><body><div id='c'></div></body>", { virtualConsole: new VirtualConsole() });
  const doc = dom.window.document;
  return { dom, doc, c: doc.getElementById("c"), win: dom.window };
}
function click(win, node) {
  const ev = new win.MouseEvent("click", { bubbles: true, cancelable: true });
  node.dispatchEvent(ev);
  return ev;
}
const MATCHES = [
  { id: "a1", name: "First half", created_at: "2026-01-01T10:00:00Z", frame_count: 1500, duration_s: 60, size_bytes: 2048 },
  { id: "b2", name: "Second half", created_at: "2026-01-02T10:00:00Z", frame_count: 3000, duration_s: 125.4, size_bytes: 5 * 1048576 },
];
const P = (o) => ({ id: 1, role: "player", team: "left", jersey: "10", distance_m: 100, avg_speed_ms: 3, max_speed_ms: 6, sprints: 1, ...o });

// ---- exports ----
test("all documented functions are exported", () => {
  for (const f of [formatDuration, formatBytes, heatColor, normalizeGrid, renderLibrary, renderPlayerTable, renderTeamCards, renderHeatmap]) {
    assert.equal(typeof f, "function");
  }
});

// ---- formatDuration ----
test("formatDuration basic values", () => {
  assert.equal(formatDuration(0), "0:00");
  assert.equal(formatDuration(5), "0:05");
  assert.equal(formatDuration(65), "1:05");
  assert.equal(formatDuration(600), "10:00");
});
test("formatDuration rounds to the nearest second", () => {
  assert.equal(formatDuration(59.4), "0:59");
  assert.equal(formatDuration(59.6), "1:00");
  assert.equal(formatDuration(125.4), "2:05");
});
test("formatDuration has no hours part", () => {
  assert.equal(formatDuration(3600), "60:00");
  assert.equal(formatDuration(5405), "90:05");
});
test("formatDuration invalid input gives 0:00", () => {
  assert.equal(formatDuration(-5), "0:00");
  assert.equal(formatDuration(NaN), "0:00");
  assert.equal(formatDuration(undefined), "0:00");
  assert.equal(formatDuration(null), "0:00");
});

// ---- formatBytes ----
test("formatBytes small values are shown in KB", () => {
  assert.equal(formatBytes(0), "0.0 KB");
  assert.equal(formatBytes(512), "0.5 KB");
  assert.equal(formatBytes(1024), "1.0 KB");
  assert.equal(formatBytes(1536), "1.5 KB");
});
test("formatBytes switches to MB at 1048576", () => {
  assert.equal(formatBytes(1048576), "1.0 MB");
  assert.equal(formatBytes(5 * 1048576), "5.0 MB");
  assert.equal(formatBytes(1572864), "1.5 MB");
  assert.equal(formatBytes(1048575), "1024.0 KB");
});
test("formatBytes invalid input gives 0.0 KB", () => {
  assert.equal(formatBytes(-1), "0.0 KB");
  assert.equal(formatBytes(NaN), "0.0 KB");
  assert.equal(formatBytes(undefined), "0.0 KB");
});

// ---- heatColor ----
test("heatColor stops", () => {
  assert.equal(heatColor(0, 10), "rgb(13, 17, 16)");
  assert.equal(heatColor(5, 10), "rgb(255, 176, 32)");
  assert.equal(heatColor(10, 10), "rgb(255, 59, 48)");
});
test("heatColor interpolates between stops", () => {
  assert.equal(heatColor(25, 100), "rgb(134, 97, 24)");
  assert.equal(heatColor(75, 100), "rgb(255, 118, 40)");
});
test("heatColor clamps and handles zero max", () => {
  assert.equal(heatColor(50, 10), "rgb(255, 59, 48)");
  assert.equal(heatColor(-3, 10), "rgb(13, 17, 16)");
  assert.equal(heatColor(4, 0), "rgb(13, 17, 16)");
  assert.equal(heatColor(NaN, 10), "rgb(13, 17, 16)");
});

// ---- normalizeGrid ----
test("normalizeGrid scales by the maximum", () => {
  const r = normalizeGrid([[0, 5], [10, 2.5]]);
  assert.equal(r.max, 10);
  assert.deepEqual(r.cells, [[0, 0.5], [1, 0.25]]);
});
test("normalizeGrid of all zeros", () => {
  const r = normalizeGrid([[0, 0], [0, 0]]);
  assert.equal(r.max, 0);
  assert.deepEqual(r.cells, [[0, 0], [0, 0]]);
});
test("normalizeGrid empty grid and invalid cells", () => {
  assert.deepEqual(normalizeGrid([]), { max: 0, cells: [] });
  const r = normalizeGrid([[4, -2, NaN, null]]);
  assert.equal(r.max, 4);
  assert.deepEqual(r.cells, [[1, 0, 0, 0]]);
});
test("normalizeGrid does not mutate its input", () => {
  const g = [[1, 2], [3, 4]];
  normalizeGrid(g);
  assert.deepEqual(g, [[1, 2], [3, 4]]);
});

// ---- renderLibrary ----
test("renderLibrary renders one row per match in order", () => {
  const { c } = setup();
  renderLibrary(c, MATCHES, {});
  const rows = [...c.querySelectorAll(".match-row")];
  assert.equal(rows.length, 2);
  assert.deepEqual(rows.map((r) => r.getAttribute("data-match-id")), ["a1", "b2"]);
  assert.equal(c.querySelector("p.empty"), null);
});
test("renderLibrary row content", () => {
  const { c } = setup();
  renderLibrary(c, MATCHES, {});
  const [r1, r2] = c.querySelectorAll(".match-row");
  assert.equal(r1.querySelector(".name").textContent, "First half");
  assert.equal(r1.querySelector(".duration").textContent, "1:00");
  assert.equal(r1.querySelector(".size").textContent, "2.0 KB");
  assert.equal(r2.querySelector(".duration").textContent, "2:05");
  assert.equal(r2.querySelector(".size").textContent, "5.0 MB");
});
test("renderLibrary open link href and delete button", () => {
  const { c } = setup();
  renderLibrary(c, MATCHES, {});
  const row = c.querySelector('[data-match-id="b2"]');
  assert.equal(row.querySelector("a.open").getAttribute("href"), "/match.html?id=b2");
  assert.equal(row.querySelector("button.delete").tagName, "BUTTON");
});
test("renderLibrary url-encodes the id in the link but not in data-match-id", () => {
  const { c } = setup();
  renderLibrary(c, [{ id: "a b&c", name: "x", duration_s: 1, size_bytes: 1 }], {});
  const row = c.querySelector(".match-row");
  assert.equal(row.getAttribute("data-match-id"), "a b&c");
  assert.equal(row.querySelector("a.open").getAttribute("href"), "/match.html?id=a%20b%26c");
});
test("renderLibrary open click calls onOpen and prevents default", () => {
  const { c, win } = setup();
  const seen = [];
  renderLibrary(c, MATCHES, { onOpen: (id) => seen.push(id) });
  const ev = click(win, c.querySelector('[data-match-id="b2"] a.open'));
  assert.deepEqual(seen, ["b2"]);
  assert.equal(ev.defaultPrevented, true);
});
test("renderLibrary open click without onOpen is not prevented", () => {
  const { c, win } = setup();
  renderLibrary(c, MATCHES, {});
  let prevented = null;
  c.addEventListener("click", (e) => { prevented = e.defaultPrevented; e.preventDefault(); });
  click(win, c.querySelector("a.open"));
  assert.equal(prevented, false);
});
test("renderLibrary delete click calls onDelete with the id only", () => {
  const { c, win } = setup();
  const calls = [];
  renderLibrary(c, MATCHES, { onDelete: (...a) => calls.push(a) });
  click(win, c.querySelector('[data-match-id="a1"] button.delete'));
  assert.deepEqual(calls, [["a1"]]);
});
test("renderLibrary delete click does not call onOpen", () => {
  const { c, win } = setup();
  let opened = 0;
  renderLibrary(c, MATCHES, { onOpen: () => { opened++; }, onDelete: () => {} });
  click(win, c.querySelector("button.delete"));
  assert.equal(opened, 0);
});
test("renderLibrary empty list shows a single p.empty with the fixed message", () => {
  const { c } = setup();
  renderLibrary(c, [], {});
  assert.equal(c.children.length, 1);
  const p = c.querySelector("p.empty");
  assert.equal(p.textContent, "No matches yet. Upload a recording to get started.");
  assert.equal(c.querySelector(".match-row"), null);
});
test("renderLibrary inserts names as text, never as HTML", () => {
  const { c } = setup();
  const evil = "<img src=x onerror=alert(1)><b>bold</b>";
  renderLibrary(c, [{ id: "z", name: evil, duration_s: 1, size_bytes: 1 }], {});
  assert.equal(c.querySelector("img"), null);
  assert.equal(c.querySelector("b"), null);
  assert.equal(c.querySelector(".name").textContent, evil);
});
test("renderLibrary falls back to the id when the name is missing or empty", () => {
  const { c } = setup();
  renderLibrary(c, [{ id: "q9", name: "", duration_s: 1, size_bytes: 1 }, { id: "q8", duration_s: 1, size_bytes: 1 }], {});
  assert.deepEqual([...c.querySelectorAll(".name")].map((n) => n.textContent), ["q9", "q8"]);
});
test("renderLibrary replaces previous children", () => {
  const { c } = setup();
  c.appendChild(c.ownerDocument.createElement("hr"));
  renderLibrary(c, MATCHES, {});
  renderLibrary(c, [MATCHES[0]], {});
  assert.equal(c.querySelector("hr"), null);
  assert.equal(c.querySelectorAll(".match-row").length, 1);
  renderLibrary(c, [], {});
  assert.equal(c.querySelectorAll(".match-row").length, 0);
  assert.equal(c.children.length, 1);
});
test("renderLibrary works without an options argument", () => {
  const { c, win } = setup();
  renderLibrary(c, MATCHES);
  assert.equal(c.querySelectorAll(".match-row").length, 2);
  click(win, c.querySelector("button.delete"));
});

// ---- renderPlayerTable ----
const KEYS = ["team", "jersey", "id", "role", "distance_m", "avg_speed_ms", "max_speed_ms", "sprints"];
const colText = (c, key) => [...c.querySelectorAll(`tbody tr td[data-key="${key}"]`)].map((t) => t.textContent);
const ids = (c) => [...c.querySelectorAll("tbody tr")].map((r) => r.getAttribute("data-player-id"));

test("renderPlayerTable header cells in order", () => {
  const { c } = setup();
  renderPlayerTable(c, [], {});
  const table = c.querySelector("table.players");
  assert.ok(table);
  assert.equal(c.children.length, 1);
  assert.deepEqual([...table.querySelectorAll("thead th")].map((t) => t.getAttribute("data-key")), KEYS);
  assert.deepEqual([...table.querySelectorAll("thead th")].map((t) => t.textContent),
    ["Team", "Jersey", "ID", "Role", "Distance (m)", "Avg speed (m/s)", "Max speed (m/s)", "Sprints"]);
});
test("renderPlayerTable empty list has a header and no rows", () => {
  const { c } = setup();
  renderPlayerTable(c, [], {});
  assert.equal(c.querySelectorAll("tbody tr").length, 0);
  assert.equal(c.querySelectorAll("th").length, 8);
});
test("renderPlayerTable one row per player with eight cells", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 7 }), P({ id: 8 })], {});
  const rows = c.querySelectorAll("tbody tr");
  assert.equal(rows.length, 2);
  assert.equal(rows[0].getAttribute("data-player-id"), "7");
  assert.deepEqual([...rows[0].querySelectorAll("td")].map((t) => t.getAttribute("data-key")), KEYS);
});
test("renderPlayerTable number formats", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 7, jersey: "10", distance_m: 105.2, avg_speed_ms: 3.76, max_speed_ms: 7.9, sprints: 2 })], {});
  const row = c.querySelector("tbody tr");
  const get = (k) => row.querySelector(`td[data-key="${k}"]`).textContent;
  assert.equal(get("team"), "left");
  assert.equal(get("jersey"), "10");
  assert.equal(get("id"), "7");
  assert.equal(get("role"), "player");
  assert.equal(get("distance_m"), "105");
  assert.equal(get("avg_speed_ms"), "3.8");
  assert.equal(get("max_speed_ms"), "7.9");
  assert.equal(get("sprints"), "2");
});
test("renderPlayerTable zero values keep their formats", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ distance_m: 0, avg_speed_ms: 0, max_speed_ms: 0, sprints: 0 })], {});
  assert.deepEqual(
    ["distance_m", "avg_speed_ms", "max_speed_ms", "sprints"].map((k) => colText(c, k)[0]),
    ["0", "0.0", "0.0", "0"]);
});
test("renderPlayerTable missing team and jersey show an en dash", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 1, team: null, jersey: null, role: "referee" })], {});
  assert.deepEqual(colText(c, "team"), [DASH]);
  assert.deepEqual(colText(c, "jersey"), [DASH]);
  assert.deepEqual(colText(c, "role"), ["referee"]);
});
test("renderPlayerTable without sortKey keeps input order and marks no header", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 3 }), P({ id: 1 }), P({ id: 2 })], {});
  assert.deepEqual(ids(c), ["3", "1", "2"]);
  assert.equal(c.querySelectorAll("th.sorted").length, 0);
  assert.equal(c.querySelectorAll("th.desc").length, 0);
});
test("renderPlayerTable sorts numbers ascending", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 1, distance_m: 300 }), P({ id: 2, distance_m: 90 }), P({ id: 3, distance_m: 1000 })], { sortKey: "distance_m" });
  assert.deepEqual(ids(c), ["2", "1", "3"]);
});
test("renderPlayerTable sorts numbers descending", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 1, max_speed_ms: 5 }), P({ id: 2, max_speed_ms: 9 }), P({ id: 3, max_speed_ms: 7 })], { sortKey: "max_speed_ms", descending: true });
  assert.deepEqual(ids(c), ["2", "3", "1"]);
});
test("renderPlayerTable header classes: sorted ascending", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({})], { sortKey: "sprints" });
  const sorted = c.querySelectorAll("th.sorted");
  assert.equal(sorted.length, 1);
  assert.equal(sorted[0].getAttribute("data-key"), "sprints");
  assert.equal(sorted[0].classList.contains("desc"), false);
});
test("renderPlayerTable header classes: sorted descending", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({})], { sortKey: "role", descending: true });
  const sorted = c.querySelectorAll("th.sorted");
  assert.equal(sorted.length, 1);
  assert.equal(sorted[0].getAttribute("data-key"), "role");
  assert.equal(sorted[0].classList.contains("desc"), true);
  assert.equal(c.querySelectorAll("th.desc").length, 1);
});
test("renderPlayerTable ties keep input order in both directions", () => {
  const { c } = setup();
  const players = [P({ id: 5, sprints: 2 }), P({ id: 6, sprints: 1 }), P({ id: 7, sprints: 2 }), P({ id: 8, sprints: 1 })];
  renderPlayerTable(c, players, { sortKey: "sprints" });
  assert.deepEqual(ids(c), ["6", "8", "5", "7"]);
  renderPlayerTable(c, players, { sortKey: "sprints", descending: true });
  assert.deepEqual(ids(c), ["5", "7", "6", "8"]);
});
test("renderPlayerTable sorts jersey numerically", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 1, jersey: "10" }), P({ id: 2, jersey: "9" }), P({ id: 3, jersey: "2" })], { sortKey: "jersey" });
  assert.deepEqual(ids(c), ["3", "2", "1"]);
});
test("renderPlayerTable sorts team and role as text", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 1, team: "right" }), P({ id: 2, team: "left" })], { sortKey: "team" });
  assert.deepEqual(ids(c), ["2", "1"]);
  renderPlayerTable(c, [P({ id: 1, role: "player" }), P({ id: 2, role: "goalkeeper" }), P({ id: 3, role: "referee" })], { sortKey: "role" });
  assert.deepEqual(ids(c), ["2", "1", "3"]);
});
test("renderPlayerTable missing values sort last in both directions", () => {
  const { c } = setup();
  const players = [P({ id: 1, jersey: null }), P({ id: 2, jersey: "7" }), P({ id: 3, jersey: null }), P({ id: 4, jersey: "3" })];
  renderPlayerTable(c, players, { sortKey: "jersey" });
  assert.deepEqual(ids(c), ["4", "2", "1", "3"]);
  renderPlayerTable(c, players, { sortKey: "jersey", descending: true });
  assert.deepEqual(ids(c), ["2", "4", "1", "3"]);
});
test("renderPlayerTable unknown sortKey leaves order and marks no header", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 3 }), P({ id: 1 })], { sortKey: "nonsense", descending: true });
  assert.deepEqual(ids(c), ["3", "1"]);
  assert.equal(c.querySelectorAll("th.sorted").length, 0);
});
test("renderPlayerTable does not mutate the players array", () => {
  const { c } = setup();
  const players = [P({ id: 3, sprints: 9 }), P({ id: 1, sprints: 1 })];
  renderPlayerTable(c, players, { sortKey: "sprints" });
  assert.deepEqual(players.map((p) => p.id), [3, 1]);
});
test("renderPlayerTable header click calls onSort with the key", () => {
  const { c, win } = setup();
  const seen = [];
  renderPlayerTable(c, [P({})], { onSort: (k) => seen.push(k) });
  click(win, c.querySelector('th[data-key="distance_m"]'));
  click(win, c.querySelector('th[data-key="team"]'));
  assert.deepEqual(seen, ["distance_m", "team"]);
});
test("renderPlayerTable header click without onSort does nothing", () => {
  const { c, win } = setup();
  renderPlayerTable(c, [P({})], {});
  click(win, c.querySelector("th"));
  assert.equal(c.querySelectorAll("tbody tr").length, 1);
});
test("renderPlayerTable replaces previous children", () => {
  const { c } = setup();
  renderPlayerTable(c, [P({ id: 1 }), P({ id: 2 })], {});
  renderPlayerTable(c, [P({ id: 3 })], {});
  assert.equal(c.querySelectorAll("table").length, 1);
  assert.deepEqual(ids(c), ["3"]);
});

// ---- renderTeamCards ----
const TEAMS = {
  left: { frames: 700, avg_centroid: [-10.24, 1.15], avg_width: 31, avg_length: 38.44 },
  right: { frames: 690, avg_centroid: [8, -0.5], avg_width: 29.55, avg_length: 41 },
};
const COLORS = { left: "rgb(208, 48, 48)", right: "rgb(32, 80, 192)" };

test("renderTeamCards renders left then right", () => {
  const { c } = setup();
  renderTeamCards(c, TEAMS, COLORS);
  const cards = [...c.querySelectorAll(".team-card")];
  assert.deepEqual(cards.map((x) => x.getAttribute("data-team")), ["left", "right"]);
  assert.equal(c.querySelector("p.empty"), null);
});
test("renderTeamCards renders left then right even if keys are in another order", () => {
  const { c } = setup();
  renderTeamCards(c, { right: TEAMS.right, left: TEAMS.left }, COLORS);
  assert.deepEqual([...c.querySelectorAll(".team-card")].map((x) => x.getAttribute("data-team")), ["left", "right"]);
});
test("renderTeamCards formats statistics with one decimal", () => {
  const { c } = setup();
  renderTeamCards(c, TEAMS, COLORS);
  const left = c.querySelector('.team-card[data-team="left"]');
  assert.equal(left.querySelector(".centroid .value").textContent, "-10.2, 1.1");
  assert.equal(left.querySelector(".width .value").textContent, "31.0 m");
  assert.equal(left.querySelector(".length .value").textContent, "38.4 m");
  const right = c.querySelector('.team-card[data-team="right"]');
  assert.equal(right.querySelector(".centroid .value").textContent, "8.0, -0.5");
  assert.equal(right.querySelector(".length .value").textContent, "41.0 m");
});
test("renderTeamCards labels and team name", () => {
  const { c } = setup();
  renderTeamCards(c, TEAMS, COLORS);
  const left = c.querySelector('.team-card[data-team="left"]');
  assert.equal(left.querySelector(".team-name").textContent, "Left");
  assert.equal(left.querySelector(".centroid .label").textContent, "Centroid");
  assert.equal(left.querySelector(".width .label").textContent, "Width");
  assert.equal(left.querySelector(".length .label").textContent, "Length");
  assert.equal(c.querySelector('[data-team="right"] .team-name').textContent, "Right");
});
test("renderTeamCards swatch carries the team colour inline", () => {
  const { c } = setup();
  renderTeamCards(c, TEAMS, COLORS);
  assert.equal(c.querySelector('[data-team="left"] .swatch').style.backgroundColor, "rgb(208, 48, 48)");
  assert.equal(c.querySelector('[data-team="right"] .swatch').style.backgroundColor, "rgb(32, 80, 192)");
});
test("renderTeamCards without colours still renders an uncoloured swatch", () => {
  const { c } = setup();
  renderTeamCards(c, TEAMS, null);
  const sw = c.querySelectorAll(".team-card .swatch");
  assert.equal(sw.length, 2);
  assert.equal(sw[0].style.backgroundColor, "");
});
test("renderTeamCards renders only the teams that exist", () => {
  const { c } = setup();
  renderTeamCards(c, { left: TEAMS.left }, COLORS);
  assert.equal(c.querySelectorAll(".team-card").length, 1);
});
test("renderTeamCards missing numbers show an en dash", () => {
  const { c } = setup();
  renderTeamCards(c, { left: { frames: 0, avg_centroid: null, avg_width: null, avg_length: null } }, COLORS);
  const card = c.querySelector(".team-card");
  assert.equal(card.querySelector(".centroid .value").textContent, DASH);
  assert.equal(card.querySelector(".width .value").textContent, DASH);
  assert.equal(card.querySelector(".length .value").textContent, DASH);
});
test("renderTeamCards with no teams shows p.empty", () => {
  const { c } = setup();
  renderTeamCards(c, {}, COLORS);
  assert.equal(c.children.length, 1);
  assert.equal(c.querySelector("p.empty").textContent, "No team data.");
  renderTeamCards(c, null, null);
  assert.equal(c.querySelector("p.empty").textContent, "No team data.");
});
test("renderTeamCards replaces previous children", () => {
  const { c } = setup();
  renderTeamCards(c, TEAMS, COLORS);
  renderTeamCards(c, { right: TEAMS.right }, COLORS);
  assert.equal(c.querySelectorAll(".team-card").length, 1);
});

// ---- renderHeatmap ----
test("renderHeatmap creates a canvas sized nx*20 by ny*20", () => {
  const { c } = setup();
  const grid = Array.from({ length: 14 }, () => Array(21).fill(0));
  renderHeatmap(c, grid);
  const cv = c.querySelector("canvas.heatmap");
  assert.ok(cv);
  assert.equal(c.children.length, 1);
  assert.equal(cv.width, 420);
  assert.equal(cv.height, 280);
});
test("renderHeatmap size for a small grid and data-max", () => {
  const { c } = setup();
  renderHeatmap(c, [[0, 3, 1], [7, 0, 2]]);
  const cv = c.querySelector("canvas.heatmap");
  assert.equal(cv.width, 60);
  assert.equal(cv.height, 40);
  assert.equal(cv.getAttribute("data-max"), "7");
});
test("renderHeatmap empty grid shows p.empty", () => {
  const { c } = setup();
  renderHeatmap(c, []);
  assert.equal(c.querySelector("canvas"), null);
  assert.equal(c.querySelector("p.empty").textContent, "No heatmap data.");
  renderHeatmap(c, [[]]);
  assert.equal(c.querySelector("p.empty").textContent, "No heatmap data.");
});
test("renderHeatmap replaces previous children", () => {
  const { c } = setup();
  renderHeatmap(c, [[1]]);
  renderHeatmap(c, [[1, 2]]);
  assert.equal(c.querySelectorAll("canvas").length, 1);
  assert.equal(c.querySelector("canvas").width, 40);
});
test("renderHeatmap does not throw when canvas drawing is unavailable", () => {
  const { c } = setup();
  assert.doesNotThrow(() => renderHeatmap(c, [[1, 2], [3, 4]]));
});
