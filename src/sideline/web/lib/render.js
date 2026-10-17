// Render helpers for the match library and match pages (see docs/WEBAPP_CONTRACT.md).
// ES module with named exports only. No dependencies, no build step, no top-level
// DOM access: every render function takes the container element first and builds
// all elements with container.ownerDocument. Every call replaces all existing
// children of the container. Data is inserted as text, never as HTML.

const DASH = "\u2013";

function emptyNote(doc, text) {
  const p = doc.createElement("p");
  p.className = "empty";
  p.textContent = text;
  return p;
}

/**
 * (seconds: number, whole seconds >= 0) -> 'm:ss' (minutes unpadded, seconds two-digit, no hours).
 * Non-finite, non-number or negative input -> '0:00'. Input unit: seconds.
 */
export function formatDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "0:00";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return minutes + ":" + String(secs).padStart(2, "0");
}

/**
 * (n: bytes, finite number >= 0) -> 'x.x KB' (n < 1048576) or 'x.x MB' (n >= 1048576), 1 KB = 1024 B.
 * Non-finite, non-number or negative input -> '0.0 KB'.
 */
export function formatBytes(n) {
  if (typeof n !== "number" || !Number.isFinite(n) || n < 0) return "0.0 KB";
  if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
  return (n / 1048576).toFixed(1) + " MB";
}

// Colour stops: t=0 dark pitch green, t=0.5 amber, t=1 tally red.
const HEAT_STOPS = [
  [0, 13, 17, 16],
  [0.5, 255, 176, 32],
  [1, 255, 59, 48],
];

/**
 * (value, max: dimensionless counts, both finite and max > 0) -> 'rgb(r, g, b)' colour string.
 * t = value/max clamped to 0..1 (else t = 0), linearly interpolated between the HEAT_STOPS below; channels are integers 0..255.
 */
export function heatColor(value, max) {
  let t = 0;
  if (
    typeof value === "number" && Number.isFinite(value) &&
    typeof max === "number" && Number.isFinite(max) && max > 0
  ) {
    t = value / max;
  }
  t = Math.min(1, Math.max(0, t));
  const seg = t <= 0.5 ? 0 : 1;
  const [t0, r0, g0, b0] = HEAT_STOPS[seg];
  const [t1, r1, g1, b1] = HEAT_STOPS[seg + 1];
  const f = (t - t0) / (t1 - t0);
  const r = Math.round(r0 + (r1 - r0) * f);
  const g = Math.round(g0 + (g1 - g0) * f);
  const b = Math.round(b0 + (b1 - b0) * f);
  return "rgb(" + r + ", " + g + ", " + b + ")";
}

/**
 * (container: element, matches: array of {id, name, created_at, frame_count, duration_s in seconds, size_bytes in bytes}, options {onOpen(id), onDelete(id)} all optional)
 * -> void; replaces all children of container with one .match-row per match in the given order (children fixed to .name, .duration, .size, a.open, button.delete), or one <p class="empty"> when empty/not an array.
 */
export function renderLibrary(container, matches, options = {}) {
  const doc = container.ownerDocument;
  container.replaceChildren();
  const { onOpen, onDelete } = options || {};
  if (!Array.isArray(matches) || matches.length === 0) {
    container.appendChild(emptyNote(doc, "No matches yet. Upload a recording to get started."));
    return;
  }
  for (const m of matches) {
    const row = doc.createElement("div");
    row.className = "match-row";
    row.setAttribute("data-match-id", String(m.id));

    const name = doc.createElement("span");
    name.className = "name";
    name.textContent = m.name !== null && m.name !== undefined && m.name !== "" ? String(m.name) : String(m.id);

    const duration = doc.createElement("span");
    duration.className = "duration";
    duration.textContent = formatDuration(m.duration_s);

    const size = doc.createElement("span");
    size.className = "size";
    size.textContent = formatBytes(m.size_bytes);

    const open = doc.createElement("a");
    open.className = "open";
    open.setAttribute("href", "/match.html?id=" + encodeURIComponent(String(m.id)));
    open.textContent = "Open";
    if (typeof onOpen === "function") {
      open.addEventListener("click", (event) => {
        event.preventDefault();
        onOpen(m.id);
      });
    }

    const del = doc.createElement("button");
    del.className = "delete";
    del.type = "button";
    del.textContent = "Delete";
    if (typeof onDelete === "function") {
      del.addEventListener("click", () => onDelete(m.id));
    }

    row.append(name, duration, size, open, del);
    container.appendChild(row);
  }
}

// Column order, header text and the en dash for missing values (U+2013).
const COLUMNS = [
  ["team", "Team"],
  ["jersey", "Jersey"],
  ["id", "ID"],
  ["role", "Role"],
  ["distance_m", "Distance (m)"],
  ["avg_speed_ms", "Avg speed (m/s)"],
  ["max_speed_ms", "Max speed (m/s)"],
  ["sprints", "Sprints"],
];
const NUMERIC_KEYS = new Set(["id", "distance_m", "avg_speed_ms", "max_speed_ms", "sprints"]);

function cellMissing(player, key) {
  const v = player ? player[key] : undefined;
  if (v === null || v === undefined || v === "") return true;
  if (typeof v === "number") return !Number.isFinite(v);
  return NUMERIC_KEYS.has(key);
}

function cellText(player, key) {
  if (cellMissing(player, key)) return DASH;
  const v = player[key];
  if (key === "distance_m") return v.toFixed(0);
  if (key === "avg_speed_ms" || key === "max_speed_ms") return v.toFixed(1);
  return String(v);
}

function compareValues(a, b, key) {
  if (NUMERIC_KEYS.has(key)) return a - b;
  if (key === "jersey") {
    if (/^\d+$/.test(a) && /^\d+$/.test(b)) return Number(a) - Number(b);
  }
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

function isSortKey(key) {
  return COLUMNS.some(([k]) => k === key);
}

/**
 * (doc: document, options {sortKey, descending, onSort}) -> <tr> with one <th> per column (data-key, sortable via click/Enter/Space when onSort is a function);
 * the active sort key's <th> gets class 'sorted' (plus 'desc' when descending); no other header gets either class.
 */
function buildHeaderRow(doc, { sortKey, descending, onSort }) {
  const headRow = doc.createElement("tr");
  for (const [key, label] of COLUMNS) {
    const th = doc.createElement("th");
    th.setAttribute("data-key", key);
    th.setAttribute("scope", "col");
    th.setAttribute("tabindex", "0");
    th.setAttribute("role", "button");
    th.textContent = label;
    if (isSortKey(sortKey) && key === sortKey) {
      th.classList.add("sorted");
      if (descending) th.classList.add("desc");
    }
    if (typeof onSort === "function") {
      th.addEventListener("click", () => onSort(key));
      // Keyboard equivalent of the click: Enter or Space sorts the column.
      th.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSort(key);
        }
      });
    }
    headRow.appendChild(th);
  }
  return headRow;
}

/**
 * (players: array (or anything else -> empty list), sortKey, descending: bool) -> a copy of the rows, sorted by sortKey when it is a valid column key; input is never modified.
 * Stable sort: ties keep input order in both directions; missing values always sort last, in input order, regardless of direction.
 */
function sortedRows(players, sortKey, descending) {
  const rows = Array.isArray(players) ? players.slice() : [];
  if (!isSortKey(sortKey)) return rows;
  const dir = descending ? -1 : 1;
  rows.sort((pa, pb) => {
    const ma = cellMissing(pa, sortKey);
    const mb = cellMissing(pb, sortKey);
    if (ma && mb) return 0;
    if (ma) return 1;
    if (mb) return -1;
    return compareValues(pa[sortKey], pb[sortKey], sortKey) * dir;
  });
  return rows;
}

/**
 * (doc: document, rows: sorted player objects) -> <tbody> with one <tr data-player-id> and eight <td data-key> per row, text via cellText.
 */
function buildBody(doc, rows) {
  const tbody = doc.createElement("tbody");
  for (const p of rows) {
    const tr = doc.createElement("tr");
    tr.setAttribute("data-player-id", String(p.id));
    for (const [key] of COLUMNS) {
      const td = doc.createElement("td");
      td.setAttribute("data-key", key);
      td.textContent = cellText(p, key);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  return tbody;
}

/**
 * (container: element, players: array of player objects from the analytics summary, options {sortKey: one of the 8 data-keys or null, descending: bool, onSort(key)})
 * -> void; replaces all children with one <table class="players"> (header via buildHeaderRow, body via buildBody over sortedRows); cell units: distance in metres (m), speed in metres per second (m/s), sprints a plain count.
 */
export function renderPlayerTable(container, players, options = {}) {
  const doc = container.ownerDocument;
  container.replaceChildren();
  const { sortKey = null, descending = false, onSort } = options || {};

  const table = doc.createElement("table");
  table.className = "players";
  const thead = doc.createElement("thead");
  thead.appendChild(buildHeaderRow(doc, { sortKey, descending, onSort }));
  table.appendChild(thead);
  table.appendChild(buildBody(doc, sortedRows(players, sortKey, descending)));
  container.appendChild(table);
}

/**
 * (grid: array of rows of numbers, rows are y and columns x, cell values are non-negative counts) -> {max: number, cells: same-shape array of each cell / max (all 0 when max is 0)}.
 * Non-finite or negative cells count as 0; max is 0 when there are no positive cells; input is not modified.
 */
export function normalizeGrid(grid) {
  if (!Array.isArray(grid)) return { max: 0, cells: [] };
  let max = 0;
  const clean = [];
  for (const row of grid) {
    const out = [];
    for (const cell of Array.isArray(row) ? row : []) {
      const v = typeof cell === "number" && Number.isFinite(cell) && cell >= 0 ? cell : 0;
      out.push(v);
      if (v > max) max = v;
    }
    clean.push(out);
  }
  const cells = clean.map((row) => row.map((v) => (max > 0 ? v / max : 0)));
  return { max, cells };
}

function centroidText(centroid) {
  if (!Array.isArray(centroid) || centroid.length < 2) return DASH;
  const x = centroid[0];
  const y = centroid[1];
  if (typeof x !== "number" || !Number.isFinite(x) || typeof y !== "number" || !Number.isFinite(y)) return DASH;
  return x.toFixed(1) + ", " + y.toFixed(1);
}

function metricText(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return DASH;
  return value.toFixed(1) + " m";
}

/**
 * (container: element, teams: summary object with optional {left, right} entries, colors: {left, right} colour strings or null) -> void;
 * replaces all children with one .team-card per present side in left, right order, or one <p class="empty">; value units: centroid x/y in metres (m), width and length in metres (m), shown with one decimal.
 */
export function renderTeamCards(container, teams, colors) {
  const doc = container.ownerDocument;
  container.replaceChildren();

  const sides = [];
  if (teams && typeof teams === "object") {
    if (teams.left !== null && teams.left !== undefined) sides.push("left");
    if (teams.right !== null && teams.right !== undefined) sides.push("right");
  }
  if (sides.length === 0) {
    container.appendChild(emptyNote(doc, "No team data."));
    return;
  }

  for (const side of sides) {
    const data = teams[side];
    const card = doc.createElement("div");
    card.className = "team-card";
    card.setAttribute("data-team", side);

    const swatch = doc.createElement("span");
    swatch.className = "swatch";
    if (colors && typeof colors === "object" && colors[side]) {
      swatch.style.backgroundColor = colors[side];
    }
    card.appendChild(swatch);

    const name = doc.createElement("h3");
    name.className = "team-name";
    name.textContent = side === "left" ? "Left" : "Right";
    card.appendChild(name);

    const stats = [
      ["centroid", "Centroid", centroidText(data ? data.avg_centroid : null)],
      ["width", "Width", metricText(data ? data.avg_width : null)],
      ["length", "Length", metricText(data ? data.avg_length : null)],
    ];
    for (const [cls, label, value] of stats) {
      const stat = doc.createElement("div");
      stat.className = "stat " + cls;
      const lab = doc.createElement("span");
      lab.className = "label";
      lab.textContent = label;
      const val = doc.createElement("span");
      val.className = "value";
      val.textContent = value;
      stat.append(lab, val);
      card.appendChild(stat);
    }
    container.appendChild(card);
  }
}

/**
 * (container: element, grid: one summary heatmap, array of ny rows of nx numbers, cell values are non-negative counts) -> void;
 * replaces all children with <canvas class="heatmap"> sized nx*20 by ny*20 px (data-max = String(max) from normalizeGrid) or one <p class="empty"> when the grid is empty, and draws each cell as a 20x20 px square.
 */
export function renderHeatmap(container, grid) {
  const doc = container.ownerDocument;
  container.replaceChildren();
  if (!Array.isArray(grid) || grid.length === 0 || !Array.isArray(grid[0]) || grid[0].length === 0) {
    container.appendChild(emptyNote(doc, "No heatmap data."));
    return;
  }
  const nx = grid[0].length;
  const ny = grid.length;
  const { max } = normalizeGrid(grid);

  const canvas = doc.createElement("canvas");
  canvas.className = "heatmap";
  canvas.width = nx * 20;
  canvas.height = ny * 20;
  canvas.setAttribute("data-max", String(max));
  container.appendChild(canvas);

  try {
    const ctx = canvas.getContext ? canvas.getContext("2d") : null;
    if (!ctx) return; // No canvas implementation (e.g. jsdom): skip drawing.
    for (let y = 0; y < ny; y++) {
      const row = Array.isArray(grid[y]) ? grid[y] : [];
      const n = Math.min(row.length, nx);
      for (let x = 0; x < n; x++) {
        const v = typeof row[x] === "number" && Number.isFinite(row[x]) && row[x] >= 0 ? row[x] : 0;
        ctx.fillStyle = heatColor(v, max);
        ctx.fillRect(x * 20, y * 20, 20, 20);
      }
    }
  } catch {
    // Drawing is best effort; a missing canvas implementation is not an error.
  }
}
