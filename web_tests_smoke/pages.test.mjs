// Smoke tests that run the real pages together with the real API responses.
//
// Loads src/sideline/web/library.html and match.html into jsdom and lets the
// pages run their own scripts. The pages' fetch is answered from the JSON
// fixtures in ./fixtures (recorded by `python tests_integration/make_fixtures.py`
// against a real create_library_app) and every request the pages make is
// recorded and must be one of those fixture answers.
//
// jsdom does not execute <script type="module">, so the page's module script
// is run as a classic script with each `import { ... } from "/static/..."`
// statement replaced by destructuring from a registry. The registry holds the
// real library modules, imported through the same /static/... -> file mapping
// the app uses to serve them (src/sideline/web is mounted at /static).

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { JSDOM, ResourceLoader, VirtualConsole } from "jsdom";

const here = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.resolve(here, "../src/sideline/web");
const FIXTURES_DIR = path.resolve(here, "fixtures");
// The origin the pages are loaded from; a page request is only in the
// fixtures if it is to this origin (the fixtures record the app's routes).
const APP_ORIGIN = "http://localhost";

const fixtures = {
  matches: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "matches.json"), "utf8")),
  emptyMatches: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "empty_matches.json"), "utf8")),
  match: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "match.json"), "utf8")),
  analytics: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "analytics.json"), "utf8")),
  frames: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "frames_0_249.json"), "utf8")),
  limits: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "limits.json"), "utf8")),
};
const MATCH_ID = fixtures.match.id;
const MATCH_NAME = fixtures.match.name;

// The real replay/render helpers, so the expectations use the same code the
// pages use (clock formatting, byte formatting).
const replay = await import(pathToFileURL(path.join(WEB_DIR, "lib/replay.js")).href);
const render = await import(pathToFileURL(path.join(WEB_DIR, "lib/render.js")).href);

// The whole match is fixtures.match.last_frame - first_frame + 1 frames at
// 25 fps; the clock shows "position / total" with the page's formatClock.
const TOTAL_S = (fixtures.match.last_frame - fixtures.match.first_frame) / 25;

// -- serving /static/... from src/sideline/web -------------------------------

// Map a /static/... URL path to the file the app serves at that URL.
function staticFileFor(urlPath) {
  if (!urlPath.startsWith("/static/")) return null;
  const file = path.resolve(WEB_DIR, urlPath.slice("/static/".length));
  const inside = file === WEB_DIR || file.startsWith(WEB_DIR + path.sep);
  if (!inside || !fs.existsSync(file) || !fs.statSync(file).isFile()) return null;
  return file;
}

// jsdom's usable-resources feature with a local loader: /static/... comes
// from src/sideline/web and external assets (the Google Fonts stylesheet)
// get an empty stylesheet, so the tests run fully offline.
class StaticResourceLoader extends ResourceLoader {
  fetch(url) {
    let parsed;
    try {
      parsed = new URL(url);
    } catch {
      return Promise.reject(new Error("not a URL: " + url));
    }
    if (parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1") {
      const file = staticFileFor(parsed.pathname);
      if (!file) return Promise.reject(new Error("no local file for " + parsed.pathname));
      return Promise.resolve(fs.readFileSync(file));
    }
    return Promise.resolve(Buffer.from("/* offline: no external assets in the smoke tests */\n"));
  }
}

// -- running one page ---------------------------------------------------------

const IMPORT_RE = /import\s*\{([^}]*)\}\s*from\s*["']([^"']+)["']\s*;?/g;
const NAME_RE = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

// Rewrite the page's one <script type="module"> into a classic <script> that
// jsdom executes: each `import { a, b } from "/static/lib/x.js";` becomes
// `const { a, b } = globalThis.__smokeModules["/static/lib/x.js"];`. The
// registry is filled with the real module namespace objects, imported from
// the files the /static mapping resolves to. A different source, a
// non-identifier name or a duplicated name fails loudly instead of
// misbehaving.
async function rewriteModuleScript(html, page) {
  const m = html.match(/<script type="module">([\s\S]*?)<\/script>/);
  assert.ok(m, page + ": expected exactly one <script type=\"module\">");
  const code = m[1];
  const namesBySource = new Map();
  const seen = new Set();
  for (const im of code.matchAll(IMPORT_RE)) {
    const names = im[1].split(",").map((s) => s.trim()).filter(Boolean);
    assert.ok(names.length > 0, page + ": import statement without names");
    for (const n of names) assert.ok(NAME_RE.test(n), page + ": not a plain name import: " + n);
    const src = im[2];
    assert.ok(src.startsWith("/static/"), page + ": unexpected module source " + src);
    if (!namesBySource.has(src)) namesBySource.set(src, []);
    for (const n of names) {
      assert.ok(!seen.has(n), page + ": duplicate imported name " + n);
      seen.add(n);
    }
    namesBySource.get(src).push(...names);
  }
  assert.ok(namesBySource.size > 0, page + ": expected the module script to import its libraries");

  const modules = {};
  for (const src of namesBySource.keys()) {
    const file = staticFileFor(src);
    assert.ok(file, page + ": " + src + " does not map to a file under src/sideline/web");
    modules[src] = await import(pathToFileURL(file).href);
  }

  const rewritten = code.replace(
    IMPORT_RE,
    (whole, names, src) =>
      "const { " + names.split(",").map((s) => s.trim()).filter(Boolean).join(", ") +
        " } = globalThis.__smokeModules[" + JSON.stringify(src) + "];",
  );
  IMPORT_RE.lastIndex = 0;
  assert.ok(!IMPORT_RE.test(rewritten), page + ": import statement left after the rewrite");
  return {
    html: html.replace(/<script type="module">[\s\S]*?<\/script>/, () => "<script>" + rewritten + "</script>"),
    modules,
  };
}

class FakeResponse {
  constructor(body, status) {
    this.ok = status >= 200 && status < 300;
    this.status = status;
    this._body = body;
  }
  json() {
    return Promise.resolve(JSON.parse(JSON.stringify(this._body)));
  }
}

// The pages' fetch: answers from `answers` (rows of [method, pathWithSearch,
// status, body]); a request that matches no row is answered with 500 and is
// then flagged by assertFixtureOnly. `reject` fails every request like a
// network error. Every request is recorded in `requests` with its full URL,
// so the origin is part of what assertFixtureOnly checks.
function makeFetch({ answers, reject, requests }) {
  return function fetch(url, init = {}) {
    const u = new URL(url, APP_ORIGIN);
    const method = String(init.method || "GET").toUpperCase();
    const key = method + " " + u.pathname + u.search;
    requests.push({ method, url: u.href });
    return new Promise((resolve, fail) => {
      if (reject) return fail(reject);
      const hit = answers.find(([m, p, s]) => m === method && p === u.pathname + u.search);
      if (!hit) return resolve(new FakeResponse({ detail: "no fixture for " + key }, 500));
      resolve(new FakeResponse(hit[3], hit[2]));
    });
  };
}

// jsdom limitation messages that are not script errors of the page: without
// the canvas npm package getContext returns null (the pages handle that),
// and the CSS parser complains about modern CSS it cannot read.
const BENIGN_JS = [/HTMLCanvasElement\.prototype\.getContext/, /Could not parse CSS/];

// -- unhandled promise rejections from page code -----------------------------
//
// jsdom runs the page's scripts in this process, and an unhandled promise
// rejection from page code does not surface as a window "error" event or a
// jsdomError (verified with jsdom 24 + Node 22): it escapes to the host and
// crashes the whole test run mid-file. Routing every rejection to a single
// owning page (the most recently opened one, named in the recorded message)
// fails that page's test instead; a rejection with no open page is recorded
// and failed by the final test.

// Open page handles in the order the pages were opened; the last one is the
// owner a rejection is attributed to.
const openPages = [];
const lateRejections = [];
process.on("unhandledRejection", (reason) => {
  const detail = reason && (reason.stack || reason.message) ? String(reason.stack || reason.message) : String(reason);
  const owner = openPages[openPages.length - 1];
  if (!owner) {
    lateRejections.push("unhandled promise rejection: " + detail);
    return;
  }
  owner.rejections.push("unhandled promise rejection (page: " + owner.page + "): " + detail);
});

async function loadPage(t, { page, url, answers, reject, withClock }) {
  const { html, modules } = await rewriteModuleScript(
    fs.readFileSync(path.join(WEB_DIR, page), "utf8"),
    page,
  );
  const requests = [];
  const windowErrors = [];
  const jsdomErrors = [];
  const rejections = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on("jsdomError", (err) => {
    const msg = String((err && err.message) || err);
    if (BENIGN_JS.some((re) => re.test(msg))) return;
    jsdomErrors.push(msg);
  });
  let pump = null;
  let rafQueue = null;
  if (withClock) {
    // Deterministic replay clock. jsdom has neither requestAnimationFrame
    // out of the box nor a controllable clock, so the test drives the page's
    // animation loop frame by frame against this clock. The page builds its
    // Timeline without a clock argument, and Timeline's default `now` reads
    // `performance` in the realm where the module was imported (node), so
    // the page gets a Timeline whose default clock is the fake one; the
    // `now` option is the seam the library provides for exactly this.
    let now = 0;
    const REPLAY_SRC = "/static/lib/replay.js";
    const ns = modules[REPLAY_SRC];
    assert.ok(ns && ns.Timeline, "the match page imports Timeline from " + REPLAY_SRC);
    const RealTimeline = ns.Timeline;
    const WindowTimeline = class extends RealTimeline {
      constructor(opts = {}) {
        super({ now: () => now, ...opts });
      }
    };
    modules[REPLAY_SRC] = { ...ns, Timeline: WindowTimeline };
    pump = (count, msPerFrame) => {
      for (let i = 0; i < count; i++) {
        now += msPerFrame;
        const batch = rafQueue.splice(0, rafQueue.length);
        for (const f of batch) f.cb(now);
      }
    };
  }
  const dom = new JSDOM(html, {
    url,
    runScripts: "dangerously",
    resources: new StaticResourceLoader(),
    virtualConsole,
    beforeParse(window) {
      window.addEventListener("error", (e) => windowErrors.push(String(e.message || (e.error && e.error.message) || e)));
      window.__smokeModules = modules;
      window.fetch = makeFetch({ answers, reject, requests });
      if (withClock) {
        let nextId = 1;
        rafQueue = [];
        Object.defineProperty(window, "performance", {
          value: { now: () => now },
          configurable: true,
          writable: true,
        });
        window.requestAnimationFrame = (cb) => {
          const id = nextId++;
          rafQueue.push({ id, cb });
          return id;
        };
        window.cancelAnimationFrame = (id) => {
          for (let i = rafQueue.length - 1; i >= 0; i--) if (rafQueue[i].id === id) rafQueue.splice(i, 1);
        };
      }
    },
  });
  const handle = {
    page,
    dom,
    doc: dom.window.document,
    win: dom.window,
    requests,
    windowErrors,
    jsdomErrors,
    rejections,
    pump,
    close: () => dom.window.close(),
  };
  openPages.push(handle);
  t.after(async () => {
    // Keep the page open briefly so an async page error that lands just after
    // the test body still fails this test, not the next one.
    await new Promise((r) => setTimeout(r, 100));
    const errs = [...windowErrors, ...jsdomErrors, ...rejections];
    if (errs.length) throw new Error("page errors after the checks: " + JSON.stringify(errs));
    openPages.splice(openPages.indexOf(handle), 1);
    handle.close();
  });
  return handle;
}

// When `handle` is given, a page that threw while waiting can never satisfy
// `check`, so fail at the first recorded script error with the error in the
// message instead of polling to the timeout (a blank page without errors
// still fails with the plain timeout message).
async function waitFor(check, what, timeout = 3000, handle = null) {
  const start = Date.now();
  const pageErrors = () => (handle ? [...handle.windowErrors, ...handle.jsdomErrors, ...handle.rejections] : []);
  for (;;) {
    const errs = pageErrors();
    if (errs.length) {
      throw new Error("page errors while waiting for " + what + ": " + JSON.stringify(errs));
    }
    if (check()) return;
    if (Date.now() - start > timeout) throw new Error("timed out waiting for: " + what);
    await new Promise((r) => setTimeout(r, 5));
  }
}

const requestKey = (r) => r.method + " " + r.url;

// The page may only ask for what the fixtures record: to the app's own
// origin (a request to another origin that reuses an allowed path is not in
// the fixtures) and only for the fixture routes.
function assertFixtureOnly(handle, allowed) {
  const offOrigin = handle.requests.filter((r) => new URL(r.url).origin !== APP_ORIGIN);
  assert.deepEqual(offOrigin, [], "requested URLs outside the app origin: " + JSON.stringify(offOrigin));
  const allowedSet = new Set(allowed.map(requestKey));
  const unexpected = handle.requests.filter((r) => !allowedSet.has(requestKey(r)));
  assert.deepEqual(unexpected, [], "requested URLs that are not in the fixtures: " + JSON.stringify(unexpected));
}

// Neither a window error event, an uncaught script error in jsdom, nor an
// unhandled promise rejection from the page's async code.
function assertNoScriptErrors(handle) {
  assert.deepEqual(handle.windowErrors, [], "window error events: " + JSON.stringify(handle.windowErrors));
  assert.deepEqual(handle.jsdomErrors, [], "uncaught script errors: " + JSON.stringify(handle.jsdomErrors));
  assert.deepEqual(handle.rejections, [], "unhandled promise rejections: " + JSON.stringify(handle.rejections));
}

// -- fixtures-backed routes ---------------------------------------------------

const LIB_ALLOWED = [
  { method: "GET", url: APP_ORIGIN + "/api/limits" },
  { method: "GET", url: APP_ORIGIN + "/api/matches" },
];
const LIB_LIST_OK = [
  ["GET", "/api/limits", 200, fixtures.limits],
  ["GET", "/api/matches", 200, fixtures.matches],
];
const LIB_LIST_EMPTY = [
  ["GET", "/api/limits", 200, fixtures.limits],
  ["GET", "/api/matches", 200, fixtures.emptyMatches],
];
const MATCH_ALLOWED = [
  { method: "GET", url: APP_ORIGIN + "/api/matches/" + MATCH_ID },
  { method: "GET", url: APP_ORIGIN + "/api/matches/" + MATCH_ID + "/analytics" },
  { method: "GET", url: APP_ORIGIN + "/api/matches/" + MATCH_ID + "/frames?start=0&end=249" },
];
const MATCH_ANSWERS = MATCH_ALLOWED.map((r) => {
  const p = r.url.slice(APP_ORIGIN.length); // fetch answer rows are keyed by path+search
  return [r.method, p, 200, fixtureFor(p)];
});

function fixtureFor(url) {
  if (url === "/api/matches/" + MATCH_ID) return fixtures.match;
  if (url === "/api/matches/" + MATCH_ID + "/analytics") return fixtures.analytics;
  return fixtures.frames;
}

// -- library page ---------------------------------------------------------------

test("library page renders the stored match from the real API shape", async (t) => {
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: LIB_LIST_OK,
  });
  await waitFor(() => h.doc.querySelectorAll("#matchList .match-row").length === 1, "one .match-row", 3000, h);

  const rows = h.doc.querySelectorAll("#matchList .match-row");
  assert.equal(rows.length, 1);
  const row = rows[0];
  assert.equal(row.getAttribute("data-match-id"), MATCH_ID);
  assert.equal(row.querySelector(".name").textContent, MATCH_NAME);
  // 750 frames 0..749 at 25 fps is 30.0 s (the sample recording).
  assert.equal(row.querySelector(".duration").textContent, "0:30");
  assert.equal(row.querySelector("a.open").getAttribute("href"), "/match.html?id=" + MATCH_ID);
  assert.equal(h.doc.querySelectorAll("#matchList p.empty").length, 0);

  assert.deepEqual(
    h.requests.map(requestKey).sort(),
    [
      "GET " + APP_ORIGIN + "/api/limits",
      "GET " + APP_ORIGIN + "/api/matches",
    ].sort(),
  );
  assertNoScriptErrors(h);
});

test("library page shows the empty note when the list is empty", async (t) => {
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: LIB_LIST_EMPTY,
  });
  await waitFor(() => h.doc.querySelector("#matchList p.empty"), "the empty note", 3000, h);

  assert.equal(h.doc.querySelectorAll("#matchList .match-row").length, 0);
  assert.ok(h.doc.querySelector("#matchList p.empty").textContent.startsWith("No matches yet"));
  assertNoScriptErrors(h);
  assertFixtureOnly(h, LIB_ALLOWED);
});

test("library page shows an error message when fetch rejects", async (t) => {
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: LIB_LIST_OK,
    reject: new TypeError("Failed to fetch (network down)"),
  });
  await waitFor(() => !h.doc.getElementById("alert").hidden, "the visible alert", 3000, h);

  const alert = h.doc.getElementById("alert");
  assert.equal(alert.hidden, false);
  assert.ok(alert.textContent.startsWith("Could not load the match list"), "alert text: " + alert.textContent);
  assert.ok(alert.textContent.includes("Failed to fetch"), "alert text: " + alert.textContent);
  assert.equal(h.doc.querySelectorAll("#matchList .match-row").length, 0);
  // The limits request failed too: the page degrades to the no-worker note
  // instead of crashing.
  assert.ok(h.doc.getElementById("uploadNote").textContent.includes("not available"),
    "note: " + h.doc.getElementById("uploadNote").textContent);
  assertNoScriptErrors(h);
  assertFixtureOnly(h, LIB_ALLOWED);
});

test("library page shows an error message when the server answers 500", async (t) => {
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 200, fixtures.limits],
      ["GET", "/api/matches", 500, { detail: "boom" }],
    ],
  });
  await waitFor(() => !h.doc.getElementById("alert").hidden, "the visible alert", 3000, h);

  const alert = h.doc.getElementById("alert");
  assert.equal(alert.hidden, false);
  assert.ok(alert.textContent.startsWith("Could not load the match list"), "alert text: " + alert.textContent);
  assert.ok(alert.textContent.includes("500"), "alert text: " + alert.textContent);
  assert.equal(h.doc.querySelectorAll("#matchList .match-row").length, 0);
  assertNoScriptErrors(h);
  assertFixtureOnly(h, LIB_ALLOWED);
});

// -- match page ------------------------------------------------------------------

test("match page renders meta, teams, player table, heatmaps and export links", async (t) => {
  const h = await loadPage(t, {
    page: "match.html",
    url: "http://localhost/match.html?id=" + MATCH_ID,
    answers: MATCH_ANSWERS,
  });
  const clockEl = h.doc.getElementById("clock");
  const ready = () =>
    h.doc.querySelectorAll("#teamCards .team-card").length === 2 &&
    clockEl.textContent === "0:00.0 / " + replay.formatClock(TOTAL_S);
  await waitFor(ready, "both team cards and the first replay chunk", 3000, h);

  // Team cards.
  const cards = h.doc.querySelectorAll("#teamCards .team-card");
  assert.deepEqual(
    [...cards].map((c) => c.getAttribute("data-team")),
    ["left", "right"],
  );
  assert.equal(cards[0].querySelector(".team-name").textContent, "Left");
  assert.equal(cards[1].querySelector(".team-name").textContent, "Right");

  // Player table: one row per player in the analytics summary.
  const table = h.doc.querySelector("#playerTable table.players");
  assert.ok(table, "the player table");
  assert.equal(h.doc.querySelectorAll("#playerTable table.players thead th").length, 8);
  assert.equal(
    h.doc.querySelectorAll("#playerTable table.players tbody tr").length,
    fixtures.analytics.players.length,
  );

  // Heatmap canvases, sized by the fixture grids (20 px per cell).
  for (const [id, grid] of [
    ["heatLeft", fixtures.analytics.heatmaps.left],
    ["heatRight", fixtures.analytics.heatmaps.right],
    ["heatBall", fixtures.analytics.heatmaps.ball],
  ]) {
    const canvas = h.doc.getElementById(id).querySelector("canvas.heatmap");
    assert.ok(canvas, id + " heatmap canvas");
    assert.equal(canvas.width, grid[0].length * 20);
    assert.equal(canvas.height, grid.length * 20);
    assert.equal(canvas.getAttribute("data-max"), String(Math.max(0, ...grid.flat())));
  }

  // Export links and match meta.
  assert.equal(h.doc.getElementById("actions").hidden, false);
  assert.equal(
    h.doc.getElementById("exportPositions").href,
    "http://localhost/api/matches/" + MATCH_ID + "/export.csv?kind=positions",
  );
  assert.equal(
    h.doc.getElementById("exportPlayers").href,
    "http://localhost/api/matches/" + MATCH_ID + "/export.csv?kind=players",
  );
  assert.equal(h.doc.getElementById("matchName").textContent, MATCH_NAME);
  assert.equal(h.doc.getElementById("barTitle").textContent, MATCH_NAME);
  const facts = [...h.doc.querySelectorAll("#matchFacts span")].map((s) => s.textContent);
  assert.equal(facts.length, 3);
  assert.ok(facts.includes("Duration " + render.formatDuration(fixtures.match.duration_s)), JSON.stringify(facts));
  assert.ok(facts.includes("Frames " + fixtures.match.frame_count), JSON.stringify(facts));
  assert.ok(facts.includes("Size " + render.formatBytes(fixtures.match.size_bytes)), JSON.stringify(facts));

  // Replay panel: canvas present, first chunk loaded, controls enabled.
  assert.equal(h.doc.getElementById("replay").tagName, "CANVAS");
  assert.equal(h.doc.getElementById("replayLoading").hidden, true);
  assert.equal(h.doc.getElementById("play").disabled, false);
  assert.equal(h.doc.getElementById("timeline").disabled, false);
  assert.equal(clockEl.textContent, "0:00.0 / " + replay.formatClock(TOTAL_S));

  const got = [...h.requests].map(requestKey).sort();
  assert.deepEqual(got, MATCH_ALLOWED.map(requestKey).sort());
  assertNoScriptErrors(h);
});

test("match page play button advances the replay clock", async (t) => {
  const h = await loadPage(t, {
    page: "match.html",
    url: "http://localhost/match.html?id=" + MATCH_ID,
    answers: MATCH_ANSWERS,
    withClock: true,
  });
  const clockEl = h.doc.getElementById("clock");
  const playBtn = h.doc.getElementById("play");
  const clickPlay = () => playBtn.dispatchEvent(new h.win.MouseEvent("click", { bubbles: true, cancelable: true }));
  await waitFor(
    () => clockEl.textContent === "0:00.0 / " + replay.formatClock(TOTAL_S),
    "the first replay chunk",
    3000,
    h,
  );

  assert.equal(playBtn.textContent, "Play");
  clickPlay();
  assert.equal(playBtn.textContent, "Pause");
  // Three animation frames at 100 ms each advance the clock by 0.3 s at 1x.
  h.pump(3, 100);
  assert.equal(clockEl.textContent, replay.formatClock(0.3) + " / " + replay.formatClock(TOTAL_S));
  assert.equal(h.doc.getElementById("timeline").value, "8"); // 0.3 s * 25 fps, rounded
  // Still inside the first 250-frame chunk: no further frames request.
  assert.equal(h.requests.length, 3);

  clickPlay();
  assert.equal(playBtn.textContent, "Play");

  assertFixtureOnly(h, MATCH_ALLOWED);
  assertNoScriptErrors(h);
});

// Safety net for a rejection that lands after the page that caused it was
// already closed (the settle wait in loadPage's after hook covers the
// realistic window; this catches anything later than that).
test("no unhandled promise rejection outlived its page", () => {
  assert.deepEqual(lateRejections, [], "unhandled promise rejections with no open page: " + JSON.stringify(lateRejections));
});
