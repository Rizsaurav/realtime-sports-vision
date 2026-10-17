// Extra adversarial tests for the video upload flow on the library page
// (tasks/video_client.md, page section).
//
// The smoke suite (web_tests_smoke/video.test.mjs) covers the happy path:
// the enabled picker/note, one upload with progress and polling to an Open
// link, one failed job, a 502 at upload time, and the no-worker page.
// These tests attack the spec rules it does not check, with the same
// offline setup: jsdom runs the page's own scripts, fetch and
// XMLHttpRequest are answered from fixture rows on a controllable clock,
// so the 3 s polling interval is advanced step by step instead of waited
// for.
//
// Probed here:
// - the job row is polled every 3 seconds (one status fetch per 3 s tick,
//   the first immediately after the upload is accepted) and polling stops
//   once the job is done;
// - two video uploads in flight give two independent job rows, each
//   becoming an Open link to *its own* match, and a job reaching done
//   never clobbers the other row;
// - a worker that becomes unreachable while polling (the server's 5xx
//   answers) keeps the row polling: the server's error is shown in the
//   row meanwhile, and polling only stops after several consecutive 5xx
//   answers (the server records no outcome for them and the job stays
//   resolvable);
// - a video file with an uppercase extension (CLIP.MP4) is still treated
//   as a video (POST /api/videos, a job row);
// - a .jsonl file still goes through the old match upload path when video
//   processing is enabled (the video extensions are added to the picker,
//   not replacing .jsonl);
// - a zero-byte video shows the server's own 400 message in the alert and
//   creates no job row;
// - a failing GET /api/limits keeps the page working as before (no worker
//   behaviour: the old picker, the "not available" note, the old upload
//   path) without any script error.

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { JSDOM, ResourceLoader, VirtualConsole } from "jsdom";

const here = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.resolve(here, "../src/sideline/web");
const FIXTURES_DIR = path.resolve(here, "../web_tests_smoke/fixtures");
const APP_ORIGIN = "http://localhost";

const fixtures = {
  limits: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "limits.json"), "utf8")),
  limitsVideo: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "limits_video.json"), "utf8")),
  emptyMatches: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "empty_matches.json"), "utf8")),
  matches: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "matches.json"), "utf8")),
  match: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "match.json"), "utf8")),
};
const M1 = fixtures.match; // the recorded sample match
const M2 = {
  id: "m_deadbeef01",
  name: "Second clip",
  created_at: "2026-01-02T00:00:00Z",
  frame_count: 5,
  first_frame: 0,
  last_frame: 4,
  duration_s: 0.2,
  delay: 0,
  colors: null,
  size_bytes: 123,
};

function jobStatus(jobId, status, extra) {
  return Object.assign(
    {
      job_id: jobId,
      status,
      progress: 0.0,
      frames: 0,
      name: "Friday night",
      error: null,
      created_at: "2026-01-01T00:00:00Z",
      seconds: null,
    },
    extra,
  );
}

// -- serving /static/... from src/sideline/web ---------------------------------

function staticFileFor(urlPath) {
  if (!urlPath.startsWith("/static/")) return null;
  const file = path.resolve(WEB_DIR, urlPath.slice("/static/".length));
  const inside = file === WEB_DIR || file.startsWith(WEB_DIR + path.sep);
  if (!inside || !fs.existsSync(file) || !fs.statSync(file).isFile()) return null;
  return file;
}

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
    return Promise.resolve(Buffer.from("/* offline: no external assets in the extra tests */\n"));
  }
}

// -- running one page -----------------------------------------------------------

const IMPORT_RE = /import\s*\{([^}]*)\}\s*from\s*["']([^"']+)["']\s*;?/g;
const NAME_RE = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

// Rewrite the page's one <script type="module"> into a classic <script> that
// jsdom executes (same technique as the smoke suite): the imports are
// replaced by lookups in a registry of real module namespaces.
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

// The page's fetch: answers from `answers` (rows of [method, pathWithSearch,
// status, body]), consumed in request order. A request that matches no
// remaining row is answered with 500 and flagged by assertFixtureOnly.
function makeFetch({ answers, requests }) {
  return function fetch(url, init = {}) {
    const u = new URL(url, APP_ORIGIN);
    const method = String(init.method || "GET").toUpperCase();
    requests.push({ method, url: u.href });
    const key = method + " " + u.pathname + u.search;
    const row = answers.find((r) => !r._used && r[0] === method && r[1] === u.pathname + u.search);
    if (!row) return Promise.resolve(new FakeResponse({ detail: "no fixture for " + key }, 500));
    row._used = true;
    return Promise.resolve(new FakeResponse(row[3], row[2]));
  };
}

// A controllable clock installed on the page window: the page's job
// polling (setInterval) and the fake XMLHttpRequest's async steps run
// against this clock, so the test advances time in steps.
function makeClock() {
  let now = 0;
  let nextId = 1;
  const timers = new Map();
  function schedule(delay, fn, repeat) {
    const id = nextId++;
    timers.set(id, { at: now + Math.max(0, delay), fn, repeat });
    return id;
  }
  const api = {
    setTimeout: (fn, delay) => schedule(delay, fn, 0),
    setInterval: (fn, delay) => schedule(delay, fn, delay),
    clearTimeout: (id) => timers.delete(id),
    clearInterval: (id) => timers.delete(id),
    later: (fn, delay) => schedule(delay, fn, 0),
    advance(ms) {
      const end = now + ms;
      for (;;) {
        let dueId = null;
        let dueAt = Infinity;
        for (const [id, t] of timers) {
          if (t.at <= end && t.at < dueAt) {
            dueAt = t.at;
            dueId = id;
          }
        }
        if (dueId === null) break;
        const t = timers.get(dueId);
        now = t.at;
        if (t.repeat > 0) t.at += t.repeat;
        else timers.delete(dueId);
        t.fn();
      }
      now = end;
    },
  };
  return api;
}

// A minimal XMLHttpRequest for the page's video upload: it answers from
// the same fixture rows as the fake fetch and fires its events on the
// clock - a progress event at 50% and 100%, then the load event.
class FakeXHR {
  constructor({ answers, clock, requests }) {
    this._answers = answers;
    this._clock = clock;
    this._requests = requests;
    this._listeners = {};
    this.readyState = 0;
    this.status = 0;
    this.responseText = "";
    this.upload = {
      addEventListener: (type, fn) => {
        const key = "upload:" + type;
        this._listeners[key] = this._listeners[key] || [];
        this._listeners[key].push(fn);
      },
    };
  }
  open(method, url) {
    this._method = method;
    this._url = url;
  }
  setRequestHeader() {}
  addEventListener(type, fn) {
    this._listeners[type] = this._listeners[type] || [];
    this._listeners[type].push(fn);
  }
  _emit(key, event) {
    (this._listeners[key] || []).forEach((fn) => fn(event));
  }
  send(body) {
    const u = new URL(this._url, APP_ORIGIN);
    const sent = { method: this._method, url: u.href };
    if (body && typeof body.get === "function") {
      const file = body.get("file");
      sent.file = file && file.name ? String(file.name) : null;
      sent.fileSize = file && typeof file.size === "number" ? file.size : 0;
      const name = body.get("name");
      sent.name = name === null ? null : String(name);
    }
    this._requests.push(sent);
    const row = this._answers.find((r) => !r._used && r[0] === this._method && r[1] === u.pathname + u.search);
    if (!row) {
      this._clock.later(() => this._emit("error", {}), 0);
      return;
    }
    row._used = true;
    const total = sent.fileSize || 0;
    this._clock.later(() => this._emit("upload:progress", { lengthComputable: total > 0, loaded: Math.ceil(total / 2), total }), 100);
    this._clock.later(() => this._emit("upload:progress", { lengthComputable: total > 0, loaded: total, total }), 200);
    this._clock.later(() => {
      this.readyState = 4;
      this.status = row[2];
      this.responseText = JSON.stringify(row[3]);
      this._emit("load", {});
    }, 300);
  }
}

const BENIGN_JS = [/HTMLCanvasElement\.prototype\.getContext/, /Could not parse CSS/];

// -- unhandled promise rejections from page code -------------------------------

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

async function loadPage(t, { page, url, answers }) {
  const { html, modules } = await rewriteModuleScript(
    fs.readFileSync(path.join(WEB_DIR, page), "utf8"),
    page,
  );
  const requests = [];
  const windowErrors = [];
  const jsdomErrors = [];
  const rejections = [];
  const clock = makeClock();
  const virtualConsole = new VirtualConsole();
  virtualConsole.on("jsdomError", (err) => {
    const msg = String((err && err.message) || err);
    if (BENIGN_JS.some((re) => re.test(msg))) return;
    jsdomErrors.push(msg);
  });
  const dom = new JSDOM(html, {
    url,
    runScripts: "dangerously",
    resources: new StaticResourceLoader(),
    virtualConsole,
    beforeParse(window) {
      window.addEventListener("error", (e) => windowErrors.push(String(e.message || (e.error && e.error.message) || e)));
      window.__smokeModules = modules;
      window.fetch = makeFetch({ answers, requests });
      window.XMLHttpRequest = function () {
        return new FakeXHR({ answers, clock, requests });
      };
      window.setTimeout = clock.setTimeout;
      window.setInterval = clock.setInterval;
      window.clearTimeout = clock.clearTimeout;
      window.clearInterval = clock.clearInterval;
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
    clock,
    close: () => dom.window.close(),
  };
  openPages.push(handle);
  t.after(async () => {
    // Keep the page open briefly so an async page error that lands just
    // after the test body still fails this test, not the next one.
    await new Promise((r) => setTimeout(r, 100));
    const errs = [...windowErrors, ...jsdomErrors, ...rejections];
    if (errs.length) throw new Error("page errors after the checks: " + JSON.stringify(errs));
    openPages.splice(openPages.indexOf(handle), 1);
    handle.close();
  });
  return handle;
}

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

// Flush the page's promise chains (the fixture answers resolve through
// microtasks, not through the controllable clock).
const tick = () => new Promise((r) => setImmediate(r));
async function ticks(count) {
  for (let i = 0; i < count; i++) await tick();
}

function assertFixtureOnly(handle, allowed) {
  const offOrigin = handle.requests.filter((r) => new URL(r.url).origin !== APP_ORIGIN);
  assert.deepEqual(offOrigin, [], "requested URLs outside the app origin: " + JSON.stringify(offOrigin));
  const allowedSet = new Set(allowed.map(requestKey));
  const unexpected = handle.requests.filter((r) => !allowedSet.has(requestKey(r)));
  assert.deepEqual(unexpected, [], "requested URLs that are not in the fixtures: " + JSON.stringify(unexpected));
}

function assertNoScriptErrors(handle) {
  assert.deepEqual(handle.windowErrors, [], "window error events: " + JSON.stringify(handle.windowErrors));
  assert.deepEqual(handle.jsdomErrors, [], "uncaught script errors: " + JSON.stringify(handle.jsdomErrors));
  assert.deepEqual(handle.rejections, [], "unhandled promise rejections: " + JSON.stringify(handle.rejections));
}

// -- page helpers ----------------------------------------------------------------

function pickFile(h, fileName, size, name) {
  const file = new h.win.File([new Uint8Array(size)], fileName, { type: "video/mp4" });
  const input = h.doc.getElementById("file");
  Object.defineProperty(input, "files", { value: [file], configurable: true });
  if (name !== undefined) h.doc.getElementById("name").value = name;
  return file;
}

function submitForm(h) {
  h.doc.getElementById("uploadForm").dispatchEvent(new h.win.Event("submit", { bubbles: true, cancelable: true }));
}

const JOB_ROWS = (h) => [...h.doc.querySelectorAll("#jobList .job-row")];
const ROW = (h, jobId) => h.doc.querySelector('#jobList .job-row[data-job-id="' + jobId + '"]');
const STATE = (h, jobId) => {
  const row = ROW(h, jobId);
  return row ? row.querySelector(".job-state").textContent : null;
};

// -- tests ------------------------------------------------------------------------

test("a video job is polled every 3 seconds and polling stops when done", async (t) => {
  const JOB = "j_000000000001";
  const RUNNING = () => jobStatus(JOB, "running", { progress: 0.42, frames: 105 });
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 200, fixtures.limitsVideo],
      ["GET", "/api/matches", 200, fixtures.emptyMatches],
      ["POST", "/api/videos", 202, { job_id: JOB, status: "queued" }],
      ["GET", "/api/videos/" + JOB, 200, RUNNING()],
      ["GET", "/api/videos/" + JOB, 200, RUNNING()],
      ["GET", "/api/videos/" + JOB, 200, RUNNING()],
      ["GET", "/api/videos/" + JOB, 200, RUNNING()],
      ["GET", "/api/videos/" + JOB, 200, jobStatus(JOB, "done", { progress: 1.0, frames: 750, seconds: 30.0, match_id: M1.id })],
      ["GET", "/api/matches", 200, fixtures.matches], // the list after done
    ],
  });
  await waitFor(() => h.doc.getElementById("uploadNote").textContent.includes("GPU machine"), "the video note", 3000, h);

  pickFile(h, "clip.mp4", 4096, "Friday night");
  submitForm(h);
  await ticks(2);
  h.clock.advance(300); // the 202 answer lands: the first poll happens now
  await ticks(3);
  const polls = () => h.requests.filter((r) => r.url === APP_ORIGIN + "/api/videos/" + JOB).length;
  assert.equal(polls(), 1, "one status fetch right after the upload is accepted");
  assert.equal(STATE(h, JOB), "Processing 42%");

  h.clock.advance(3000); // t = 3 s
  await ticks(3);
  assert.equal(polls(), 2, "a poll on the 3 s tick");
  assert.equal(STATE(h, JOB), "Processing 42%");

  h.clock.advance(3000); // t = 6 s
  await ticks(3);
  assert.equal(polls(), 3, "a poll on the 6 s tick");

  h.clock.advance(3000); // t = 9 s
  await ticks(3);
  assert.equal(polls(), 4, "a poll on the 9 s tick");

  h.clock.advance(3000); // t = 12 s: the job is done
  await ticks(3);
  assert.equal(polls(), 5, "a poll on the 12 s tick");
  assert.equal(STATE(h, JOB), "Done");
  const open = ROW(h, JOB).querySelector("a.open");
  assert.ok(open, "the Open link");
  assert.equal(open.getAttribute("href"), "/match.html?id=" + M1.id);

  // No further polls after the job is done.
  h.clock.advance(9000);
  await ticks(3);
  assert.equal(polls(), 5, "polling stopped after done");

  assert.deepEqual(
    h.requests.map(requestKey),
    [
      "GET " + APP_ORIGIN + "/api/limits",
      "GET " + APP_ORIGIN + "/api/matches",
      "POST " + APP_ORIGIN + "/api/videos",
      "GET " + APP_ORIGIN + "/api/videos/" + JOB,
      "GET " + APP_ORIGIN + "/api/videos/" + JOB,
      "GET " + APP_ORIGIN + "/api/videos/" + JOB,
      "GET " + APP_ORIGIN + "/api/videos/" + JOB,
      "GET " + APP_ORIGIN + "/api/videos/" + JOB,
      "GET " + APP_ORIGIN + "/api/matches",
    ],
  );
  assertNoScriptErrors(h);
});

test("two video uploads get their own job rows and their own Open links", async (t) => {
  const A = "j_00000000000a";
  const B = "j_00000000000b";
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 200, fixtures.limitsVideo],
      ["GET", "/api/matches", 200, fixtures.emptyMatches],
      ["POST", "/api/videos", 202, { job_id: A, status: "queued" }],
      ["POST", "/api/videos", 202, { job_id: B, status: "queued" }],
      ["GET", "/api/videos/" + A, 200, jobStatus(A, "running", { progress: 0.2, frames: 50, name: "Match A" })],
      ["GET", "/api/videos/" + B, 200, jobStatus(B, "running", { progress: 0.1, frames: 25, name: "Match B" })],
      ["GET", "/api/videos/" + A, 200, jobStatus(A, "done", { progress: 1.0, frames: 250, name: "Match A", match_id: M1.id })],
      ["GET", "/api/videos/" + B, 200, jobStatus(B, "done", { progress: 1.0, frames: 5, name: "Match B", match_id: M2.id })],
      ["GET", "/api/matches", 200, { items: [M1] }], // after A is done
      ["GET", "/api/matches", 200, { items: [M1, M2] }], // after B is done
    ],
  });
  await waitFor(() => h.doc.getElementById("uploadNote").textContent.includes("GPU machine"), "the video note", 3000, h);

  // Upload A (t = 0), its 202 answer lands at t = 300.
  pickFile(h, "a.mp4", 2048, "Match A");
  submitForm(h);
  await ticks(2);
  h.clock.advance(300);
  await ticks(3);
  assert.equal(JOB_ROWS(h).length, 1, "one job row while only A is in flight");
  assert.equal(STATE(h, A), "Processing 20%");

  // Upload B while A is still running (t = 300), its answer at t = 600.
  pickFile(h, "b.mp4", 2048, "Match B");
  submitForm(h);
  await ticks(2);
  h.clock.advance(300);
  await ticks(3);
  assert.equal(JOB_ROWS(h).length, 2, "two job rows for two uploads");
  assert.equal(STATE(h, B), "Processing 10%");
  assert.equal(STATE(h, A), "Processing 20%", "B's poll must not clobber A's row");

  // t = 3300: A's 3 s poll - A is done, B is still running.
  h.clock.advance(2700);
  await ticks(3);
  assert.equal(STATE(h, A), "Done");
  assert.equal(ROW(h, A).querySelector("a.open").getAttribute("href"), "/match.html?id=" + M1.id);
  assert.equal(STATE(h, B), "Processing 10%", "B's row is untouched while A resolves");
  assert.equal(h.doc.querySelectorAll("#matchList .match-row").length, 1, "A's match joins the list");

  // t = 3600: B's 3 s poll - B is done too.
  h.clock.advance(300);
  await ticks(3);
  assert.equal(STATE(h, B), "Done");
  assert.equal(ROW(h, B).querySelector("a.open").getAttribute("href"), "/match.html?id=" + M2.id);
  const ids = [...h.doc.querySelectorAll("#matchList .match-row")].map((r) => r.getAttribute("data-match-id")).sort();
  assert.deepEqual(ids, [M1.id, M2.id].sort(), "both matches are in the list");

  // Both jobs stop polling after they are done.
  const pollsA = () => h.requests.filter((r) => r.url === APP_ORIGIN + "/api/videos/" + A).length;
  const pollsB = () => h.requests.filter((r) => r.url === APP_ORIGIN + "/api/videos/" + B).length;
  const a = pollsA();
  const b = pollsB();
  h.clock.advance(9000);
  await ticks(3);
  assert.equal(pollsA(), a, "A stopped polling after done");
  assert.equal(pollsB(), b, "B stopped polling after done");

  const uploads = h.requests.filter((r) => r.url === APP_ORIGIN + "/api/videos");
  assert.equal(uploads.length, 2);
  assert.deepEqual(
    uploads.map((r) => [r.file, r.name]).sort(),
    [["a.mp4", "Match A"], ["b.mp4", "Match B"]].sort(),
  );
  assertNoScriptErrors(h);
});

test("a video with an uppercase extension is still treated as a video", async (t) => {
  const JOB = "j_00000000000d";
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 200, fixtures.limitsVideo],
      ["GET", "/api/matches", 200, fixtures.emptyMatches],
      ["POST", "/api/videos", 202, { job_id: JOB, status: "queued" }],
      ["GET", "/api/videos/" + JOB, 200, jobStatus(JOB, "queued", { name: "Upper clip" })],
    ],
  });
  await waitFor(() => h.doc.getElementById("uploadNote").textContent.includes("GPU machine"), "the video note", 3000, h);

  pickFile(h, "CLIP.MP4", 2048, "Upper clip");
  submitForm(h);
  await ticks(2);
  h.clock.advance(300);
  await ticks(3);

  const upload = h.requests.find((r) => r.url === APP_ORIGIN + "/api/videos");
  assert.ok(upload, "the file went to the video endpoint");
  assert.equal(upload.method, "POST");
  assert.equal(upload.file, "CLIP.MP4");
  assert.ok(!h.requests.some((r) => r.url === APP_ORIGIN + "/api/matches" && r.method === "POST"), "not the match upload path");
  const row = ROW(h, JOB);
  assert.ok(row, "the job row");
  assert.ok(STATE(h, JOB).length > 0, "the row shows a state: " + STATE(h, JOB));
  assert.equal(h.doc.getElementById("jobsSection").hidden, false);
  assertNoScriptErrors(h);
});

test("a .jsonl file still goes through the match upload path when video is enabled", async (t) => {
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 200, fixtures.limitsVideo],
      ["GET", "/api/matches", 200, fixtures.emptyMatches],
      ["POST", "/api/matches", 201, M1],
      ["GET", "/api/matches", 200, { items: [M1] }], // the list after the upload
    ],
  });
  await waitFor(() => h.doc.getElementById("uploadNote").textContent.includes("GPU machine"), "the video note", 3000, h);

  pickFile(h, "frames.jsonl", 2048, "Clip");
  submitForm(h);
  await waitFor(() => h.doc.querySelectorAll("#matchList .match-row").length === 1, "the match row", 3000, h);

  assert.ok(h.requests.some((r) => r.url === APP_ORIGIN + "/api/matches" && r.method === "POST"), "the .jsonl went to the match endpoint");
  assert.ok(!h.requests.some((r) => r.url === APP_ORIGIN + "/api/videos"), "no video upload for a .jsonl");
  assert.equal(JOB_ROWS(h).length, 0, "no job row");
  assert.equal(h.doc.getElementById("jobsSection").hidden, true);
  assert.equal(h.doc.querySelector("#matchList .match-row").getAttribute("data-match-id"), M1.id);
  assertNoScriptErrors(h);
});

test("a zero-byte video shows the server's 400 message and creates no job row", async (t) => {
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 200, fixtures.limitsVideo],
      ["GET", "/api/matches", 200, fixtures.emptyMatches],
      ["POST", "/api/videos", 400, { error: "file is empty; choose a video file with content" }],
    ],
  });
  await waitFor(() => h.doc.getElementById("uploadNote").textContent.includes("GPU machine"), "the video note", 3000, h);

  pickFile(h, "tiny.mp4", 0);
  submitForm(h);
  await ticks(2);
  h.clock.advance(300); // the 400 answer lands
  await ticks(3);

  const alert = h.doc.getElementById("alert");
  assert.equal(alert.hidden, false);
  assert.equal(alert.textContent, "Video upload failed: file is empty; choose a video file with content.");
  assert.equal(JOB_ROWS(h).length, 0, "no job row for a rejected upload");
  assert.equal(h.doc.getElementById("jobsSection").hidden, true);
  assert.equal(h.doc.getElementById("uploadBtn").disabled, false);
  assert.equal(h.doc.getElementById("uploadProgress").hidden, true);
  assertNoScriptErrors(h);
});

test("a failing /api/limits keeps the page working as before (no video)", async (t) => {
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 500, { detail: "boom" }],
      ["GET", "/api/matches", 200, fixtures.emptyMatches],
      ["POST", "/api/matches", 400, { error: "file is not valid UTF-8" }],
    ],
  });
  await waitFor(
    () => h.doc.getElementById("uploadNote").textContent.includes("not available"),
    "the no-video note",
    3000,
    h,
  );

  const note = h.doc.getElementById("uploadNote").textContent;
  assert.ok(note.includes("Video processing is not available on this server"), "note: " + note);
  // The file picker is unchanged.
  assert.equal(h.doc.getElementById("file").getAttribute("accept"), "video/*,.jsonl");

  // A picked video goes through the old match upload path and the page
  // shows the old-style error for the server's 400.
  pickFile(h, "clip.mp4", 2048, "Clip");
  submitForm(h);
  await waitFor(() => !h.doc.getElementById("alert").hidden, "the visible alert", 3000, h);
  assert.equal(h.doc.getElementById("alert").textContent, "Upload failed: the server answered 400.");
  assert.ok(h.requests.some((r) => r.url === APP_ORIGIN + "/api/matches" && r.method === "POST"), "the old upload path");
  assert.equal(JOB_ROWS(h).length, 0, "no job row");
  assert.equal(h.doc.getElementById("jobsSection").hidden, true);
  assertNoScriptErrors(h);
});

// Safety net for a rejection that lands after the page that caused it was
// already closed.
test("no unhandled promise rejection outlived its page", () => {
  assert.deepEqual(lateRejections, [], "unhandled promise rejections with no open page: " + JSON.stringify(lateRejections));
});
