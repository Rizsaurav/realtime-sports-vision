// Smoke tests for the video upload flow on the library page.
//
// Loads src/sideline/web/library.html into jsdom and lets the page run its
// own scripts. The page's fetch is answered from the JSON fixtures in
// ./fixtures (recorded by `python tests_integration/make_fixtures.py`
// against a real create_library_app), and the page's XMLHttpRequest (the
// video upload path) is answered from the same fixture rows. Every request
// is recorded and must be one of the fixture answers.
//
// The page's job polling (setInterval) and the fake XMLHttpRequest run on
// a controllable clock installed on the page window, so the test advances
// time in 3 s steps instead of waiting for the real polling interval.

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { JSDOM, ResourceLoader, VirtualConsole } from "jsdom";

const here = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.resolve(here, "../src/sideline/web");
const FIXTURES_DIR = path.resolve(here, "fixtures");
// The origin the page is loaded from; a request is only in the fixtures if
// it is to this origin (the fixtures record the app's routes).
const APP_ORIGIN = "http://localhost";

const fixtures = {
  limits: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "limits.json"), "utf8")),
  limitsVideo: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "limits_video.json"), "utf8")),
  emptyMatches: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "empty_matches.json"), "utf8")),
  matches: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "matches.json"), "utf8")),
  match: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "match.json"), "utf8")),
};
const MATCH_ID = fixtures.match.id;
const JOB = "j_000000000001";
const JOB2 = "j_000000000002";

// The worker job status shape (worker.py /jobs answer) used by the answers.
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

// The page's fetch: answers from `answers` (rows of [method, pathWithSearch,
// status, body]). Rows are consumed in request order, so the same route can
// answer differently over the test (the match list is empty at start and
// holds the new match once the video job is done). A request that matches
// no remaining row is answered with 500 and is then flagged by
// assertFixtureOnly. Every request is recorded in `requests`.
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

// A controllable clock installed on the page window before its scripts
// run: the page's job polling (setInterval) and the fake XMLHttpRequest's
// async steps run against this clock, so the test advances time in steps
// instead of waiting for the real 3 s polling interval.
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
// the same fixture rows as the fake fetch (consuming them, in request
// order) and fires its events on the clock - a progress event at 50% and
// 100%, then the load event with the fixture answer.
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
    const key = this._method + " " + u.pathname + u.search;
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

// Flush the page's promise chains (the fixture answers resolve through
// microtasks, not through the controllable clock).
const tick = () => new Promise((r) => setImmediate(r));
async function ticks(count) {
  for (let i = 0; i < count; i++) await tick();
}

// The page may only ask for what the fixtures record: to the app's own
// origin and only for the fixture routes.
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

// -- page helpers --------------------------------------------------------------

// Pick a fake video file in the form's file input (and set the name field
// when given). Returns the File.
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

const JOB_ROW = (h) => h.doc.querySelector("#jobList .job-row");

// -- tests ---------------------------------------------------------------------

test("video-enabled page: the picker accepts the five video extensions and the note says the GPU machine", async (t) => {
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 200, fixtures.limitsVideo],
      ["GET", "/api/matches", 200, fixtures.emptyMatches],
    ],
  });
  await waitFor(() => h.doc.getElementById("uploadNote").textContent.includes("GPU machine"), "the video note", 3000, h);

  const note = h.doc.getElementById("uploadNote").textContent;
  assert.ok(note.includes("Videos are processed on the GPU machine and may take several minutes"), "note: " + note);
  assert.ok(note.includes("up to 2 GB"), "note: " + note);
  const accept = h.doc.getElementById("file").getAttribute("accept");
  for (const ext of [".avi", ".mkv", ".mov", ".mp4", ".webm"]) {
    assert.ok(accept.includes(ext), "accept misses " + ext + ": " + accept);
  }
  assert.ok(accept.includes(".jsonl"), "accept keeps .jsonl: " + accept);
  // The jobs section stays hidden until a video job exists.
  assert.equal(h.doc.getElementById("jobsSection").hidden, true);
  assertNoScriptErrors(h);
  assertFixtureOnly(h, [
    { method: "GET", url: APP_ORIGIN + "/api/limits" },
    { method: "GET", url: APP_ORIGIN + "/api/matches" },
  ]);
});

test("video upload: progress, the job row polls every 3 s, done becomes an Open link", async (t) => {
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 200, fixtures.limitsVideo],
      ["GET", "/api/matches", 200, fixtures.emptyMatches], // the list at start
      ["POST", "/api/videos", 202, { job_id: JOB, status: "queued" }],
      ["GET", "/api/videos/" + JOB, 200, jobStatus(JOB, "running", { progress: 0.42, frames: 78 })],
      ["GET", "/api/videos/" + JOB, 200, jobStatus(JOB, "done", { progress: 1.0, frames: 750, seconds: 30.0, match_id: MATCH_ID })],
      ["GET", "/api/matches", 200, fixtures.matches], // the list after done
    ],
  });
  await waitFor(() => h.doc.getElementById("uploadNote").textContent.includes("GPU machine"), "the video note", 3000, h);

  pickFile(h, "clip.mp4", 4096, "Friday night");
  submitForm(h);
  await ticks(2);

  // The upload is in flight: the progress indicator is visible at 0%.
  assert.equal(h.doc.getElementById("uploadProgress").hidden, false);
  assert.equal(h.doc.getElementById("status").textContent, "Uploading clip.mp4\u2026");
  // The file (and the name) went to the worker endpoint, not the match upload.
  const upload = h.requests.find((r) => r.url.endsWith("/api/videos"));
  assert.ok(upload, "the POST /api/videos request");
  assert.equal(upload.method, "POST");
  assert.equal(upload.file, "clip.mp4");
  assert.equal(upload.name, "Friday night");

  h.clock.advance(100); // first progress event
  await ticks(2);
  assert.equal(h.doc.getElementById("status").textContent, "Uploading clip.mp4\u2026 50%");
  assert.equal(h.doc.getElementById("uploadProgressBar").style.width, "50%");

  h.clock.advance(100); // second progress event
  await ticks(2);
  assert.equal(h.doc.getElementById("status").textContent, "Uploading clip.mp4\u2026 100%");
  assert.equal(h.doc.getElementById("uploadProgressBar").style.width, "100%");

  h.clock.advance(100); // the 202 answer lands
  await ticks(3);
  // The upload is done; the job row is shown with the first polled status.
  assert.equal(h.doc.getElementById("uploadProgress").hidden, true);
  assert.equal(h.doc.getElementById("uploadBtn").disabled, false);
  const row = JOB_ROW(h);
  assert.ok(row, "the job row");
  assert.equal(row.getAttribute("data-job-id"), JOB);
  assert.equal(row.querySelector(".name").textContent, "Friday night");
  assert.equal(row.querySelector(".job-state").textContent, "Processing 42%");
  assert.equal(h.doc.getElementById("jobsSection").hidden, false);

  h.clock.advance(3000); // the 3 s poll: the job is done
  await ticks(3);
  const doneRow = JOB_ROW(h);
  assert.equal(doneRow.querySelector(".job-state").textContent, "Done");
  const open = doneRow.querySelector("a.open");
  assert.ok(open, "the Open link");
  assert.equal(open.getAttribute("href"), "/match.html?id=" + MATCH_ID);
  // The new match joined the match list.
  await waitFor(() => h.doc.querySelectorAll("#matchList .match-row").length === 1, "the match row", 3000, h);
  assert.equal(h.doc.querySelector("#matchList .match-row").getAttribute("data-match-id"), MATCH_ID);

  // No further polls after the job is done.
  const pollsBefore = h.requests.filter((r) => r.url === APP_ORIGIN + "/api/videos/" + JOB).length;
  h.clock.advance(9000);
  await ticks(3);
  const pollsAfter = h.requests.filter((r) => r.url === APP_ORIGIN + "/api/videos/" + JOB).length;
  assert.equal(pollsAfter, pollsBefore, "polling stopped after done");

  assert.deepEqual(
    h.requests.map(requestKey),
    [
      "GET " + APP_ORIGIN + "/api/limits",
      "GET " + APP_ORIGIN + "/api/matches",
      "POST " + APP_ORIGIN + "/api/videos",
      "GET " + APP_ORIGIN + "/api/videos/" + JOB,
      "GET " + APP_ORIGIN + "/api/videos/" + JOB,
      "GET " + APP_ORIGIN + "/api/matches",
    ],
  );
  assertNoScriptErrors(h);
});

test("a failed video job shows the worker's error in the job row", async (t) => {
  const ERROR = "processing failed (exit 1): the decoder gave up";
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 200, fixtures.limitsVideo],
      ["GET", "/api/matches", 200, fixtures.emptyMatches],
      ["POST", "/api/videos", 202, { job_id: JOB2, status: "queued" }],
      ["GET", "/api/videos/" + JOB2, 200, jobStatus(JOB2, "running", { progress: 0.1, frames: 25, name: "Crash clip" })],
      ["GET", "/api/videos/" + JOB2, 200, jobStatus(JOB2, "failed", { progress: 0.1, frames: 25, name: "Crash clip", error: ERROR })],
    ],
  });
  await waitFor(() => h.doc.getElementById("uploadNote").textContent.includes("GPU machine"), "the video note", 3000, h);

  pickFile(h, "crash.mp4", 2048, "Crash clip");
  submitForm(h);
  await ticks(2);
  h.clock.advance(400); // upload done + first poll
  await ticks(3);
  let row = JOB_ROW(h);
  assert.equal(row.querySelector(".name").textContent, "Crash clip");
  assert.equal(row.querySelector(".job-state").textContent, "Processing 10%");

  h.clock.advance(3000); // the 3 s poll: the job failed
  await ticks(3);
  row = JOB_ROW(h);
  assert.equal(row.querySelector(".job-state").textContent, ERROR);
  assert.equal(row.querySelector("a.open"), null, "no Open link for a failed job");

  const pollsBefore = h.requests.filter((r) => r.url === APP_ORIGIN + "/api/videos/" + JOB2).length;
  h.clock.advance(9000);
  await ticks(3);
  const pollsAfter = h.requests.filter((r) => r.url === APP_ORIGIN + "/api/videos/" + JOB2).length;
  assert.equal(pollsAfter, pollsBefore, "polling stopped after failed");

  assertFixtureOnly(h, [
    { method: "GET", url: APP_ORIGIN + "/api/limits" },
    { method: "GET", url: APP_ORIGIN + "/api/matches" },
    { method: "POST", url: APP_ORIGIN + "/api/videos" },
    { method: "GET", url: APP_ORIGIN + "/api/videos/" + JOB2 },
  ]);
  assertNoScriptErrors(h);
});

test("a worker that is down: the page shows the server's own 502 message", async (t) => {
  const ERROR = "the video processing machine is not reachable; check that the worker is running and try again";
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 200, fixtures.limitsVideo],
      ["GET", "/api/matches", 200, fixtures.emptyMatches],
      ["POST", "/api/videos", 502, { error: ERROR }],
    ],
  });
  await waitFor(() => h.doc.getElementById("uploadNote").textContent.includes("GPU machine"), "the video note", 3000, h);

  pickFile(h, "clip.mp4", 2048);
  submitForm(h);
  await ticks(2);
  h.clock.advance(400); // the 502 answer lands
  await ticks(3);

  const alert = h.doc.getElementById("alert");
  assert.equal(alert.hidden, false);
  assert.equal(alert.textContent, "Video upload failed: " + ERROR + ".");
  assert.equal(h.doc.getElementById("uploadBtn").disabled, false);
  assert.equal(h.doc.getElementById("uploadProgress").hidden, true);
  assert.equal(JOB_ROW(h), null, "no job row for a rejected upload");

  assertFixtureOnly(h, [
    { method: "GET", url: APP_ORIGIN + "/api/limits" },
    { method: "GET", url: APP_ORIGIN + "/api/matches" },
    { method: "POST", url: APP_ORIGIN + "/api/videos" },
  ]);
  assertNoScriptErrors(h);
});

test("without a worker the page behaves as before and says videos are not available", async (t) => {
  const h = await loadPage(t, {
    page: "library.html",
    url: "http://localhost/library.html",
    answers: [
      ["GET", "/api/limits", 200, fixtures.limits], // video_enabled: false
      ["GET", "/api/matches", 200, fixtures.emptyMatches],
      ["POST", "/api/matches", 400, { error: "line 1: invalid JSON" }],
    ],
  });
  await waitFor(() => h.doc.getElementById("uploadNote").textContent.includes("not available"), "the no-video note", 3000, h);

  const note = h.doc.getElementById("uploadNote").textContent;
  assert.ok(note.includes("Video processing is not available on this server"), "note: " + note);
  // The file picker is unchanged.
  assert.equal(h.doc.getElementById("file").getAttribute("accept"), "video/*,.jsonl");

  // A picked video still goes through the old match upload path and the
  // page shows the old-style error for the server's 400.
  pickFile(h, "clip.mp4", 2048, "Clip");
  submitForm(h);
  await waitFor(() => !h.doc.getElementById("alert").hidden, "the visible alert", 3000, h);

  const alert = h.doc.getElementById("alert");
  assert.equal(alert.textContent, "Upload failed: the server answered 400.");
  const toMatches = h.requests.filter((r) => r.url === APP_ORIGIN + "/api/matches" && r.method === "POST");
  assert.equal(toMatches.length, 1, "the video went to the old upload path");
  assert.equal(JOB_ROW(h), null, "no job row without a worker");
  assert.equal(h.doc.getElementById("jobsSection").hidden, true);

  assertFixtureOnly(h, [
    { method: "GET", url: APP_ORIGIN + "/api/limits" },
    { method: "GET", url: APP_ORIGIN + "/api/matches" },
    { method: "POST", url: APP_ORIGIN + "/api/matches" },
  ]);
  assertNoScriptErrors(h);
});

// Safety net for a rejection that lands after the page that caused it was
// already closed (the settle wait in loadPage's after hook covers the
// realistic window; this catches anything later than that).
test("no unhandled promise rejection outlived its page", () => {
  assert.deepEqual(lateRejections, [], "unhandled promise rejections with no open page: " + JSON.stringify(lateRejections));
});
