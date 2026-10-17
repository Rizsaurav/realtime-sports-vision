// Extra adversarial tests for the security-header hardening of the pages
// (tasks/service_hardening.md, item 7: "The pages must still work: check
// that no page uses inline event handlers or inline scripts that the
// policy would block").
//
// The Python suite asserts the headers the server sends. These tests
// check the other side of the contract, against the page sources the
// server serves byte-for-byte from src/sideline/web:
//
// - no inline event handler attributes on any element of either page;
// - each page has exactly one inline <script> - the module script the
//   per-page CSP hash pins - and no other inline scripts, so the policy
//   (script-src 'self' plus one sha256 hash) blocks nothing the page
//   needs; every other <script src> is a same-origin /static/ file that
//   exists, and no src= in the page points at a remote origin;
// - no <style> elements (the style-src carries no 'unsafe-inline'), and
//   the same-origin stylesheets the pages link really exist;
// - any <img> uses only same-origin or data: sources (img-src 'self'
//   data:);
// - the page scripts (the inline module plus the /static/lib modules it
//   imports) never assign event handlers, never set on* attributes and
//   never document.write a script;
// - the library page still runs and renders under the setup the policy
//   assumes: its own single inline module, same-origin libraries only,
//   answered from the recorded API fixtures, with no script errors.

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
const PAGES = ["library.html", "match.html"];

function readPage(name) {
  return fs.readFileSync(path.join(WEB_DIR, name), "utf8");
}

// -- static checks over the page sources -------------------------------------

test("neither page uses inline event handler attributes", () => {
  for (const page of PAGES) {
    const doc = new JSDOM(readPage(page), { url: APP_ORIGIN + "/" + page })
      .window.document;
    const handlerAttrs = [];
    for (const el of doc.querySelectorAll("*")) {
      for (const attr of el.attributes) {
        if (/^on[a-z]+$/i.test(attr.name)) {
          handlerAttrs.push(el.tagName + "." + attr.name);
        }
      }
    }
    assert.deepEqual(
      handlerAttrs,
      [],
      page + " inline event handler attributes: " + handlerAttrs,
    );
  }
});

test("each page has exactly one inline script (the pinned module) and no remote scripts", () => {
  for (const page of PAGES) {
    const html = readPage(page);
    const tags = [...html.matchAll(/<script\b[^>]*>/g)].map((m) => m[0]);
    assert.ok(tags.length >= 1, page + " has scripts");
    const inline = tags.filter((t) => !/\bsrc\s*=/i.test(t));
    assert.equal(
      inline.length,
      1,
      page + " inline script tags (the CSP pins exactly one): " + JSON.stringify(inline),
    );
    assert.ok(
      /type\s*=\s*["']module["']/.test(inline[0]),
      page + " the one inline script is the module script",
    );
    const body = html.match(/<script type="module">([\s\S]*?)<\/script>/);
    assert.ok(body && body[1].trim().length > 0, page + " the module script has content");
    for (const tag of tags) {
      if (!/\bsrc\s*=/i.test(tag)) continue;
      const src = tag.match(/\bsrc\s*=\s*["']([^"']+)["']/i)[1];
      // every external script is a same-origin file served under /static
      assert.ok(src.startsWith("/static/"), page + " script src is same-origin: " + src);
      const file = path.resolve(WEB_DIR, src.slice("/static/".length));
      assert.ok(
        file.startsWith(WEB_DIR + path.sep) && fs.existsSync(file),
        page + " script file exists: " + src,
      );
    }
    // nothing in the page loads a script from a remote origin
    assert.ok(
      !/src\s*=\s*["']http/i.test(html),
      page + " has a remote src attribute",
    );
  }
});

test("the pages have no <style> elements and their same-origin stylesheets exist", () => {
  for (const page of PAGES) {
    const doc = new JSDOM(readPage(page), { url: APP_ORIGIN + "/" + page })
      .window.document;
    assert.equal(
      doc.querySelectorAll("style").length,
      0,
      page + " has inline <style> elements the policy would block",
    );
    const hrefs = [...doc.querySelectorAll('link[rel="stylesheet"]')].map(
      (l) => l.getAttribute("href"),
    );
    assert.ok(hrefs.length > 0, page + " links its stylesheets");
    const sameOrigin = hrefs.filter((h) => h && h.startsWith("/static/"));
    assert.ok(
      sameOrigin.length > 0,
      page + " loads same-origin stylesheets (style-src 'self'): " + JSON.stringify(hrefs),
    );
    for (const href of sameOrigin) {
      const file = path.resolve(WEB_DIR, href.slice("/static/".length));
      assert.ok(
        file.startsWith(WEB_DIR + path.sep) && fs.existsSync(file),
        page + " stylesheet file exists: " + href,
      );
    }
  }
});

test("every <img> on the pages uses a same-origin or data: source", () => {
  for (const page of PAGES) {
    const doc = new JSDOM(readPage(page), { url: APP_ORIGIN + "/" + page })
      .window.document;
    for (const img of doc.querySelectorAll("img")) {
      const src = img.getAttribute("src") || "";
      assert.ok(
        src === "" || src.startsWith("/") || src.startsWith("data:"),
        page + " img src is same-origin or data: (img-src 'self' data:): " + src,
      );
    }
  }
});

const IMPORT_RE = /import\s*\{([^}]*)\}\s*from\s*["']([^"']+)["']\s*;?/g;

test("the page scripts never assign event handlers or write scripts", () => {
  for (const page of PAGES) {
    const html = readPage(page);
    const code = html.match(/<script type="module">([\s\S]*?)<\/script>/);
    assert.ok(code, page + " has its inline module script");
    const texts = new Map([[page, html]]);
    for (const im of code[1].matchAll(IMPORT_RE)) {
      const src = im[2];
      assert.ok(src.startsWith("/static/"), page + " module source: " + src);
      const file = path.join(WEB_DIR, src.slice("/static/".length));
      texts.set(src, fs.readFileSync(file, "utf8"));
    }
    assert.ok(texts.size > 1, page + " imports its libraries from /static");
    for (const [name, text] of texts) {
      assert.ok(!/\.on[a-z]+\s*=[^=]/i.test(text), name + " assigns an event handler");
      assert.ok(
        !/setAttribute\(\s*["'`]on[a-z]+["'`]/i.test(text),
        name + " sets an on* attribute",
      );
      assert.ok(!/document\s*\.\s*write/i.test(text), name + " uses document.write");
    }
  }
});

// -- the library page still works under the policy's assumptions -------------

const fixtures = {
  limits: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "limits.json"), "utf8")),
  matches: JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "matches.json"), "utf8")),
};

function staticFileFor(urlPath) {
  if (!urlPath.startsWith("/static/")) return null;
  const file = path.resolve(WEB_DIR, urlPath.slice("/static/".length));
  const inside = file === WEB_DIR || file.startsWith(WEB_DIR + path.sep);
  if (!inside || !fs.existsSync(file) || !fs.statSync(file).isFile()) return null;
  return file;
}

// The page's fetch: answers the two init requests from the fixtures,
// records every request, and fails loudly on anything else.
function makeFetch({ answers, requests }) {
  return function fetch(url, init = {}) {
    const u = new URL(url, APP_ORIGIN);
    requests.push({ method: String(init.method || "GET").toUpperCase(), url: u.href });
    const key = u.pathname + u.search;
    const row = answers.find((r) => !r._used && r[1] === key);
    if (!row) {
      return Promise.resolve(new Response({ detail: "no fixture for " + key }, { status: 500 }));
    }
    row._used = true;
    return Promise.resolve(new Response(JSON.stringify(row[3]), { status: row[2] }));
  };
}

const lateRejections = [];
const openPages = [];
process.on("unhandledRejection", (reason) => {
  const detail = reason && (reason.stack || reason.message) ? String(reason.stack || reason.message) : String(reason);
  const owner = openPages[openPages.length - 1];
  if (!owner) {
    lateRejections.push("unhandled promise rejection: " + detail);
    return;
  }
  owner.errors.push("unhandled promise rejection: " + detail);
});

const tick = () => new Promise((r) => setImmediate(r));
async function ticks(count) {
  for (let i = 0; i < count; i++) await tick();
}

// Run the library page's own inline module (rewritten to a classic script
// importing the real same-origin libraries, as the smoke suite does) and
// let it do its init: fetch /api/limits and /api/matches, render the list.
async function runLibraryPage(t) {
  const html = readPage("library.html");
  const code = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
  const modules = {};
  for (const im of code.matchAll(IMPORT_RE)) {
    const src = im[2];
    const file = staticFileFor(src);
    assert.ok(file, "module source maps to a file: " + src);
    modules[src] = await import(pathToFileURL(file).href);
  }
  const rewritten = code.replace(
    IMPORT_RE,
    (whole, names, src) =>
      "const { " +
      names.split(",").map((s) => s.trim()).filter(Boolean).join(", ") +
      " } = globalThis.__hardeningModules[" + JSON.stringify(src) + "];",
  );
  IMPORT_RE.lastIndex = 0;

  const requests = [];
  const errors = [];
  const answers = [
    ["GET", "/api/limits", 200, fixtures.limits],
    ["GET", "/api/matches", 200, fixtures.matches],
  ];
  const virtualConsole = new VirtualConsole();
  const benign = [/Could not parse CSS/, /HTMLCanvasElement\.prototype\.getContext/];
  virtualConsole.on("jsdomError", (err) => {
    const msg = String((err && err.message) || err);
    if (benign.some((re) => re.test(msg))) return;
    errors.push(msg);
  });
  const dom = new JSDOM(
    html.replace(
      /<script type="module">[\s\S]*?<\/script>/,
      () => "<script>" + rewritten + "</script>",
    ),
    {
      url: APP_ORIGIN + "/library.html",
      runScripts: "dangerously",
      resources: new (class extends ResourceLoader {
        fetch(url) {
          const u = new URL(url);
          if (u.hostname !== "localhost" && u.hostname !== "127.0.0.1") {
            return Promise.resolve(Buffer.from("/* offline: no external assets */\n"));
          }
          const file = staticFileFor(u.pathname);
          if (!file) return Promise.reject(new Error("no local file for " + u.pathname));
          return Promise.resolve(fs.readFileSync(file));
        }
      })(),
      virtualConsole,
      beforeParse(window) {
        window.addEventListener("error", (e) =>
          errors.push(String(e.message || (e.error && e.error.message) || e)),
        );
        window.__hardeningModules = modules;
        window.fetch = makeFetch({ answers, requests });
        // The page only uses timers for job polling (no jobs are created
        // here); keep the run deterministic by scheduling nothing.
        window.setTimeout = () => 0;
        window.setInterval = () => 0;
        window.clearTimeout = () => {};
        window.clearInterval = () => {};
      },
    },
  );
  const handle = {
    doc: dom.window.document,
    requests,
    errors,
    close: () => dom.window.close(),
  };
  openPages.push(handle);
  t.after(async () => {
    await new Promise((r) => setTimeout(r, 100));
    if (errors.length) throw new Error("page errors after the checks: " + JSON.stringify(errors));
    openPages.splice(openPages.indexOf(handle), 1);
    handle.close();
  });
  await ticks(8);
  return handle;
}

test("the library page still works: its own module, same-origin libraries, no script errors", async (t) => {
  const h = await runLibraryPage(t);
  assert.deepEqual(h.errors, [], "script errors: " + JSON.stringify(h.errors));
  // the init requests, and only those, in the app's routes
  assert.deepEqual(
    h.requests.map((r) => r.method + " " + r.url),
    ["GET " + APP_ORIGIN + "/api/limits", "GET " + APP_ORIGIN + "/api/matches"],
  );
  // the match list rendered from the fixture
  const rows = h.doc.querySelectorAll("#matchList .match-row");
  assert.equal(rows.length, 1, "one fixture match rendered");
  assert.equal(rows[0].getAttribute("data-match-id"), fixtures.matches.items[0].id);
  // the upload form is intact (the page still works for the user)
  assert.ok(h.doc.getElementById("uploadForm"), "the upload form");
  assert.ok(h.doc.getElementById("file"), "the file picker");
  assert.equal(h.doc.getElementById("uploadBtn").disabled, false);
});

// Safety net for a rejection that lands after the page that caused it was
// already closed.
test("no unhandled promise rejection outlived its page", () => {
  assert.deepEqual(lateRejections, [], "unhandled promise rejections with no open page: " + JSON.stringify(lateRejections));
});
