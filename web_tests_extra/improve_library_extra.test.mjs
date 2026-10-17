// Extra adversarial tests for the match library task (tasks/library_server.md),
// JS side.
//
// The library server's whole job is to serve the two pages together with the
// web folder mounted at /static, "so /static/lib/render.js and
// /static/lib/replay.js work".  A live socket is not needed (and is not
// available in this sandbox) to check the JS half of that contract: the pages
// must be HTML documents, everything they reference under /static must exist
// in the served folder, and every module they import must actually export the
// names they import.  The serving itself (status codes, content types, the
// 307 redirect, the API) is covered by the Python TestClient suite.

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = process.env.LIBRARY_WEB_DIR
  ? path.resolve(process.env.LIBRARY_WEB_DIR)
  : path.resolve(here, "../src/sideline/web");

const PAGES = ["library.html", "match.html"];

function readPage(name) {
  return fs.readFileSync(path.join(webDir, name), "utf8");
}

// All "/static/..." references in a page (link/script hrefs and import
// specifiers), de-duplicated, in order of first appearance.
function staticRefs(pageText) {
  const seen = new Set();
  for (const m of pageText.matchAll(/\/static\/[A-Za-z0-9._/-]+/g)) {
    seen.add(m[0]);
  }
  return [...seen];
}

// All `import { a, b as c } from "/static/lib/x.js"` statements in a page.
function importStatements(pageText) {
  const out = [];
  const re = /import\s*\{([\s\S]*?)\}\s*from\s*"\/static\/lib\/([\w.-]+)"/g;
  for (const m of pageText.matchAll(re)) {
    const names = m[1].split(",").map((s) => s.trim().split(/\s+as\s+/)[0].trim()).filter(Boolean);
    out.push({ names, file: m[2] });
  }
  return out;
}

test("both pages are HTML documents in the served web folder", () => {
  for (const name of PAGES) {
    const p = path.join(webDir, name);
    assert.ok(fs.existsSync(p), `${name} must exist in the web folder`);
    const text = readPage(name);
    assert.match(text.trimStart().toLowerCase(), /^<!doctype html/, `${name} is an HTML document`);
    assert.ok(text.toLowerCase().includes("<html"), `${name} has an <html> element`);
    assert.ok(text.includes('<script type="module">'), `${name} uses a module script`);
  }
});

test("the pages load the libraries from /static", () => {
  // tasks/library_server.md: the folder is served under /static "so
  // /static/lib/render.js and /static/lib/replay.js work".
  for (const lib of ["lib/render.js", "lib/replay.js"]) {
    assert.ok(fs.statSync(path.join(webDir, lib)).isFile(), `/static/${lib} must exist`);
  }
  const library = readPage("library.html");
  const match = readPage("match.html");
  assert.ok(library.includes('"/static/lib/render.js"'), "library.html imports /static/lib/render.js");
  assert.ok(match.includes('"/static/lib/render.js"'), "match.html imports /static/lib/render.js");
  assert.ok(match.includes('"/static/lib/replay.js"'), "match.html imports /static/lib/replay.js");
});

test("every /static/ reference in the pages resolves to a file in the web folder", () => {
  let refs = 0;
  for (const name of PAGES) {
    const list = staticRefs(readPage(name));
    assert.ok(list.length >= 3, `${name} should reference several /static assets`);
    for (const ref of list) {
      refs += 1;
      // /static is the mount point of the web folder itself.
      const target = path.join(webDir, ref.slice("/static".length).replace(/^\/+/, ""));
      assert.ok(fs.existsSync(target), `${name} references /static that the folder does not contain: ${ref}`);
      assert.ok(fs.statSync(target).isFile(), `${ref} must be a file, not a directory`);
    }
  }
  assert.ok(refs >= 6);
});

test("the static modules export every name the pages import", async () => {
  const seen = new Map(); // file -> Set(names)
  for (const name of PAGES) {
    for (const { names, file } of importStatements(readPage(name))) {
      assert.ok(names.length > 0, `import in ${name} from /static/lib/${file} has names`);
      if (!seen.has(file)) seen.set(file, new Set());
      for (const n of names) seen.get(file).add(n);
    }
  }
  assert.ok(seen.has("render.js"), "at least one page imports /static/lib/render.js");
  assert.ok(seen.has("replay.js"), "at least one page imports /static/lib/replay.js");
  for (const [file, names] of seen) {
    const mod = await import(pathToFileURL(path.join(webDir, "lib", file)).href);
    for (const n of names) {
      assert.ok(n in mod, `/static/lib/${file} must export "${n}" (imported by a page)`);
      assert.notEqual(typeof mod[n], "undefined", `"${n}" in /static/lib/${file} must be a value`);
    }
  }
});
