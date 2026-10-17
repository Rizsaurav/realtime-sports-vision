import test from "node:test";
import assert from "node:assert/strict";

const lib = await import(
  process.env.PITCH_LIB ? "file://" + process.env.PITCH_LIB : "../src/sideline/web/lib/pitch.js"
);
const { PITCH, pitchGeometry, pitchToCanvas, canvasToPitch, personStyle, chunkRanges, chunkStartFor, FrameCache } = lib;

const fr = (n) => ({ frame: n, people: [], ball: null, colors: null });
const G = { k: 10, ox: 565, oy: 380 };
const COLORS = { left: "#d03030", right: "#2050c0" };
const EDGE = "rgba(0,0,0,0.55)";

// Adversarial extras (second pass) for tasks/replay_view.md part 1. Targets
// spec rules the base suite (web_tests/pitch.test.mjs) and the first extra
// pass (web_tests_extra/improve_pitch_extra.test.mjs) do not check:
//  - pitchGeometry (0, 0) and the k = 1 boundary (metres map 1:1);
//  - personStyle: role matching is exact (a capitalised "Referee" or
//    "Goalkeeper" gets NO special fill or outline, per the spec's
//    `role === "referee"` / `role === "goalkeeper"` rules), team values that
//    are not exactly "left"/"right" are grey, and label is String(jersey)
//    for the remaining odd values (" " stays " ", -5 becomes "-5", [] is "");
//  - chunkRanges: a large off-grid range starting negative covers every
//    frame exactly once, a small all-negative block, and string first/last
//    throwing RangeError (integers only, coercion not allowed);
//  - chunkStartFor: frameNumber is NOT validated (a string is coerced by
//    the formula, no throw) and huge frame numbers stay exact;
//  - FrameCache: constructor rejects string/boolean/infinite/NaN/-0 values
//    for both parameters, chunkSize 1 and the default 250 bound exactly,
//    put(-0, ...) is a valid integer start sharing Map key 0,
//    frameAt(-Infinity)/1e999 give null, a stored frame beyond the covered
//    range is never returned for an in-range query, negative frame numbers
//    inside a negative chunk, a long deterministic LRU walk-through, and
//    clear() on an empty cache plus reuse after clear.
// Every part-1 function is synchronous, so there are no concurrency
// effects; the ordering effects are the LRU rules, exercised below.

// ---------------------------------------------------------------- pitchGeometry
test("pitchGeometry: (0, 0) gives {k: 0, ox: 0, oy: 0}", () => {
  assert.deepEqual(pitchGeometry(0, 0), { k: 0, ox: 0, oy: 0 });
});
test("pitchToCanvas: k = 1 geometry maps metres one to one", () => {
  const g = pitchGeometry(113, 76); // exactly the pitch+margin: k 1, ox 56.5, oy 38
  assert.deepEqual(g, { k: 1, ox: 56.5, oy: 38 });
  assert.deepEqual(pitchToCanvas([1, 2], g), [57.5, 40]);
  assert.deepEqual(pitchToCanvas([-52.5, 34], g), [4, 72]);
  assert.deepEqual(pitchToCanvas([52.5, -34], g), [109, 4]);
});
test("canvasToPitch: k = 1 inverse maps back exactly", () => {
  const g = pitchGeometry(113, 76);
  assert.deepEqual(canvasToPitch([4, 72], g), [-52.5, 34]);
  assert.deepEqual(canvasToPitch([57.5, 40], g), [1, 2]);
  assert.deepEqual(canvasToPitch(pitchToCanvas([33.33, -11.11], g), g), [33.33, -11.11]);
});

// ---------------------------------------------------------------- personStyle
test("personStyle: role matching is exact (capitalised roles are not special)", () => {
  // Spec rule 1 triggers only on role === "referee"; "Referee" falls through
  // to the team rule, and spec's outline rule triggers only on
  // role === "goalkeeper", so "Goalkeeper" gets the default outline.
  const s1 = personStyle({ role: "Referee", team: "left", jersey: "4" }, COLORS);
  assert.deepEqual(s1, { fill: "#d03030", stroke: EDGE, strokeWidth: 1.5, label: "4" });
  const s2 = personStyle({ role: "Goalkeeper", team: "right", jersey: "1" }, COLORS);
  assert.deepEqual(s2, { fill: "#2050c0", stroke: EDGE, strokeWidth: 1.5, label: "1" });
  const s3 = personStyle({ role: "REFEREE" }, COLORS);
  assert.equal(s3.fill, "#9aa59f");
  assert.equal(s3.stroke, EDGE);
  assert.equal(s3.strokeWidth, 1.5);
});
test("personStyle: team values that are not exactly 'left' or 'right' are grey", () => {
  // Spec rule 3: "Anything else (team null, missing, or any other value...)".
  assert.equal(personStyle({ team: "" }, null).fill, "#9aa59f");
  assert.equal(personStyle({ team: " " }, null).fill, "#9aa59f");
  assert.equal(personStyle({ team: 0 }, null).fill, "#9aa59f");
  assert.equal(personStyle({ team: "left " }, COLORS).fill, "#9aa59f");
  assert.equal(personStyle({ team: "LEFT" }, COLORS).fill, "#9aa59f");
});
test("personStyle: label is String(jersey) for the remaining odd values", () => {
  // Spec: label is "" only when jersey is null, undefined or ""; otherwise
  // String(person.jersey) - so a whitespace string survives, numbers are
  // stringified, and String([]) is "".
  assert.equal(personStyle({ jersey: " " }, null).label, " ");
  assert.equal(personStyle({ jersey: -5 }, null).label, "-5");
  assert.equal(personStyle({ jersey: [] }, null).label, "");
  assert.equal(personStyle({ role: "referee", jersey: 100 }, null).label, "100");
});
test("personStyle: an empty colours object falls back per side to the built-ins", () => {
  assert.equal(personStyle({ team: "left" }, {}).fill, "#e4572e");
  assert.equal(personStyle({ team: "right" }, {}, {}).fill, "#3e8ed0");
  // usable colours still win over an empty fallback
  assert.equal(personStyle({ team: "left" }, { left: "#abcdef" }, {}).fill, "#abcdef");
});

// ---------------------------------------------------------------- chunkRanges
test("chunkRanges: large off-grid range from a negative first covers every frame once", () => {
  const r = chunkRanges(-1000, 1000000, 250);
  // starts: -1000 + i*250 while <= 1000000 -> i = 0..4004 -> 4005 blocks
  assert.equal(r.length, 4005);
  assert.deepEqual(r[0], { start: -1000, end: -751 });
  assert.deepEqual(r[1], { start: -750, end: -501 });
  assert.deepEqual(r.at(-1), { start: 1000000, end: 1000000 }); // single frame
  for (let i = 1; i < r.length; i++) assert.equal(r[i].start, r[i - 1].end + 1);
  for (const b of r.slice(0, -1)) assert.equal(b.end - b.start + 1, 250);
});
test("chunkRanges: a single small all-negative block", () => {
  assert.deepEqual(chunkRanges(-5, -3, 10), [{ start: -5, end: -3 }]);
  assert.deepEqual(chunkRanges(-3, -3, 1), [{ start: -3, end: -3 }]);
});
test("chunkRanges: string first or last throws RangeError (no coercion)", () => {
  assert.throws(() => chunkRanges("0", 10, 5), RangeError);
  assert.throws(() => chunkRanges(0, "10", 5), RangeError);
  assert.throws(() => chunkRanges(null, 10, 5), RangeError);
  assert.throws(() => chunkRanges(0, undefined, 5), RangeError);
});

// ---------------------------------------------------------------- chunkStartFor
test("chunkStartFor: frameNumber is not validated (strings coerce via the formula)", () => {
  // Spec: "size and first are validated like in chunkRanges (RangeError);
  // frameNumber is not validated." The formula is applied unchanged, so JS
  // coercion happens and nothing throws.
  assert.doesNotThrow(() => chunkStartFor("250", 0, 250));
  assert.equal(chunkStartFor("250", 0, 250), 250);
  assert.equal(chunkStartFor("50", 100, 250), -150); // "50" - 100 = -50 -> floor(-0.2) = -1
  assert.equal(chunkStartFor("99", 100, 250), -150);
});
test("chunkStartFor: huge frame numbers stay exact", () => {
  // 1e15 is divisible by 250 and all values involved are < 2^53.
  assert.equal(chunkStartFor(1e15, 0, 250), 1e15);
  assert.equal(chunkStartFor(1e15 + 249, 0, 250), 1e15);
  assert.equal(chunkStartFor(1e15 + 250, 0, 250), 1e15 + 250);
  // (n - 100) = 1e15 + 149; floor((1e15 + 149) / 250) * 250 = 1e15, then + first
  assert.equal(chunkStartFor(1e15 + 249, 100, 250), 1e15 + 100);
});

// ---------------------------------------------------------------- FrameCache
test("FrameCache: constructor rejects string, boolean, infinite, NaN and -0 values", () => {
  for (const bad of ["2", true, -0, NaN, Infinity, -Infinity]) {
    assert.throws(() => new FrameCache(bad), RangeError, `maxChunks ${String(bad)}`);
    assert.throws(() => new FrameCache(2, bad), RangeError, `chunkSize ${String(bad)}`);
  }
  // chunkSize 1: a chunk covers exactly its start
  const c1 = new FrameCache(2, 1);
  c1.put(5, [fr(5)]);
  assert.equal(c1.frameAt(5).frame, 5);
  assert.equal(c1.frameAt(6), null);
  // default chunkSize is 250: covers 0..249, not 250
  const c2 = new FrameCache(1);
  c2.put(0, [fr(0), fr(249)]);
  assert.equal(c2.frameAt(249).frame, 249);
  assert.equal(c2.frameAt(250), null);
});
test("FrameCache: put(-0, ...) is a valid integer start sharing Map key 0", () => {
  const c = new FrameCache(2);
  assert.doesNotThrow(() => c.put(-0, [fr(0)]));
  assert.equal(c.has(0), true); // Map keys use SameValueZero: -0 and 0 are one key
  assert.equal(c.has(-0), true);
  assert.equal(c.size, 1);
  assert.equal(c.starts().length, 1);
  assert.equal(c.frameAt(0).frame, 0);
});
test("FrameCache: frameAt(-Infinity) and 1e999 (which is Infinity) give null", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(0)]);
  assert.equal(c.frameAt(-Infinity), null);
  assert.equal(c.frameAt(1e999), null);
  assert.equal(c.frameAt(-1e999), null);
  // and a miss changes nothing
  assert.deepEqual(c.starts(), [0]);
});
test("FrameCache: a stored frame beyond the covered range is never returned", () => {
  const c = new FrameCache(2);
  const a = [fr(0), fr(249), fr(250)]; // 250 is stored but outside 0..249
  c.put(0, a);
  assert.equal(c.frameAt(249).frame, 249); // last frame <= 249, skipping the stray 250
  assert.equal(c.frameAt(250), null);       // outside the covered range
  assert.equal(c.frameAt(249.5), null);     // 249.5 > 0 + 250 - 1
});
test("FrameCache: negative frame numbers inside a negative chunk", () => {
  const c = new FrameCache(2);
  const a = [fr(-250), fr(-100), fr(-1)]; // chunk -250 covers -250..-1
  c.put(-250, a);
  assert.equal(c.frameAt(-150), a[0]);  // last frame <= -150 is -250
  assert.equal(c.frameAt(-100), a[1]);
  assert.equal(c.frameAt(-99), a[1]);   // -1 is greater than -99
  assert.equal(c.frameAt(-251), null);  // below the covered range
  assert.equal(c.frameAt(0), null);     // above the covered range
});
test("FrameCache: long LRU walk-through is deterministic across repeated runs", () => {
  function scenario() {
    const c = new FrameCache(2);
    const a = [fr(0), fr(10)];
    const b = [fr(250)];
    c.put(0, a);
    c.put(250, b);
    assert.deepEqual(c.starts(), [0, 250]);
    assert.equal(c.frameAt(10), a[1]); // touches chunk 0
    assert.equal(c.frameAt(10), a[1]); // repeated: same reference, still MRU
    assert.deepEqual(c.starts(), [250, 0]);
    assert.equal(c.frameAt(1000), null); // miss: no touch
    assert.deepEqual(c.starts(), [250, 0]);
    assert.equal(c.frameAt(300).frame, 250); // chunk 250 covers 300 -> touched
    assert.deepEqual(c.starts(), [0, 250]);
    c.put(500, [fr(500)]); // evicts chunk 0
    assert.deepEqual(c.starts(), [250, 500]);
    assert.equal(c.has(0), false);
    c.put(250, b); // replace existing start: touched, size unchanged, no eviction
    assert.equal(c.size, 2);
    assert.deepEqual(c.starts(), [500, 250]);
    c.put(750, [fr(750)]); // evicts chunk 500
    assert.deepEqual(c.starts(), [250, 750]);
    return { starts: c.starts(), size: c.size };
  }
  assert.deepEqual(scenario(), scenario());
  assert.deepEqual(scenario(), { starts: [250, 750], size: 2 });
});
test("FrameCache: clear on an empty cache, and reuse after clear", () => {
  const c = new FrameCache(2);
  assert.equal(c.clear(), undefined);
  assert.equal(c.size, 0);
  assert.deepEqual(c.starts(), []);
  c.put(0, [fr(0)]);
  c.put(250, [fr(250)]);
  assert.equal(c.clear(), undefined);
  assert.equal(c.size, 0);
  assert.equal(c.has(0), false);
  assert.equal(c.frameAt(0), null);
  // the cache is fully usable after a clear
  c.put(0, [fr(7)]);
  assert.equal(c.size, 1);
  assert.equal(c.frameAt(7).frame, 7);
  assert.equal(c.frameAt(3), null); // no frame <= 3 in the chunk
});
