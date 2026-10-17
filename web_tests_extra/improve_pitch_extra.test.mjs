import test from "node:test";
import assert from "node:assert/strict";

const lib = await import(
  process.env.PITCH_LIB ? "file://" + process.env.PITCH_LIB : "../src/sideline/web/lib/pitch.js"
);
const { PITCH, pitchGeometry, pitchToCanvas, canvasToPitch, personStyle, chunkRanges, chunkStartFor, FrameCache } = lib;

const close = (actual, expected, eps = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= eps, `expected ${actual} to be within ${eps} of ${expected}`);
const closeXY = (actual, expected, eps = 1e-9) => {
  assert.ok(Array.isArray(actual) && actual.length === 2, "expected a 2-element array");
  close(actual[0], expected[0], eps);
  close(actual[1], expected[1], eps);
};
const fr = (n) => ({ frame: n, people: [], ball: null, colors: null });
const G = { k: 10, ox: 565, oy: 380 };
const COLORS = { left: "#d03030", right: "#2050c0" };
const EDGE = "rgba(0,0,0,0.55)";

// Adversarial extras for tasks/replay_view.md part 1. Covers spec rules the
// base suite (web_tests/pitch.test.mjs) does not check: module export shape,
// PITCH being frozen, non-numeric/negative geometry inputs, exact-limit and
// extreme canvas sizes, the new-array rule of canvasToPitch, whitespace and
// non-string "usable" colours, exact team matching, numeric/bool jerseys,
// all-negative and size-boundary chunkRanges, non-finite/negative frame
// numbers for chunkStartFor, three-way chunk overlap, the covered-range
// boundary start + chunkSize - 1 for fractional queries, negative chunk
// starts, empty-chunk eviction accounting, and large repeated workloads.
// The part-1 functions are all synchronous and pure, so there are no
// concurrency effects to test; ordering effects are the LRU rules, which are
// exercised throughout the FrameCache tests below.

// ---------------------------------------------------------------- module
test("module: exactly the eight named exports, no default", () => {
  assert.deepEqual(Object.keys(lib).sort(), [
    "FrameCache", "PITCH", "canvasToPitch", "chunkRanges", "chunkStartFor",
    "personStyle", "pitchGeometry", "pitchToCanvas",
  ]);
  assert.equal(lib.default, undefined);
});

// ---------------------------------------------------------------- PITCH
test("PITCH is frozen", () => {
  assert.ok(Object.isFrozen(PITCH), "PITCH must be Object.freeze(d)");
});

// ---------------------------------------------------------------- pitchGeometry
test("pitchGeometry: negative width or height is treated as 0", () => {
  assert.deepEqual(pitchGeometry(1130, -1), { k: 0, ox: 565, oy: 0 });
  assert.deepEqual(pitchGeometry(500, -Infinity), { k: 0, ox: 250, oy: 0 });
  assert.deepEqual(pitchGeometry(-Infinity, 760), { k: 0, ox: 0, oy: 380 });
  assert.deepEqual(pitchGeometry(-1, -2), { k: 0, ox: 0, oy: 0 });
});
test("pitchGeometry: non-number width or height is treated as 0", () => {
  assert.deepEqual(pitchGeometry("1130", 760), { k: 0, ox: 0, oy: 380 });
  assert.deepEqual(pitchGeometry(1130, "760"), { k: 0, ox: 565, oy: 0 });
  assert.deepEqual(pitchGeometry(true, 0), { k: 0, ox: 0, oy: 0 });
  assert.deepEqual(pitchGeometry(null, 76), { k: 0, ox: 0, oy: 38 });
  assert.deepEqual(pitchGeometry({}, 76), { k: 0, ox: 0, oy: 38 });
});
test("pitchGeometry: exactly equal limits give k 1", () => {
  assert.deepEqual(pitchGeometry(113, 76), { k: 1, ox: 56.5, oy: 38 });
});
test("pitchGeometry: tiny canvases, single axis limiting", () => {
  const gw = pitchGeometry(113, 1);
  close(gw.k, 1 / 76);
  assert.equal(gw.ox, 56.5);
  assert.equal(gw.oy, 0.5);
  const gh = pitchGeometry(1, 76);
  close(gh.k, 1 / 113);
  assert.equal(gh.ox, 0.5);
  assert.equal(gh.oy, 38);
});
test("pitchGeometry: large canvas keeps the exact rule", () => {
  assert.deepEqual(pitchGeometry(113000, 76000), { k: 1000, ox: 56500, oy: 38000 });
});
test("pitchGeometry: result is a plain object", () => {
  const g = pitchGeometry(100, 100);
  assert.equal(Object.getPrototypeOf(g), Object.prototype);
});

// ---------------------------------------------------------------- pitchToCanvas / canvasToPitch
test("pitchToCanvas: zero scale collapses every point to [ox, oy]", () => {
  closeXY(pitchToCanvas([12.5, -3], { k: 0, ox: 5, oy: 7 }), [5, 7]);
  const g = pitchGeometry(0, 500); // k 0, ox 0, oy 250
  closeXY(pitchToCanvas([99, -99], g), [0, 250]);
});
test("pitchToCanvas: large coordinates, no rounding or clipping", () => {
  closeXY(pitchToCanvas([1e6, -1e6], G), [565 + 1e7, 380 - 1e7]);
});
test("canvasToPitch: returns a new array (k !== 0)", () => {
  const q = [1090, 720];
  const out = canvasToPitch(q, G);
  assert.notEqual(out, q);
  assert.notEqual(out, canvasToPitch(q, G));
  assert.deepEqual(q, [1090, 720]); // input untouched
  out[0] = 0;
  assert.deepEqual(q, [1090, 720]);
});
test("canvasToPitch: negative canvas point", () => {
  closeXY(canvasToPitch([40, 40], G), [-52.5, -34]);
});

// ---------------------------------------------------------------- personStyle
test("personStyle: whitespace-only strings are usable colours", () => {
  assert.equal(personStyle({ team: "left" }, { left: " " }).fill, " ");
  assert.equal(personStyle({ team: "right" }, null, { right: "  " }).fill, "  ");
});
test("personStyle: non-string usable-looking values are not usable", () => {
  assert.equal(personStyle({ team: "left" }, null, { left: 0 }).fill, "#e4572e");
  assert.equal(personStyle({ team: "left" }, null, { left: true }).fill, "#e4572e");
  assert.equal(personStyle({ team: "left" }, { left: "" }, { left: 123 }).fill, "#e4572e");
  assert.equal(personStyle({ team: "right" }, { right: null }, { right: 1 }).fill, "#3e8ed0");
});
test("personStyle: team match is exact (case and whitespace sensitive)", () => {
  assert.equal(personStyle({ team: "Left" }, COLORS).fill, "#9aa59f");
  assert.equal(personStyle({ team: "RIGHT" }, COLORS).fill, "#9aa59f");
  assert.equal(personStyle({ team: "right " }, COLORS).fill, "#9aa59f");
});
test("personStyle: numeric and boolean jerseys become their String()", () => {
  assert.equal(personStyle({ jersey: 0 }, null).label, "0");
  assert.equal(personStyle({ jersey: 3.5 }, null).label, "3.5");
  assert.equal(personStyle({ jersey: false }, null).label, "false");
});
test("personStyle: goalkeeper keeps its jersey label", () => {
  const s = personStyle({ role: "goalkeeper", team: "left", jersey: 1 }, COLORS);
  assert.deepEqual(s, { fill: "#d03030", stroke: "#ffffff", strokeWidth: 2.5, label: "1" });
});

// ---------------------------------------------------------------- chunkRanges
test("chunkRanges: all-negative range", () => {
  assert.deepEqual(chunkRanges(-10, -1, 3), [
    { start: -10, end: -8 }, { start: -7, end: -5 }, { start: -4, end: -2 }, { start: -1, end: -1 },
  ]);
});
test("chunkRanges: boundaries relative to the block size", () => {
  assert.deepEqual(chunkRanges(0, 249, 250), [{ start: 0, end: 249 }]);
  assert.deepEqual(chunkRanges(0, 250, 250), [{ start: 0, end: 249 }, { start: 250, end: 250 }]);
  assert.deepEqual(chunkRanges(-3, 246, 250), [{ start: -3, end: 246 }]); // exactly one full block
});
test("chunkRanges: large range covers every frame exactly once", () => {
  const r = chunkRanges(0, 249999, 250);
  assert.equal(r.length, 1000);
  assert.deepEqual(r[0], { start: 0, end: 249 });
  assert.deepEqual(r[r.length - 1], { start: 249750, end: 249999 });
  for (let i = 1; i < r.length; i++) assert.equal(r[i].start, r[i - 1].end + 1);
  for (const b of r.slice(0, -1)) assert.equal(b.end - b.start + 1, 250);
});
test("chunkRanges: more invalid arguments, and negative last < first", () => {
  assert.throws(() => chunkRanges(0, 10, -0), RangeError);
  assert.throws(() => chunkRanges(0, 10, Infinity), RangeError);
  assert.throws(() => chunkRanges(0, Infinity, 5), RangeError);
  assert.throws(() => chunkRanges(true, 10, 5), RangeError);
  assert.throws(() => chunkRanges(0, true, 5), RangeError);
  assert.throws(() => chunkRanges(NaN, NaN, 5), RangeError);
  assert.deepEqual(chunkRanges(-2, -5, 3), []);
});

// ---------------------------------------------------------------- chunkStartFor
test("chunkStartFor: the formula applies to non-finite and negative frame numbers", () => {
  assert.ok(Number.isNaN(chunkStartFor(NaN, 0, 250)));
  assert.equal(chunkStartFor(Infinity, 0, 250), Infinity);
  assert.equal(chunkStartFor(-Infinity, 0, 250), -Infinity);
  assert.equal(chunkStartFor(-1, 0, 250), -250);
});
test("chunkStartFor: non-integer frame numbers around block boundaries", () => {
  assert.equal(chunkStartFor(249.999, 0, 250), 0);
  assert.equal(chunkStartFor(250.001, 0, 250), 250);
  assert.equal(chunkStartFor(99.5, 100, 250), -150);
});
test("chunkStartFor: size 1", () => {
  assert.equal(chunkStartFor(5.7, 0, 1), 5);
  assert.equal(chunkStartFor(5.3, 10, 1), 5);
  assert.equal(chunkStartFor(-4.2, -10, 1), -5);
});
test("chunkStartFor: negative first", () => {
  assert.equal(chunkStartFor(-3, -10, 5), -5);
  assert.equal(chunkStartFor(-11, -10, 5), -15);
});
test("chunkStartFor: boolean size or first, and negative size, throw", () => {
  assert.throws(() => chunkStartFor(5, true, 10), RangeError);
  assert.throws(() => chunkStartFor(5, 0, true), RangeError);
  assert.throws(() => chunkStartFor(5, 0, -1), RangeError);
});

// ---------------------------------------------------------------- FrameCache
test("FrameCache: three-way overlap uses the greatest start", () => {
  const c = new FrameCache(3, 10); // chunks cover start .. start + 9
  c.put(0, [fr(0), fr(2)]);
  c.put(5, [fr(5), fr(7)]);
  c.put(8, [fr(8), fr(10)]);
  assert.equal(c.frameAt(8).frame, 8);  // chunks 0, 5 and 8 all cover 8 -> chunk 8
  assert.equal(c.frameAt(7).frame, 7);  // chunks 0 and 5 cover 7 -> chunk 5
  assert.equal(c.frameAt(2).frame, 2);  // only chunk 0 covers 2
  assert.equal(c.frameAt(10).frame, 10); // chunks 5 and 8 cover 10 -> chunk 8
  assert.equal(c.frameAt(11).frame, 10); // last frame <= 11 of chunk 8
  assert.equal(c.frameAt(17).frame, 10); // 17 is the last number chunk 8 covers
  assert.equal(c.frameAt(18), null);     // nothing covers 18
  assert.equal(c.frameAt(-1), null);
});
test("FrameCache: recency across overlapping chunks and eviction", () => {
  const c = new FrameCache(2, 10);
  c.put(0, [fr(0), fr(3)]);
  c.put(5, [fr(5)]);
  assert.deepEqual(c.starts(), [0, 5]);
  c.frameAt(6); // covered by chunks 0 and 5 -> touches 5 (already MRU)
  assert.deepEqual(c.starts(), [0, 5]);
  c.frameAt(2); // only chunk 0 covers 2 -> touches 0
  assert.deepEqual(c.starts(), [5, 0]);
  c.put(10, [fr(10)]); // evicts chunk 5
  assert.deepEqual(c.starts(), [0, 10]);
  assert.equal(c.has(5), false);
  assert.equal(c.frameAt(7).frame, 3);   // chunk 0 (0..9); chunk 5 is gone
  assert.equal(c.frameAt(12).frame, 10); // chunk 10 (10..19)
  assert.equal(c.frameAt(19).frame, 10); // 19 is the last number chunk 10 covers
  assert.equal(c.frameAt(20), null);
});
test("FrameCache: covered range ends at start + chunkSize - 1, even for fractional queries", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(0), fr(249)]);
  assert.equal(c.frameAt(249).frame, 249);
  assert.equal(c.frameAt(249.0000001), null); // > 0 + 250 - 1, outside the covered range
  c.put(250, [fr(250), fr(500)]);
  assert.equal(c.frameAt(250).frame, 250);
  assert.equal(c.frameAt(499).frame, 250);   // last frame <= 499 in chunk 250
  assert.equal(c.frameAt(499.5), null);      // 499.5 > 250 + 250 - 1
  assert.equal(c.frameAt(500), null);        // outside the range even though fr(500) is stored
  assert.equal(c.frameAt(749), null);
});
test("FrameCache: query at the chunk start and just below it", () => {
  const c = new FrameCache(2);
  c.put(250, [fr(250)]);
  assert.equal(c.frameAt(250).frame, 250);
  assert.equal(c.frameAt(249.999), null);
});
test("FrameCache: negative chunk starts are valid", () => {
  const c = new FrameCache(2);
  c.put(-250, [fr(-250), fr(-1)]); // covers -250 .. -1
  assert.equal(c.has(-250), true);
  assert.equal(c.frameAt(-250).frame, -250);
  assert.equal(c.frameAt(-1).frame, -1);
  assert.equal(c.frameAt(0), null);
});
test("FrameCache: put rejects boolean start, typed arrays and array-likes, changing nothing", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(0)]);
  assert.throws(() => c.put(true, []), RangeError);
  assert.throws(() => c.put(-0.5, []), RangeError);
  assert.throws(() => c.put(10, new Uint8Array(1)), TypeError);
  assert.throws(() => c.put(10, { 0: fr(0), length: 1 }), TypeError);
  assert.throws(() => c.put(1.5, null)); // both invalid: some error, no change
  assert.equal(c.size, 1);
  assert.deepEqual(c.starts(), [0]);
  assert.equal(c.frameAt(0).frame, 0);
});
test("FrameCache: empty chunks count toward size and get evicted", () => {
  const c = new FrameCache(2);
  c.put(0, []);
  c.put(250, []);
  assert.equal(c.size, 2);
  c.put(500, []);
  assert.equal(c.size, 2);
  assert.equal(c.has(0), false);
  assert.equal(c.has(250), true);
  assert.equal(c.has(500), true);
  assert.equal(c.frameAt(300), null); // chunk 250 covers 300 but holds no frames
});
test("FrameCache: frameAt rejects other non-number types", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(0)]);
  assert.equal(c.frameAt(true), null);
  assert.equal(c.frameAt({}), null);
  assert.equal(c.frameAt(Symbol.iterator), null);
});
test("FrameCache: has is false for non-integer or differently-typed starts", () => {
  const c = new FrameCache(2);
  c.put(0, []);
  assert.equal(c.has(0.5), false);
  assert.equal(c.has("0"), false);
});
test("FrameCache: paging 1000 chunks with a cap of 100", () => {
  const c = new FrameCache(100);
  for (let s = 0; s < 250000; s += 250) c.put(s, [fr(s)]);
  assert.equal(c.size, 100);
  assert.equal(c.has(0), false);
  assert.equal(c.has(249750), true);
  const st = c.starts();
  assert.equal(st.length, 100);
  assert.equal(st[0], 225000);
  assert.equal(st[st.length - 1], 249750);
  for (let i = 1; i < st.length; i++) assert.equal(st[i], st[i - 1] + 250);
  assert.equal(c.frameAt(225000).frame, 225000); // LRU chunk still answers
  assert.equal(c.frameAt(250000), null);         // beyond the newest covered range
});
test("FrameCache: repeated scenarios give the same order", () => {
  function scenario() {
    const c = new FrameCache(3);
    c.put(0, [fr(0), fr(10)]);
    c.put(250, [fr(250)]);
    c.frameAt(10);
    c.frameAt(1000); // miss: no touch
    c.put(500, [fr(500)]);
    return { starts: c.starts(), size: c.size };
  }
  assert.deepEqual(scenario(), scenario());
  assert.deepEqual(scenario(), { starts: [250, 0, 500], size: 3 });
});
