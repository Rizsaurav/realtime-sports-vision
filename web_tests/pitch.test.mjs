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

// ---------------------------------------------------------------- PITCH
test("PITCH constants", () => {
  assert.equal(PITCH.length, 105);
  assert.equal(PITCH.width, 68);
  assert.equal(PITCH.margin, 4);
  assert.deepEqual(Object.keys(PITCH).sort(), ["length", "margin", "width"]);
});

// ---------------------------------------------------------------- pitchGeometry
test("pitchGeometry: 1130x760 gives k 10", () => {
  assert.deepEqual(pitchGeometry(1130, 760), { k: 10, ox: 565, oy: 380 });
});
test("pitchGeometry: height limits", () => {
  assert.deepEqual(pitchGeometry(1130, 380), { k: 5, ox: 565, oy: 190 });
});
test("pitchGeometry: width limits", () => {
  assert.deepEqual(pitchGeometry(226, 1000), { k: 2, ox: 113, oy: 500 });
});
test("pitchGeometry: non-integer scale", () => {
  const g = pitchGeometry(900, 600);
  close(g.k, Math.min(900 / 113, 600 / 76));
  close(g.k, 600 / 76);
  assert.equal(g.ox, 450);
  assert.equal(g.oy, 300);
});
test("pitchGeometry: square canvas is limited by width", () => {
  const g = pitchGeometry(500, 500);
  close(g.k, 500 / 113);
});
test("pitchGeometry: zero width", () => {
  assert.deepEqual(pitchGeometry(0, 500), { k: 0, ox: 0, oy: 250 });
});
test("pitchGeometry: NaN and negative behave like zero", () => {
  assert.deepEqual(pitchGeometry(NaN, 500), { k: 0, ox: 0, oy: 250 });
  assert.deepEqual(pitchGeometry(-10, 500), { k: 0, ox: 0, oy: 250 });
  assert.deepEqual(pitchGeometry(500, Infinity), { k: 0, ox: 250, oy: 0 });
  assert.deepEqual(pitchGeometry(undefined, undefined), { k: 0, ox: 0, oy: 0 });
});
test("pitchGeometry: pitch with margin fits the canvas", () => {
  const g = pitchGeometry(1000, 700);
  assert.ok((105 + 8) * g.k <= 1000 + 1e-9);
  assert.ok((68 + 8) * g.k <= 700 + 1e-9);
});
test("pitchGeometry: returns a fresh object with exactly three keys", () => {
  const a = pitchGeometry(100, 100), b = pitchGeometry(100, 100);
  assert.notEqual(a, b);
  assert.deepEqual(Object.keys(a).sort(), ["k", "ox", "oy"]);
});

// ---------------------------------------------------------------- pitchToCanvas / canvasToPitch
test("pitchToCanvas: centre, corners, interior point", () => {
  closeXY(pitchToCanvas([0, 0], G), [565, 380]);
  closeXY(pitchToCanvas([52.5, 34], G), [1090, 720]);
  closeXY(pitchToCanvas([-52.5, -34], G), [40, 40]);
  closeXY(pitchToCanvas([1.25, -2], G), [577.5, 360]);
});
test("pitchToCanvas: y is not flipped, no clipping", () => {
  const p = pitchToCanvas([100, 50], G);
  closeXY(p, [1565, 880]);
  assert.ok(pitchToCanvas([0, 10], G)[1] > pitchToCanvas([0, 0], G)[1]);
});
test("pitchToCanvas: returns a new array and does not touch input", () => {
  const xy = [3, 4];
  const out = pitchToCanvas(xy, G);
  assert.notEqual(out, xy);
  assert.deepEqual(xy, [3, 4]);
  assert.equal(out.length, 2);
});
test("pitchToCanvas: matches the viewer rule X = ox + x*k", () => {
  const g = pitchGeometry(840, 560);
  const [X, Y] = pitchToCanvas([-12.34, 5.6], g);
  close(X, g.ox + -12.34 * g.k);
  close(Y, g.oy + 5.6 * g.k);
});
test("canvasToPitch: known values", () => {
  closeXY(canvasToPitch([565, 380], G), [0, 0]);
  closeXY(canvasToPitch([1090, 720], G), [52.5, 34]);
  closeXY(canvasToPitch([577.5, 360], G), [1.25, -2]);
});
test("canvasToPitch: round trip", () => {
  const g = pitchGeometry(777, 431);
  for (const p of [[0, 0], [-52.5, 34], [13.37, -20.2], [60, 45], [-0.001, 0.001]]) {
    closeXY(canvasToPitch(pitchToCanvas(p, g), g), p, 1e-9);
  }
});
test("canvasToPitch: other direction round trip", () => {
  const g = pitchGeometry(640, 480);
  for (const q of [[0, 0], [100, 50], [640, 480], [321.5, 99.25]]) {
    closeXY(pitchToCanvas(canvasToPitch(q, g), g), q, 1e-9);
  }
});
test("canvasToPitch: k of zero gives [0, 0] and a new array", () => {
  const g = pitchGeometry(0, 0);
  const out = canvasToPitch([30, 40], g);
  assert.deepEqual(out, [0, 0]);
  assert.notEqual(out, canvasToPitch([30, 40], g));
});

// ---------------------------------------------------------------- personStyle
test("personStyle: left player with colours", () => {
  assert.deepEqual(personStyle({ role: "player", team: "left", jersey: "10" }, COLORS), {
    fill: "#d03030", stroke: EDGE, strokeWidth: 1.5, label: "10",
  });
});
test("personStyle: right player with colours", () => {
  assert.equal(personStyle({ role: "player", team: "right", jersey: "3" }, COLORS).fill, "#2050c0");
});
test("personStyle: built-in fallback when colours are null", () => {
  assert.equal(personStyle({ role: "player", team: "left" }, null).fill, "#e4572e");
  assert.equal(personStyle({ role: "player", team: "right" }, null).fill, "#3e8ed0");
  assert.equal(personStyle({ role: "player", team: "left" }).fill, "#e4572e");
});
test("personStyle: custom fallback is used before the built-in", () => {
  assert.equal(personStyle({ team: "left" }, null, { left: "#111111" }).fill, "#111111");
  assert.equal(personStyle({ team: "right" }, null, { left: "#111111" }).fill, "#3e8ed0");
  assert.equal(personStyle({ team: "right" }, undefined, { left: "#111111", right: "#222222" }).fill, "#222222");
});
test("personStyle: colours win over fallback", () => {
  assert.equal(personStyle({ team: "left" }, COLORS, { left: "#111111", right: "#222222" }).fill, "#d03030");
});
test("personStyle: partial colours use fallback per side", () => {
  const c = { left: "#d03030" };
  assert.equal(personStyle({ team: "left" }, c).fill, "#d03030");
  assert.equal(personStyle({ team: "right" }, c).fill, "#3e8ed0");
  assert.equal(personStyle({ team: "right" }, c, { right: "#abcdef" }).fill, "#abcdef");
});
test("personStyle: empty or non-string colour values are not usable", () => {
  assert.equal(personStyle({ team: "left" }, { left: "", right: "#2050c0" }).fill, "#e4572e");
  assert.equal(personStyle({ team: "left" }, { left: null }, { left: "" }).fill, "#e4572e");
  assert.equal(personStyle({ team: "left" }, { left: 5 }).fill, "#e4572e");
});
test("personStyle: colour string is returned unchanged", () => {
  assert.equal(personStyle({ team: "left" }, { left: "#D03030" }).fill, "#D03030");
  assert.equal(personStyle({ team: "left" }, { left: "red" }).fill, "red");
});
test("personStyle: referee is yellow regardless of team", () => {
  const s = personStyle({ role: "referee", team: null, jersey: null }, COLORS);
  assert.deepEqual(s, { fill: "#f2d544", stroke: EDGE, strokeWidth: 1.5, label: "" });
  assert.equal(personStyle({ role: "referee", team: "left" }, COLORS).fill, "#f2d544");
});
test("personStyle: unknown team is grey", () => {
  assert.equal(personStyle({ role: "player", team: null }, COLORS).fill, "#9aa59f");
  assert.equal(personStyle({ role: "player" }, COLORS).fill, "#9aa59f");
  assert.equal(personStyle({ role: "player", team: "home" }, COLORS).fill, "#9aa59f");
});
test("personStyle: goalkeeper outline", () => {
  assert.deepEqual(personStyle({ role: "goalkeeper", team: "right", jersey: null }, COLORS), {
    fill: "#2050c0", stroke: "#ffffff", strokeWidth: 2.5, label: "",
  });
});
test("personStyle: goalkeeper with unknown team is grey with white outline", () => {
  const s = personStyle({ role: "goalkeeper", team: null, jersey: "1" }, COLORS);
  assert.equal(s.fill, "#9aa59f");
  assert.equal(s.stroke, "#ffffff");
  assert.equal(s.strokeWidth, 2.5);
});
test("personStyle: non-goalkeepers have the thin dark outline", () => {
  for (const role of ["player", "referee", undefined, "coach"]) {
    const s = personStyle({ role, team: "left" }, COLORS);
    assert.equal(s.stroke, EDGE);
    assert.equal(s.strokeWidth, 1.5);
  }
});
test("personStyle: label rules", () => {
  assert.equal(personStyle({ jersey: "10" }, null).label, "10");
  assert.equal(personStyle({ jersey: "0" }, null).label, "0");
  assert.equal(personStyle({ jersey: 7 }, null).label, "7");
  assert.equal(personStyle({ jersey: null }, null).label, "");
  assert.equal(personStyle({ jersey: "" }, null).label, "");
  assert.equal(personStyle({}, null).label, "");
});
test("personStyle: referee keeps its jersey as label", () => {
  assert.equal(personStyle({ role: "referee", jersey: "5" }, null).label, "5");
});
test("personStyle: missing person behaves like an empty object", () => {
  const expected = { fill: "#9aa59f", stroke: EDGE, strokeWidth: 1.5, label: "" };
  assert.deepEqual(personStyle(null, COLORS), expected);
  assert.deepEqual(personStyle(undefined, COLORS), expected);
  assert.deepEqual(personStyle({}, COLORS), expected);
});
test("personStyle: exactly four keys, fresh object, inputs untouched", () => {
  const person = { role: "player", team: "left", jersey: "9" };
  const colors = { ...COLORS };
  const a = personStyle(person, colors), b = personStyle(person, colors);
  assert.notEqual(a, b);
  assert.deepEqual(Object.keys(a).sort(), ["fill", "label", "stroke", "strokeWidth"]);
  assert.deepEqual(person, { role: "player", team: "left", jersey: "9" });
  assert.deepEqual(colors, COLORS);
});

// ---------------------------------------------------------------- chunkRanges
test("chunkRanges: exact multiple", () => {
  assert.deepEqual(chunkRanges(0, 749, 250), [
    { start: 0, end: 249 }, { start: 250, end: 499 }, { start: 500, end: 749 },
  ]);
});
test("chunkRanges: short last block", () => {
  assert.deepEqual(chunkRanges(0, 600, 250), [
    { start: 0, end: 249 }, { start: 250, end: 499 }, { start: 500, end: 600 },
  ]);
});
test("chunkRanges: single frame", () => {
  assert.deepEqual(chunkRanges(100, 100, 250), [{ start: 100, end: 100 }]);
});
test("chunkRanges: range smaller than one block", () => {
  assert.deepEqual(chunkRanges(7, 20, 250), [{ start: 7, end: 20 }]);
});
test("chunkRanges: last below first is empty", () => {
  assert.deepEqual(chunkRanges(10, 5, 250), []);
  assert.deepEqual(chunkRanges(10, 9, 3), []);
});
test("chunkRanges: blocks start at first, not at a multiple of size", () => {
  assert.deepEqual(chunkRanges(5, 14, 5), [{ start: 5, end: 9 }, { start: 10, end: 14 }]);
  assert.deepEqual(chunkRanges(100, 360, 250), [{ start: 100, end: 349 }, { start: 350, end: 360 }]);
});
test("chunkRanges: size 1", () => {
  const r = chunkRanges(0, 9, 1);
  assert.equal(r.length, 10);
  assert.deepEqual(r[9], { start: 9, end: 9 });
  assert.ok(r.every((b) => b.start === b.end));
});
test("chunkRanges: negative first", () => {
  assert.deepEqual(chunkRanges(-3, 3, 4), [{ start: -3, end: 0 }, { start: 1, end: 3 }]);
});
test("chunkRanges: blocks cover everything exactly once", () => {
  const r = chunkRanges(3, 1000, 250);
  assert.equal(r[0].start, 3);
  assert.equal(r.at(-1).end, 1000);
  for (let i = 1; i < r.length; i++) assert.equal(r[i].start, r[i - 1].end + 1);
  for (const b of r.slice(0, -1)) assert.equal(b.end - b.start + 1, 250);
});
test("chunkRanges: invalid arguments throw RangeError", () => {
  assert.throws(() => chunkRanges(0, 10, 0), RangeError);
  assert.throws(() => chunkRanges(0, 10, -5), RangeError);
  assert.throws(() => chunkRanges(0, 10, 2.5), RangeError);
  assert.throws(() => chunkRanges(0, 10, NaN), RangeError);
  assert.throws(() => chunkRanges(0, 10, "5"), RangeError);
  assert.throws(() => chunkRanges(0, 10.5, 5), RangeError);
  assert.throws(() => chunkRanges(0.5, 10, 5), RangeError);
  assert.throws(() => chunkRanges(10, 5, 0), RangeError);
});
test("chunkRanges: returns new objects each call", () => {
  const a = chunkRanges(0, 10, 5), b = chunkRanges(0, 10, 5);
  assert.notEqual(a, b);
  assert.notEqual(a[0], b[0]);
});

// ---------------------------------------------------------------- chunkStartFor
test("chunkStartFor: first 0, size 250", () => {
  assert.equal(chunkStartFor(0, 0, 250), 0);
  assert.equal(chunkStartFor(249, 0, 250), 0);
  assert.equal(chunkStartFor(250, 0, 250), 250);
  assert.equal(chunkStartFor(499.9, 0, 250), 250);
  assert.equal(chunkStartFor(1000, 0, 250), 1000);
});
test("chunkStartFor: first 100", () => {
  assert.equal(chunkStartFor(100, 100, 250), 100);
  assert.equal(chunkStartFor(349, 100, 250), 100);
  assert.equal(chunkStartFor(350, 100, 250), 350);
});
test("chunkStartFor: below first applies the formula", () => {
  assert.equal(chunkStartFor(99, 100, 250), -150);
});
test("chunkStartFor: agrees with chunkRanges", () => {
  const ranges = chunkRanges(37, 900, 100);
  for (const r of ranges) {
    for (const n of [r.start, r.end, Math.floor((r.start + r.end) / 2)]) {
      assert.equal(chunkStartFor(n, 37, 100), r.start);
    }
  }
});
test("chunkStartFor: invalid size or first throws RangeError", () => {
  assert.throws(() => chunkStartFor(5, 0, 0), RangeError);
  assert.throws(() => chunkStartFor(5, 0, 1.5), RangeError);
  assert.throws(() => chunkStartFor(5, 0.5, 10), RangeError);
});

// ---------------------------------------------------------------- FrameCache
test("FrameCache: constructor validation", () => {
  assert.throws(() => new FrameCache(0), RangeError);
  assert.throws(() => new FrameCache(-1), RangeError);
  assert.throws(() => new FrameCache(1.5), RangeError);
  assert.throws(() => new FrameCache(), RangeError);
  assert.throws(() => new FrameCache("2"), RangeError);
  assert.throws(() => new FrameCache(2, 0), RangeError);
  assert.throws(() => new FrameCache(2, 2.5), RangeError);
  assert.doesNotThrow(() => new FrameCache(1));
  assert.doesNotThrow(() => new FrameCache(3, 10));
});
test("FrameCache: starts empty", () => {
  const c = new FrameCache(3);
  assert.equal(c.size, 0);
  assert.equal(c.has(0), false);
  assert.equal(c.frameAt(0), null);
  assert.deepEqual(c.starts(), []);
});
test("FrameCache: put, has, size", () => {
  const c = new FrameCache(3);
  assert.equal(c.put(0, [fr(0)]), undefined);
  c.put(250, [fr(250)]);
  assert.equal(c.size, 2);
  assert.equal(c.has(0), true);
  assert.equal(c.has(250), true);
  assert.equal(c.has(500), false);
  assert.equal(c.has(1), false);
});
test("FrameCache: frameAt returns the last frame at or before the number", () => {
  const c = new FrameCache(2);
  const f = [fr(0), fr(5), fr(10)];
  c.put(0, f);
  assert.equal(c.frameAt(0), f[0]);
  assert.equal(c.frameAt(4), f[0]);
  assert.equal(c.frameAt(5), f[1]);
  assert.equal(c.frameAt(7), f[1]);
  assert.equal(c.frameAt(7.9), f[1]);
  assert.equal(c.frameAt(10), f[2]);
  assert.equal(c.frameAt(249), f[2]);
});
test("FrameCache: frameAt returns the same object reference", () => {
  const c = new FrameCache(2);
  const f = [fr(0), fr(1)];
  c.put(0, f);
  assert.strictEqual(c.frameAt(1), f[1]);
});
test("FrameCache: frameAt null outside cached ranges", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(0), fr(5), fr(10)]);
  assert.equal(c.frameAt(-1), null);
  assert.equal(c.frameAt(250), null);
  assert.equal(c.frameAt(1000), null);
});
test("FrameCache: frameAt never looks at other chunks", () => {
  const c = new FrameCache(3);
  c.put(0, [fr(0), fr(249)]);
  c.put(500, [fr(500)]);
  assert.equal(c.frameAt(300), null);
  assert.equal(c.frameAt(499), null);
});
test("FrameCache: frameAt null when no frame in the chunk is early enough", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(3), fr(4)]);
  assert.equal(c.frameAt(2), null);
  assert.equal(c.frameAt(3).frame, 3);
});
test("FrameCache: empty chunk is valid and returns null", () => {
  const c = new FrameCache(2);
  c.put(0, []);
  assert.equal(c.has(0), true);
  assert.equal(c.size, 1);
  assert.equal(c.frameAt(10), null);
});
test("FrameCache: frameAt rejects non-finite or non-number input", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(0)]);
  assert.equal(c.frameAt(NaN), null);
  assert.equal(c.frameAt(Infinity), null);
  assert.equal(c.frameAt(undefined), null);
  assert.equal(c.frameAt("0"), null);
  assert.equal(c.frameAt(null), null);
});
test("FrameCache: custom chunk size sets the covered range", () => {
  const c = new FrameCache(2, 10);
  c.put(0, [fr(0), fr(9)]);
  assert.equal(c.frameAt(9).frame, 9);
  assert.equal(c.frameAt(10), null);
});
test("FrameCache: chunks not starting at a multiple of the size", () => {
  const c = new FrameCache(2);
  c.put(100, [fr(100), fr(300)]);
  assert.equal(c.frameAt(99), null);
  assert.equal(c.frameAt(349).frame, 300);
  assert.equal(c.frameAt(350), null);
});
test("FrameCache: evicts the least recently put chunk", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(0)]);
  c.put(250, [fr(250)]);
  c.put(500, [fr(500)]);
  assert.equal(c.size, 2);
  assert.equal(c.has(0), false);
  assert.deepEqual(c.starts(), [250, 500]);
  assert.equal(c.frameAt(10), null);
});
test("FrameCache: frameAt hit touches the chunk", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(0)]);
  c.put(250, [fr(250)]);
  assert.equal(c.frameAt(10).frame, 0);
  assert.deepEqual(c.starts(), [250, 0]);
  c.put(500, [fr(500)]);
  assert.equal(c.has(250), false);
  assert.equal(c.has(0), true);
  assert.deepEqual(c.starts(), [0, 500]);
});
test("FrameCache: has does not touch", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(0)]);
  c.put(250, [fr(250)]);
  assert.equal(c.has(0), true);
  assert.deepEqual(c.starts(), [0, 250]);
  c.put(500, [fr(500)]);
  assert.equal(c.has(0), false);
});
test("FrameCache: frameAt miss does not touch", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(0)]);
  c.put(250, [fr(250)]);
  assert.equal(c.frameAt(9999), null);
  assert.deepEqual(c.starts(), [0, 250]);
});
test("FrameCache: frameAt touches even when it returns null", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(5)]);
  c.put(250, [fr(250)]);
  assert.equal(c.frameAt(2), null);
  assert.deepEqual(c.starts(), [250, 0]);
});
test("FrameCache: put on an existing start replaces and touches", () => {
  const c = new FrameCache(2);
  const old = [fr(0)], fresh = [fr(0), fr(1)];
  c.put(0, old);
  c.put(250, [fr(250)]);
  c.put(0, fresh);
  assert.equal(c.size, 2);
  assert.deepEqual(c.starts(), [250, 0]);
  assert.equal(c.frameAt(1), fresh[1]);
  c.put(500, [fr(500)]);
  assert.deepEqual(c.starts(), [0, 500]);
});
test("FrameCache: the chunk just put is never evicted (max 1)", () => {
  const c = new FrameCache(1);
  c.put(0, [fr(0)]);
  c.put(250, [fr(250)]);
  assert.equal(c.size, 1);
  assert.equal(c.has(0), false);
  assert.equal(c.has(250), true);
  c.put(250, [fr(250), fr(251)]);
  assert.equal(c.size, 1);
  assert.equal(c.frameAt(251).frame, 251);
});
test("FrameCache: walk-through from the task text", () => {
  const c = new FrameCache(2);
  const a = [fr(0), fr(10)];
  c.put(0, a);
  c.put(250, [fr(250)]);
  assert.deepEqual(c.starts(), [0, 250]);
  c.frameAt(10);
  assert.deepEqual(c.starts(), [250, 0]);
  c.put(500, [fr(500)]);
  assert.deepEqual(c.starts(), [0, 500]);
  assert.equal(c.size, 2);
  assert.equal(c.has(250), false);
  c.put(0, [fr(0)]);
  assert.deepEqual(c.starts(), [500, 0]);
});
test("FrameCache: frames are stored by reference", () => {
  const c = new FrameCache(2);
  const f = [fr(0)];
  c.put(0, f);
  f.push(fr(3));
  assert.equal(c.frameAt(3).frame, 3);
});
test("FrameCache: put validates arguments and changes nothing on error", () => {
  const c = new FrameCache(2);
  c.put(0, [fr(0)]);
  assert.throws(() => c.put(1.5, []), RangeError);
  assert.throws(() => c.put("250", []), RangeError);
  assert.throws(() => c.put(NaN, []), RangeError);
  assert.throws(() => c.put(250, null), TypeError);
  assert.throws(() => c.put(250, {}), TypeError);
  assert.equal(c.size, 1);
  assert.deepEqual(c.starts(), [0]);
});
test("FrameCache: clear empties the cache", () => {
  const c = new FrameCache(3);
  c.put(0, [fr(0)]);
  c.put(250, [fr(250)]);
  assert.equal(c.clear(), undefined);
  assert.equal(c.size, 0);
  assert.equal(c.has(0), false);
  assert.equal(c.frameAt(0), null);
  assert.deepEqual(c.starts(), []);
});
test("FrameCache: size is a getter, not a method", () => {
  const c = new FrameCache(2);
  assert.equal(typeof c.size, "number");
});
test("FrameCache: overlapping chunks use the greatest start", () => {
  const c = new FrameCache(3, 10);
  c.put(0, [fr(0), fr(8)]);
  c.put(5, [fr(5), fr(9)]);
  assert.equal(c.frameAt(8).frame, 5);
  assert.equal(c.frameAt(12).frame, 9);
  assert.equal(c.frameAt(3).frame, 0);
});
test("FrameCache: starts returns a copy", () => {
  const c = new FrameCache(2);
  c.put(0, []);
  c.starts().push(99);
  assert.deepEqual(c.starts(), [0]);
});
test("FrameCache: paging through a match keeps at most maxChunks", () => {
  const c = new FrameCache(4);
  const first = 0;
  for (const r of chunkRanges(first, 2499, 250)) {
    const frames = [];
    for (let n = r.start; n <= r.end; n += 5) frames.push(fr(n));
    c.put(r.start, frames);
    assert.ok(c.size <= 4);
    assert.equal(c.frameAt(r.start + 7).frame, r.start + 5);
  }
  assert.deepEqual(c.starts(), [1500, 1750, 2000, 2250]);
});
