import test from "node:test";
import assert from "node:assert/strict";

const lib = await import("../src/sideline/web/lib/replay.js");
const { frameIndexAt, interpolatePeople, interpolateBall, formatClock, Timeline } = lib;

const close = (actual, expected, eps = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= eps, `expected ${actual} to be within ${eps} of ${expected}`);
const closeXY = (actual, expected) => {
  assert.equal(actual.length, 2);
  close(actual[0], expected[0]);
  close(actual[1], expected[1]);
};
function person(id, xy, box, extra = {}) {
  return { id, role: "player", team: "left", jersey: "10", box, xy, ...extra };
}
function frame(n, people = [], ball = null) {
  return { frame: n, delay: 0, people, ball, colors: null };
}
function makeClock(start = 0) {
  const c = { t: start, now: () => c.t, advance(ms) { c.t += ms; } };
  return c;
}
function fiveFrames() {
  return [0, 25, 50, 75, 100].map((n) => frame(n));
}
function timeline(frames, opts = {}) {
  const clock = makeClock(1000);
  const tl = new Timeline({ frames, now: clock.now, ...opts });
  return { tl, clock };
}
function deepFreeze(o) {
  if (o && typeof o === "object" && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const v of Object.values(o)) deepFreeze(v);
  }
  return o;
}

// ---------- frameIndexAt ----------
test("frameIndexAt: non-integer frameNumber", () => {
  const fs = [frame(0), frame(5), frame(10)];
  assert.equal(frameIndexAt(fs, 4.99999), 0);
  assert.equal(frameIndexAt(fs, 5.000001), 1);
  assert.equal(frameIndexAt(fs, 9.999999), 1);
  assert.equal(frameIndexAt(fs, 10.5), 2);
  assert.equal(frameIndexAt(fs, 0.25), 0);
  assert.equal(frameIndexAt(fs, -0.0000001), -1);
});
test("frameIndexAt: single-element list", () => {
  const fs = [frame(5)];
  assert.equal(frameIndexAt(fs, 5), 0);
  assert.equal(frameIndexAt(fs, 4.999999), -1);
  assert.equal(frameIndexAt(fs, 5.0000001), 0);
  assert.equal(frameIndexAt(fs, 100), 0);
});
test("frameIndexAt: negative frame numbers", () => {
  const fs = [frame(-10), frame(5)];
  assert.equal(frameIndexAt(fs, -10.5), -1);
  assert.equal(frameIndexAt(fs, -10), 0);
  assert.equal(frameIndexAt(fs, -9.999), 0);
  assert.equal(frameIndexAt(fs, 4.999), 0);
  assert.equal(frameIndexAt(fs, 5), 1);
  assert.equal(frameIndexAt(fs, 6), 1);
});
test("frameIndexAt: 10 000 frames with gaps, checked against the spec definition", () => {
  const fs = [];
  for (let i = 0; i < 10000; i++) fs.push(frame(i * 7 + (i % 3))); // strictly increasing
  const oracle = (q) => {
    let ans = -1;
    for (let i = 0; i < fs.length; i++) {
      if (fs[i].frame <= q) ans = i;
      else break;
    }
    return ans;
  };
  for (const q of [-1, 0, 1.5, 42.75, 20000, 35000.5, 69994, 69995, 69995.9, 70000, 1e9]) {
    assert.equal(frameIndexAt(fs, q), oracle(q), `query ${q}`);
  }
});
test("frameIndexAt: very large frame numbers", () => {
  const fs = [frame(1000000), frame(1000005)];
  assert.equal(frameIndexAt(fs, 999999), -1);
  assert.equal(frameIndexAt(fs, 1000000), 0);
  assert.equal(frameIndexAt(fs, 1000002.5), 0);
  assert.equal(frameIndexAt(fs, 1000004.99999), 0);
  assert.equal(frameIndexAt(fs, 1000005), 1);
  assert.equal(frameIndexAt(fs, 1e9), 1);
});

// ---------- interpolatePeople ----------
test("interpolatePeople: id appearing twice in b uses the first occurrence", () => {
  const a = frame(0, [person(7, [0, 0], [0, 0, 10, 10])]);
  const b = frame(1, [
    person(7, [10, 0], [10, 0, 10, 10]),
    person(7, [20, 0], [20, 0, 10, 10]),
  ]);
  const p = interpolatePeople(a, b, 0.5)[0];
  closeXY(p.xy, [5, 0]); // from the first b entry, not [10, 0]
});
test("interpolatePeople: all fields come from a; extra fields of b are ignored", () => {
  const a = frame(0, [{ id: 4, role: "referee", team: null, jersey: null, box: [0, 0, 10, 10], xy: [1, 1], extraA: "keep" }]);
  const b = frame(1, [{ id: 4, role: "player", team: "left", jersey: "9", box: [20, 20, 10, 10], xy: [3, 3], extraB: "drop" }]);
  const p = interpolatePeople(a, b, 0.5)[0];
  assert.equal(p.id, 4);
  assert.equal(p.role, "referee");
  assert.equal(p.team, null);
  assert.equal(p.jersey, null);
  assert.equal(p.extraA, "keep");
  assert.ok(!("extraB" in p));
  closeXY(p.xy, [2, 2]);
  assert.deepEqual(p.box, [10, 10, 10, 10]);
});
test("interpolatePeople: xy null in both frames keeps a's person exactly", () => {
  const a = frame(0, [person(1, null, [1, 2, 3, 4])]);
  const b = frame(1, [person(1, null, [5, 6, 7, 8])]);
  const p = interpolatePeople(a, b, 0.5)[0];
  assert.equal(p.xy, null);
  assert.deepEqual(p.box, [1, 2, 3, 4]);
});
test("interpolatePeople: xy is interpolated without rounding", () => {
  const a = frame(0, [person(1, [1, 2], [0, 0, 10, 10])]);
  const b = frame(1, [person(1, [2, 5], [10, 10, 10, 10])]);
  closeXY(interpolatePeople(a, b, 0.25)[0].xy, [1.25, 2.75]);
  closeXY(interpolatePeople(a, b, 0.1)[0].xy, [1.1, 2.3]);
});
test("interpolatePeople: result follows a's order even when b is reordered", () => {
  const a = frame(0, [person(3, [0, 0], [0, 0, 1, 1]), person(1, [0, 0], [0, 0, 1, 1]), person(2, [0, 0], [0, 0, 1, 1])]);
  const b = frame(1, [person(1, [1, 1], [1, 1, 1, 1]), person(2, [2, 2], [2, 2, 1, 1]), person(3, [3, 3], [3, 3, 1, 1]), person(9, [9, 9], [9, 9, 1, 1])]);
  assert.deepEqual(interpolatePeople(a, b, 0.5).map((p) => p.id), [3, 1, 2]);
});
test("interpolatePeople: repeated calls are deterministic and return fresh objects", () => {
  const a = frame(0, [person(1, [0, 0], [0, 0, 10, 10])]);
  const b = frame(1, [person(1, [4, 6], [2, 2, 12, 12])]);
  const o1 = interpolatePeople(a, b, 0.5);
  const o2 = interpolatePeople(a, b, 0.5);
  assert.deepEqual(o1, o2);
  assert.notEqual(o1, o2);
  assert.notEqual(o1[0], o2[0]);
  assert.notEqual(o1[0].xy, o2[0].xy);
  assert.notEqual(o1[0].box, o2[0].box);
  assert.notEqual(o1[0].xy, a.people[0].xy);
  assert.notEqual(o1[0].box, a.people[0].box);
  assert.notEqual(o1[0].box, b.people[0].box);
});
test("interpolatePeople: never mutates frozen inputs", () => {
  const a = deepFreeze(frame(0, [person(1, [0, 0], [0, 0, 10, 10])]));
  const b = deepFreeze(frame(1, [person(1, [10, 20], [5, 5, 15, 15])]));
  const p = interpolatePeople(a, b, 0.5)[0];
  closeXY(p.xy, [5, 10]);
  assert.deepEqual(p.box, [3, 3, 13, 13]); // raw [2.5, 2.5, 12.5, 12.5], Math.round rounds .5 up
  const a2 = deepFreeze(frame(0, [person(2, null, [1, 2, 3, 4])]));
  const p2 = interpolatePeople(a2, frame(1, [person(2, [9, 9], [9, 9, 9, 9])]), 0.9)[0];
  assert.equal(p2.xy, null);
  assert.deepEqual(p2.box, [1, 2, 3, 4]);
});
test("interpolatePeople: id appearing twice in a yields one entry per occurrence of a", () => {
  const a = frame(0, [person(1, [0, 0], [0, 0, 1, 1]), person(1, [10, 10], [10, 10, 1, 1])]);
  const b = frame(1, [person(1, [20, 20], [20, 20, 2, 2])]);
  const out = interpolatePeople(a, b, 0.5);
  assert.equal(out.length, 2);
  closeXY(out[0].xy, [10, 10]);
  closeXY(out[1].xy, [15, 15]);
});
test("interpolatePeople: 3000 people, order and values spot-checked", () => {
  let s = 123456789;
  const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const shuffle = (arr) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  };
  const orderA = shuffle(Array.from({ length: 3000 }, (_, i) => i));
  const orderB = shuffle(Array.from({ length: 3000 }, (_, i) => i).filter((id) => id % 100 !== 0));
  for (let k = 0; k < 10; k++) orderB.push(9000 + k); // b-only ids, must be dropped
  const a = frame(0, orderA.map((id) => person(id, [id, -id], [id % 100, 0, 10, 10])));
  const b = frame(1, orderB.map((id) => person(id, [id + 1, -id + 2], [(id + 1) % 100, 1, 11, 11])));
  const out = interpolatePeople(a, b, 0.5);
  const expectedIds = orderA.filter((id) => id % 100 !== 0);
  assert.equal(out.length, 2970);
  assert.deepEqual(out.map((p) => p.id), expectedIds);
  for (const pos of [0, 1485, 2969]) {
    const id = expectedIds[pos];
    const p = out[pos];
    closeXY(p.xy, [id + 0.5, -id + 1]);
    assert.deepEqual(p.box, [
      Math.round((id % 100) + (((id + 1) % 100) - (id % 100)) * 0.5),
      Math.round(0.5),
      Math.round(10.5),
      Math.round(10.5),
    ]);
  }
});

// ---------- interpolateBall ----------
test("interpolateBall: t outside [0,1] is clamped", () => {
  const a = frame(0, [], { box: [0, 0, 10, 10], xy: [0, 0] });
  const b = frame(1, [], { box: [10, 20, 20, 20], xy: [10, 20] });
  const lo = interpolateBall(a, b, -2);
  closeXY(lo.xy, [0, 0]);
  assert.deepEqual(lo.box, [0, 0, 10, 10]);
  const hi = interpolateBall(a, b, 3);
  closeXY(hi.xy, [10, 20]);
  assert.deepEqual(hi.box, [10, 20, 20, 20]);
});
test("interpolateBall: t=0 gives a's values, t=1 gives b's values", () => {
  const a = frame(0, [], { box: [0, 0, 10, 10], xy: [0, 0] });
  const b = frame(1, [], { box: [10, 20, 20, 20], xy: [10, 20] });
  closeXY(interpolateBall(a, b, 0).xy, [0, 0]);
  assert.deepEqual(interpolateBall(a, b, 0).box, [0, 0, 10, 10]);
  closeXY(interpolateBall(a, b, 1).xy, [10, 20]);
  assert.deepEqual(interpolateBall(a, b, 1).box, [10, 20, 20, 20]);
});
test("interpolateBall: extra fields come from a's ball, b's are ignored", () => {
  const a = frame(0, [], { box: [0, 0, 10, 10], xy: [0, 0], label: "a" });
  const b = frame(1, [], { box: [10, 0, 10, 10], xy: [4, 0], label: "b" });
  const r = interpolateBall(a, b, 0.5);
  assert.equal(r.label, "a");
  closeXY(r.xy, [2, 0]);
  assert.deepEqual(r.box, [5, 0, 10, 10]);
});
test("interpolateBall: xy null in both balls keeps a's ball exactly", () => {
  const a = frame(0, [], { box: [1, 2, 3, 4], xy: null });
  const b = frame(1, [], { box: [5, 6, 7, 8], xy: null });
  const r = interpolateBall(a, b, 0.5);
  assert.equal(r.xy, null);
  assert.deepEqual(r.box, [1, 2, 3, 4]);
});
test("interpolateBall: returns new objects and arrays, inputs untouched", () => {
  const a = frame(0, [], { box: [0, 0, 10, 10], xy: [0, 0] });
  const b = frame(1, [], { box: [8, 12, 14, 16], xy: [4, 4] });
  const r = interpolateBall(a, b, 0.5);
  assert.notEqual(r, a.ball);
  assert.notEqual(r, b.ball);
  assert.notEqual(r.xy, a.ball.xy);
  assert.notEqual(r.xy, b.ball.xy);
  assert.notEqual(r.box, a.ball.box);
  assert.notEqual(r.box, b.ball.box);
  assert.deepEqual(a.ball.xy, [0, 0]);
  assert.deepEqual(a.ball.box, [0, 0, 10, 10]);
  assert.deepEqual(b.ball.xy, [4, 4]);
  assert.deepEqual(b.ball.box, [8, 12, 14, 16]);
  // null-xy copy path also returns fresh arrays
  const a2 = frame(0, [], { box: [1, 2, 3, 4], xy: null });
  const r2 = interpolateBall(a2, frame(1, [], { box: [5, 6, 7, 8], xy: [1, 1] }), 0.5);
  assert.notEqual(r2, a2.ball);
  assert.notEqual(r2.box, a2.ball.box);
});
test("interpolateBall: box rounding at another t", () => {
  const a = frame(0, [], { box: [0, 0, 10, 10], xy: [0, 0] });
  const b = frame(1, [], { box: [1, 3, 11, 14], xy: [1, 1] });
  // raw [0.3, 0.9, 10.3, 11.2] -> [0, 1, 10, 11]
  assert.deepEqual(interpolateBall(a, b, 0.3).box, [0, 1, 10, 11]);
});

// ---------- formatClock ----------
test("formatClock: tenths are truncated, never rounded", () => {
  assert.equal(formatClock(0.05), "0:00.0");
  assert.equal(formatClock(0.15), "0:00.1");
  assert.equal(formatClock(0.25), "0:00.2");
  assert.equal(formatClock(0.999), "0:00.9");
  assert.equal(formatClock(59.995), "0:59.9");
  assert.equal(formatClock(59.9999), "0:59.9");
});
test("formatClock: minutes roll over with no padding or upper limit", () => {
  assert.equal(formatClock(99.99), "1:39.9");
  assert.equal(formatClock(599.99), "9:59.9");
  assert.equal(formatClock(60000), "1000:00.0");
  assert.equal(formatClock(1000000), "16666:40.0");
  assert.equal(formatClock(60000000), "1000000:00.0");
});
test("formatClock: -Infinity and tiny negatives give 0:00.0", () => {
  assert.equal(formatClock(-Infinity), "0:00.0");
  assert.equal(formatClock(-0.0000001), "0:00.0");
});
test("formatClock: more exact tenths survive floating point error", () => {
  assert.equal(formatClock(0.1), "0:00.1");
  assert.equal(formatClock(9.9), "0:09.9");
});
test("formatClock: boundary around a minute rollover", () => {
  assert.equal(formatClock(60.05), "1:00.0");
  assert.equal(formatClock(60.15), "1:00.1");
});

// ---------- Timeline: constructor ----------
test("Timeline: fps must be a finite number > 0", () => {
  for (const bad of [NaN, Infinity, -Infinity, "25", true, 0, -1, -0]) {
    assert.throws(() => new Timeline({ frames: [], fps: bad, now: () => 0 }), RangeError, String(bad));
  }
});
test("Timeline: missing frames is treated as []", () => {
  const tl = new Timeline({ now: () => 0 });
  assert.equal(tl.duration, 0);
  assert.equal(tl.current(), null);
  assert.equal(tl.tick(), null);
  tl.seek(5);
  assert.equal(tl.position, 0);
  tl.play();
  assert.equal(tl.tick(), null);
  assert.equal(tl.playing, false);
});
test("Timeline: the frames reference is kept, not copied", () => {
  const frames = fiveFrames();
  const { tl } = timeline(frames);
  assert.equal(tl.frames, frames);
  assert.equal(tl.duration, 4);
  frames.push(frame(125));
  assert.equal(tl.duration, 5);
});
test("Timeline: the default clock is not consulted at construction time", () => {
  const orig = performance.now.bind(performance);
  let calls = 0;
  performance.now = () => {
    calls++;
    return orig();
  };
  try {
    const tl = new Timeline({ frames: fiveFrames() });
    assert.equal(calls, 0, "constructor must not call the clock");
    tl.play();
    assert.ok(calls >= 1, "play() must consult the clock");
    tl.pause();
  } finally {
    performance.now = orig;
  }
});

// ---------- Timeline: current() ----------
test("Timeline: current() does not touch last (no hidden time accounting)", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  clock.advance(1000);
  tl.current();
  tl.current();
  tl.tick();
  close(tl.position, 1);
});
test("Timeline: current() returns the original frame objects by reference", () => {
  const frames = fiveFrames();
  const { tl } = timeline(frames);
  tl.seek(1.5);
  const v = tl.current();
  assert.equal(v.a, frames[1]);
  assert.equal(v.b, frames[2]);
  tl.seek(4);
  const w = tl.current();
  assert.equal(w.a, frames[4]);
  assert.equal(w.b, frames[4]);
});
test("Timeline: fps is used for the position-to-frame mapping", () => {
  const frames = [0, 100].map((n) => frame(n));
  const { tl } = timeline(frames, { fps: 100 });
  assert.equal(tl.duration, 1);
  tl.seek(0.5); // fn = 0 + 0.5 * 100 = 50
  const v = tl.current();
  assert.equal(v.index, 0);
  close(v.t, 0.5);
  assert.equal(v.a, frames[0]);
  assert.equal(v.b, frames[1]);
  tl.seek(0.05); // fn = 5
  const w = tl.current();
  assert.equal(w.index, 0);
  close(w.t, 0.05);
});

// ---------- Timeline: seek ----------
test("Timeline: seek(Infinity) and seek(-Infinity) clamp to the ends", () => {
  const { tl } = timeline(fiveFrames());
  tl.seek(Infinity);
  assert.equal(tl.position, 4);
  tl.seek(-Infinity);
  assert.equal(tl.position, 0);
});
test("Timeline: seek with a non-number throws and changes nothing, not even last", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.seek(2);
  assert.throws(() => tl.seek("2"), RangeError);
  assert.equal(tl.position, 2);
  tl.play();
  assert.throws(() => tl.seek("2"), RangeError);
  assert.equal(tl.playing, true);
  tl.seek(2); // resets last at clock 1000
  clock.advance(500);
  tl.tick();
  close(tl.position, 2.5); // the failed seek must not have reset last
});
test("Timeline: seeking to the end while playing stays playing until the next tick", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  tl.seek(4);
  assert.equal(tl.position, 4);
  assert.equal(tl.playing, true);
  clock.advance(1);
  tl.tick();
  assert.equal(tl.position, 4);
  assert.equal(tl.playing, false);
});
test("Timeline: seeking backwards while playing restarts the clock from the seek", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  clock.advance(2000);
  tl.tick();
  close(tl.position, 2);
  tl.seek(0.5);
  assert.equal(tl.position, 0.5);
  clock.advance(1000);
  tl.tick();
  close(tl.position, 1.5);
});

// ---------- Timeline: play / pause / tick ----------
test("Timeline: pause while already paused discards the elapsed time", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.seek(1);
  clock.advance(5000);
  tl.pause();
  assert.equal(tl.position, 1);
  assert.equal(tl.playing, false);
  tl.play();
  clock.advance(500);
  tl.tick();
  close(tl.position, 1.5);
});
test("Timeline: tick without clock advance does not move the position", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  clock.advance(1000);
  tl.tick();
  close(tl.position, 1);
  tl.tick();
  close(tl.position, 1);
});
test("Timeline: single frame stops immediately on the first tick", () => {
  const f = frame(40);
  const { tl } = timeline([f]);
  tl.play();
  assert.equal(tl.playing, true);
  const v = tl.tick(); // no time has passed
  assert.equal(tl.playing, false);
  assert.equal(tl.position, 0);
  assert.equal(v.index, 0);
  assert.equal(v.t, 0);
  assert.equal(v.a, f);
  assert.equal(v.b, f);
});
test("Timeline: repeated play/pause cycles accumulate exactly", () => {
  const { tl, clock } = timeline(fiveFrames());
  for (let i = 0; i < 5; i++) {
    tl.play();
    clock.advance(200);
    tl.pause();
  }
  close(tl.position, 1);
  assert.equal(tl.playing, false);
});
test("Timeline: two timelines sharing one clock and one frames list are independent", () => {
  const frames = fiveFrames();
  const clock = makeClock(0);
  const t1 = new Timeline({ frames, now: clock.now });
  const t2 = new Timeline({ frames, now: clock.now });
  assert.equal(t1.frames, frames);
  assert.equal(t2.frames, frames);
  t1.play(); // last1 = 0
  clock.advance(500);
  t2.play(); // last2 = 500
  clock.advance(500);
  t1.tick();
  t2.tick();
  close(t1.position, 1); // 1000 ms of playback
  close(t2.position, 0.5); // only 500 ms of playback
  assert.equal(t1.playing, true);
  assert.equal(t2.playing, true);
  t1.pause();
  t2.pause();
});

// ---------- Timeline: setSpeed ----------
test("Timeline: a rejected setSpeed changes nothing, not even last", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  clock.advance(1000);
  assert.throws(() => tl.setSpeed(3), RangeError);
  assert.equal(tl.speed, 1);
  tl.tick();
  close(tl.position, 1); // elapsed time was not discarded by the failed call
});
test("Timeline: setSpeed while paused at a nonzero position applies on play", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.seek(1);
  tl.setSpeed(8);
  assert.equal(tl.position, 1);
  assert.equal(tl.speed, 8);
  tl.play();
  clock.advance(250);
  tl.tick();
  close(tl.position, 3); // 1 + 0.25 * 8
});

// ---------- Timeline: large input ----------
test("Timeline: 10 000 frames, duration and end view", () => {
  const frames = Array.from({ length: 10000 }, (_, i) => frame(i));
  const { tl } = timeline(frames);
  close(tl.duration, 9999 / 25);
  tl.seek(399.96);
  const v = tl.current();
  assert.equal(v.index, 9999);
  assert.equal(v.t, 0);
  assert.equal(v.a, frames[9999]);
  assert.equal(v.b, frames[9999]);
});
test("Timeline: frames with a large offset", () => {
  const frames = [1000000, 1000001, 1000002].map((n) => frame(n));
  const { tl } = timeline(frames);
  close(tl.duration, 0.08);
  assert.equal(tl.current().index, 0);
  tl.seek(0.04); // fn = 1000000 + 1 = 1000001
  const v = tl.current();
  assert.equal(v.index, 1);
  assert.equal(v.t, 0);
  assert.equal(v.a, frames[1]);
  assert.equal(v.a.frame, 1000001);
});
test("Timeline: fps scales playback speed (fps 50)", () => {
  const frames = [0, 50, 100].map((n) => frame(n));
  const { tl, clock } = timeline(frames, { fps: 50 });
  assert.equal(tl.duration, 2);
  tl.play();
  clock.advance(1000);
  const v = tl.tick();
  close(tl.position, 1);
  assert.equal(v.index, 1); // fn = 0 + 1 * 50 = 50
  assert.equal(v.t, 0);
});
