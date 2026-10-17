import test from "node:test";
import assert from "node:assert/strict";

const lib = await import(
  process.env.REPLAY_LIB ? "file://" + process.env.REPLAY_LIB : "../src/sideline/web/lib/replay.js"
);
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
// frames 0, 25, 50, 75, 100 (4 seconds at 25 fps)
function fiveFrames() {
  return [0, 25, 50, 75, 100].map((n) => frame(n));
}
function timeline(frames, opts = {}) {
  const clock = makeClock(1000);
  const tl = new Timeline({ frames, now: clock.now, ...opts });
  return { tl, clock };
}

// ---------- frameIndexAt ----------
test("frameIndexAt: empty list gives -1", () => {
  assert.equal(frameIndexAt([], 5), -1);
});
test("frameIndexAt: before the first frame gives -1", () => {
  assert.equal(frameIndexAt([frame(10), frame(11)], 9), -1);
});
test("frameIndexAt: exact match", () => {
  const fs = [frame(0), frame(5), frame(10)];
  assert.equal(frameIndexAt(fs, 0), 0);
  assert.equal(frameIndexAt(fs, 5), 1);
  assert.equal(frameIndexAt(fs, 10), 2);
});
test("frameIndexAt: between frames and gaps picks the earlier one", () => {
  const fs = [frame(0), frame(5), frame(10)];
  assert.equal(frameIndexAt(fs, 4), 0);
  assert.equal(frameIndexAt(fs, 7), 1);
  assert.equal(frameIndexAt(fs, 9.9), 1);
});
test("frameIndexAt: after the last frame gives the last index", () => {
  assert.equal(frameIndexAt([frame(0), frame(5)], 1000), 1);
});
test("frameIndexAt: large list", () => {
  const fs = [];
  for (let i = 0; i < 10000; i++) fs.push(frame(i * 2));
  assert.equal(frameIndexAt(fs, 9999), 4999);
  assert.equal(frameIndexAt(fs, 10000), 5000);
  assert.equal(frameIndexAt(fs, -1), -1);
});

// ---------- interpolatePeople ----------
test("interpolatePeople: halfway xy and box", () => {
  const a = frame(0, [person(1, [0, 0], [100, 200, 40, 80])]);
  const b = frame(1, [person(1, [10, -4], [110, 220, 50, 100])]);
  const out = interpolatePeople(a, b, 0.5);
  assert.equal(out.length, 1);
  closeXY(out[0].xy, [5, -2]);
  assert.deepEqual(out[0].box, [105, 210, 45, 90]);
});
test("interpolatePeople: t=0 returns a's values, t=1 returns b's values", () => {
  const a = frame(0, [person(1, [1, 2], [10, 20, 30, 40])]);
  const b = frame(1, [person(1, [3, 6], [20, 40, 60, 80])]);
  const o0 = interpolatePeople(a, b, 0)[0];
  closeXY(o0.xy, [1, 2]);
  assert.deepEqual(o0.box, [10, 20, 30, 40]);
  const o1 = interpolatePeople(a, b, 1)[0];
  closeXY(o1.xy, [3, 6]);
  assert.deepEqual(o1.box, [20, 40, 60, 80]);
});
test("interpolatePeople: box is rounded to integers", () => {
  const a = frame(0, [person(1, [0, 0], [0, 0, 10, 10])]);
  const b = frame(1, [person(1, [1, 1], [1, 3, 11, 14])]);
  // t = 0.4 -> [0.4, 1.2, 10.4, 11.6] -> [0, 1, 10, 12]
  assert.deepEqual(interpolatePeople(a, b, 0.4)[0].box, [0, 1, 10, 12]);
  for (const v of interpolatePeople(a, b, 0.37)[0].box) assert.ok(Number.isInteger(v));
});
test("interpolatePeople: only people present in both frames, in a's order", () => {
  const a = frame(0, [person(5, [0, 0], [0, 0, 1, 1]), person(2, [0, 0], [0, 0, 1, 1]), person(9, [0, 0], [0, 0, 1, 1])]);
  const b = frame(1, [person(9, [2, 2], [2, 2, 1, 1]), person(7, [0, 0], [0, 0, 1, 1]), person(5, [2, 2], [2, 2, 1, 1])]);
  const out = interpolatePeople(a, b, 0.5);
  assert.deepEqual(out.map((p) => p.id), [5, 9]);
});
test("interpolatePeople: role, team, jersey come from a", () => {
  const a = frame(0, [person(1, [0, 0], [0, 0, 1, 1], { role: "goalkeeper", team: "right", jersey: "1" })]);
  const b = frame(1, [person(1, [2, 2], [2, 2, 1, 1], { role: "player", team: "left", jersey: null })]);
  const p = interpolatePeople(a, b, 0.5)[0];
  assert.equal(p.role, "goalkeeper");
  assert.equal(p.team, "right");
  assert.equal(p.jersey, "1");
  assert.equal(p.id, 1);
});
test("interpolatePeople: xy null in a keeps a's values unchanged (box not interpolated)", () => {
  const a = frame(0, [person(1, null, [10, 10, 5, 5])]);
  const b = frame(1, [person(1, [4, 4], [30, 30, 5, 5])]);
  const p = interpolatePeople(a, b, 0.5)[0];
  assert.equal(p.xy, null);
  assert.deepEqual(p.box, [10, 10, 5, 5]);
});
test("interpolatePeople: xy null in b keeps a's values unchanged", () => {
  const a = frame(0, [person(1, [4, 4], [10, 10, 5, 5])]);
  const b = frame(1, [person(1, null, [30, 30, 5, 5])]);
  const p = interpolatePeople(a, b, 0.9)[0];
  assert.deepEqual(p.xy, [4, 4]);
  assert.deepEqual(p.box, [10, 10, 5, 5]);
});
test("interpolatePeople: empty people gives empty array", () => {
  assert.deepEqual(interpolatePeople(frame(0, []), frame(1, [person(1, [0, 0], [0, 0, 1, 1])]), 0.5), []);
  assert.deepEqual(interpolatePeople(frame(0, [person(1, [0, 0], [0, 0, 1, 1])]), frame(1, []), 0.5), []);
});
test("interpolatePeople: does not modify its inputs and returns fresh objects", () => {
  const pa = person(1, [0, 0], [0, 0, 10, 10]);
  const pb = person(1, [2, 2], [2, 2, 12, 12]);
  const a = frame(0, [pa]);
  const b = frame(1, [pb]);
  const out = interpolatePeople(a, b, 0.5);
  assert.deepEqual(pa.xy, [0, 0]);
  assert.deepEqual(pa.box, [0, 0, 10, 10]);
  assert.deepEqual(pb.xy, [2, 2]);
  assert.notEqual(out[0], pa);
  assert.notEqual(out[0].xy, pa.xy);
  const nullA = frame(0, [person(1, null, [1, 1, 1, 1])]);
  const keep = interpolatePeople(nullA, b, 0.5)[0];
  assert.notEqual(keep, nullA.people[0]);
  assert.notEqual(keep.box, nullA.people[0].box);
});
test("interpolatePeople: t outside [0,1] is clamped", () => {
  const a = frame(0, [person(1, [0, 0], [0, 0, 10, 10])]);
  const b = frame(1, [person(1, [10, 10], [10, 10, 20, 20])]);
  closeXY(interpolatePeople(a, b, -3)[0].xy, [0, 0]);
  closeXY(interpolatePeople(a, b, 7)[0].xy, [10, 10]);
});

// ---------- interpolateBall ----------
test("interpolateBall: halfway", () => {
  const a = frame(0, [], { box: [100, 100, 10, 10], xy: [0, 0] });
  const b = frame(1, [], { box: [110, 120, 10, 10], xy: [4, -8] });
  const ball = interpolateBall(a, b, 0.5);
  closeXY(ball.xy, [2, -4]);
  assert.deepEqual(ball.box, [105, 110, 10, 10]);
});
test("interpolateBall: null when either ball is null", () => {
  const ball = { box: [0, 0, 1, 1], xy: [0, 0] };
  assert.equal(interpolateBall(frame(0, [], null), frame(1, [], ball), 0.5), null);
  assert.equal(interpolateBall(frame(0, [], ball), frame(1, [], null), 0.5), null);
  assert.equal(interpolateBall(frame(0, [], null), frame(1, [], null), 0.5), null);
});
test("interpolateBall: xy null in either frame returns a copy of a's ball", () => {
  const withXY = { box: [50, 50, 10, 10], xy: [1, 1] };
  const noXY = { box: [90, 90, 10, 10], xy: null };
  const r1 = interpolateBall(frame(0, [], noXY), frame(1, [], withXY), 0.5);
  assert.equal(r1.xy, null);
  assert.deepEqual(r1.box, [90, 90, 10, 10]);
  const r2 = interpolateBall(frame(0, [], withXY), frame(1, [], noXY), 0.5);
  assert.deepEqual(r2.xy, [1, 1]);
  assert.deepEqual(r2.box, [50, 50, 10, 10]);
  assert.notEqual(r2, withXY);
});
test("interpolateBall: box rounded and inputs untouched", () => {
  const ba = { box: [0, 0, 10, 10], xy: [0, 0] };
  const bb = { box: [1, 3, 11, 14], xy: [1, 1] };
  const r = interpolateBall(frame(0, [], ba), frame(1, [], bb), 0.4);
  assert.deepEqual(r.box, [0, 1, 10, 12]);
  assert.deepEqual(ba.box, [0, 0, 10, 10]);
  assert.notEqual(r, ba);
});

// ---------- formatClock ----------
test("formatClock: zero", () => assert.equal(formatClock(0), "0:00.0"));
test("formatClock: seconds and tenths", () => {
  assert.equal(formatClock(5.25), "0:05.2");
  assert.equal(formatClock(12.5), "0:12.5");
});
test("formatClock: truncates instead of rounding", () => {
  assert.equal(formatClock(59.99), "0:59.9");
  assert.equal(formatClock(0.09), "0:00.0");
  assert.equal(formatClock(61.25), "1:01.2");
});
test("formatClock: minutes roll over and are not padded or capped", () => {
  assert.equal(formatClock(60), "1:00.0");
  assert.equal(formatClock(754.3), "12:34.3");
  assert.equal(formatClock(3600), "60:00.0");
  assert.equal(formatClock(6000.7), "100:00.7");
});
test("formatClock: exact tenths survive floating point error", () => {
  assert.equal(formatClock(0.3), "0:00.3");
  assert.equal(formatClock(1.1), "0:01.1");
  assert.equal(formatClock(4.1), "0:04.1");
  assert.equal(formatClock(2.3), "0:02.3");
});
test("formatClock: negative, NaN and infinite values give 0:00.0", () => {
  assert.equal(formatClock(-5), "0:00.0");
  assert.equal(formatClock(-0.01), "0:00.0");
  assert.equal(formatClock(NaN), "0:00.0");
  assert.equal(formatClock(Infinity), "0:00.0");
});

// ---------- Timeline basics ----------
test("Timeline: initial state", () => {
  const { tl } = timeline(fiveFrames());
  assert.equal(tl.duration, 4);
  assert.equal(tl.position, 0);
  assert.equal(tl.playing, false);
  assert.equal(tl.speed, 1);
});
test("Timeline: duration uses first and last frame numbers and fps", () => {
  const frames = [100, 101, 150].map((n) => frame(n));
  assert.equal(timeline(frames).tl.duration, 2);
  assert.equal(timeline(frames, { fps: 50 }).tl.duration, 1);
});
test("Timeline: position 0 is the first frame even when it is not frame 0", () => {
  const frames = [100, 125, 150].map((n) => frame(n));
  const { tl } = timeline(frames);
  const v = tl.current();
  assert.equal(v.index, 0);
  assert.equal(v.t, 0);
  tl.seek(1);
  const w = tl.current();
  assert.equal(w.index, 1);
  assert.equal(w.a.frame, 125);
});
test("Timeline: empty frames", () => {
  const { tl, clock } = timeline([]);
  assert.equal(tl.duration, 0);
  assert.equal(tl.current(), null);
  assert.equal(tl.tick(), null);
  tl.seek(5);
  assert.equal(tl.position, 0);
  tl.play();
  clock.advance(1000);
  assert.equal(tl.tick(), null);
  assert.equal(tl.playing, false);
});
test("Timeline: single frame", () => {
  const f = frame(40);
  const { tl, clock } = timeline([f]);
  assert.equal(tl.duration, 0);
  const v = tl.current();
  assert.equal(v.index, 0);
  assert.equal(v.t, 0);
  assert.equal(v.a, f);
  assert.equal(v.b, f);
  tl.play();
  assert.equal(tl.playing, true);
  clock.advance(500);
  const w = tl.tick();
  assert.equal(tl.playing, false);
  assert.equal(tl.position, 0);
  assert.equal(w.a, f);
});

// ---------- current() views ----------
test("current: exactly on a frame gives t = 0 and a = that frame", () => {
  const frames = fiveFrames();
  const { tl } = timeline(frames);
  tl.seek(1); // frame 25
  const v = tl.current();
  assert.equal(v.index, 1);
  assert.equal(v.t, 0);
  assert.equal(v.a, frames[1]);
  assert.equal(v.b, frames[2]);
});
test("current: between frames gives the fraction", () => {
  const frames = fiveFrames();
  const { tl } = timeline(frames);
  tl.seek(1.5); // frame 37.5
  const v = tl.current();
  assert.equal(v.index, 1);
  close(v.t, 0.5);
  assert.equal(v.a, frames[1]);
  assert.equal(v.b, frames[2]);
});
test("current: single-frame steps at 25 fps (0.12 s is frame 3)", () => {
  const frames = Array.from({ length: 10 }, (_, i) => frame(i));
  const { tl } = timeline(frames);
  tl.seek(0.12);
  const v = tl.current();
  assert.equal(v.index, 3);
  assert.equal(v.t, 0);
});
test("current: missing frames are interpolated across the gap", () => {
  const frames = [frame(0), frame(10), frame(11)];
  const { tl } = timeline(frames);
  tl.seek(0.2); // frame 5
  const v = tl.current();
  assert.equal(v.index, 0);
  assert.equal(v.b, frames[1]);
  close(v.t, 0.5);
});
test("current: at the end the last frame is both a and b with t = 0", () => {
  const frames = fiveFrames();
  const { tl } = timeline(frames);
  tl.seek(4);
  const v = tl.current();
  assert.equal(v.index, 4);
  assert.equal(v.t, 0);
  assert.equal(v.a, frames[4]);
  assert.equal(v.b, frames[4]);
});
test("current: does not advance the position while playing", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  clock.advance(2000);
  tl.current();
  assert.equal(tl.position, 0);
  assert.equal(tl.current().index, 0);
});

// ---------- seek ----------
test("seek: clamps to [0, duration]", () => {
  const { tl } = timeline(fiveFrames());
  tl.seek(-3);
  assert.equal(tl.position, 0);
  tl.seek(99);
  assert.equal(tl.position, 4);
  tl.seek(2.5);
  assert.equal(tl.position, 2.5);
});
test("seek: NaN throws RangeError and leaves position alone", () => {
  const { tl } = timeline(fiveFrames());
  tl.seek(1);
  assert.throws(() => tl.seek(NaN), RangeError);
  assert.equal(tl.position, 1);
});
test("seek: while paused does not start playback", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.seek(2);
  clock.advance(5000);
  tl.tick();
  assert.equal(tl.position, 2);
  assert.equal(tl.playing, false);
});
test("seek: while playing discards elapsed time before the seek", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  clock.advance(700);
  tl.seek(2);
  clock.advance(300);
  tl.tick();
  close(tl.position, 2.3);
});

// ---------- play / pause / tick ----------
test("tick: advances by elapsed clock time at speed 1", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  assert.equal(tl.playing, true);
  clock.advance(1000);
  tl.tick();
  close(tl.position, 1);
  clock.advance(500);
  const v = tl.tick();
  close(tl.position, 1.5);
  assert.equal(v.index, 1);
  close(v.t, 0.5);
});
test("tick: play does not count time that passed before play()", () => {
  const { tl, clock } = timeline(fiveFrames());
  clock.advance(10000);
  tl.play();
  clock.advance(250);
  tl.tick();
  close(tl.position, 0.25);
});
test("tick: when paused, position stays and the view is returned", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.seek(1);
  clock.advance(3000);
  const v = tl.tick();
  assert.equal(tl.position, 1);
  assert.equal(v.index, 1);
  assert.equal(tl.playing, false);
});
test("pause then play resumes without a jump", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  clock.advance(1000);
  tl.pause();
  assert.equal(tl.playing, false);
  close(tl.position, 1);
  clock.advance(60000);
  tl.tick();
  close(tl.position, 1);
  tl.play();
  clock.advance(500);
  tl.tick();
  close(tl.position, 1.5);
});
test("pause: applies the time elapsed since the last tick", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  clock.advance(400);
  tl.tick();
  clock.advance(600);
  tl.pause();
  close(tl.position, 1);
});
test("play: calling play twice does not reset the elapsed time", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  clock.advance(500);
  tl.play();
  clock.advance(500);
  tl.tick();
  close(tl.position, 1);
});
test("pause: when already paused is harmless", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.pause();
  clock.advance(1000);
  tl.pause();
  assert.equal(tl.position, 0);
  assert.equal(tl.playing, false);
});
test("tick: reaching the end clamps, stops, and returns the last frame view", () => {
  const frames = fiveFrames();
  const { tl, clock } = timeline(frames);
  tl.play();
  clock.advance(10000);
  const v = tl.tick();
  assert.equal(tl.position, 4);
  assert.equal(tl.playing, false);
  assert.equal(v.index, 4);
  assert.equal(v.a, frames[4]);
  clock.advance(1000);
  tl.tick();
  assert.equal(tl.position, 4);
});
test("tick: landing exactly on the duration also stops", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.seek(3);
  tl.play();
  clock.advance(1000);
  tl.tick();
  assert.equal(tl.position, 4);
  assert.equal(tl.playing, false);
});
test("play: at the end restarts from the beginning", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.seek(4);
  tl.play();
  assert.equal(tl.playing, true);
  assert.equal(tl.position, 0);
  clock.advance(1000);
  tl.tick();
  close(tl.position, 1);
});
test("tick: a clock that goes backwards never moves the position back", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  clock.advance(1000);
  tl.tick();
  clock.advance(-500);
  tl.tick();
  close(tl.position, 1);
  clock.advance(500);
  tl.tick();
  close(tl.position, 1.5);
});

// ---------- speed ----------
test("setSpeed: every allowed value is accepted and scales playback", () => {
  for (const s of [0.25, 0.5, 1, 2, 4, 8]) {
    const { tl, clock } = timeline(fiveFrames());
    tl.setSpeed(s);
    assert.equal(tl.speed, s);
    tl.play();
    clock.advance(400);
    tl.tick();
    close(tl.position, 0.4 * s);
  }
});
test("setSpeed: other values throw RangeError and leave the speed unchanged", () => {
  const { tl } = timeline(fiveFrames());
  tl.setSpeed(2);
  for (const bad of [0, -1, 3, 16, 0.75, NaN, "2", null, undefined]) {
    assert.throws(() => tl.setSpeed(bad), RangeError, String(bad));
  }
  assert.equal(tl.speed, 2);
});
test("setSpeed: mid-play applies the old speed to time already elapsed", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.play();
  clock.advance(1000);
  tl.setSpeed(2);
  close(tl.position, 1);
  clock.advance(1000);
  tl.tick();
  close(tl.position, 3);
});
test("setSpeed: while paused has no effect on position and is kept for play", () => {
  const { tl, clock } = timeline(fiveFrames());
  clock.advance(5000);
  tl.setSpeed(4);
  assert.equal(tl.position, 0);
  tl.play();
  clock.advance(250);
  tl.tick();
  close(tl.position, 1);
});
test("setSpeed: fast playback stops at the end", () => {
  const { tl, clock } = timeline(fiveFrames());
  tl.setSpeed(8);
  tl.play();
  clock.advance(1000);
  tl.tick();
  assert.equal(tl.position, 4);
  assert.equal(tl.playing, false);
});

// ---------- constructor ----------
test("Timeline: default fps is 25 and a custom fps is used", () => {
  const frames = [0, 30, 60].map((n) => frame(n));
  assert.equal(new Timeline({ frames, now: () => 0 }).duration, 2.4);
  assert.equal(new Timeline({ frames, fps: 30, now: () => 0 }).duration, 2);
  const tl = new Timeline({ frames, fps: 30, now: () => 0 });
  tl.seek(1);
  assert.equal(tl.current().a.frame, 30);
});
test("Timeline: non-positive fps throws RangeError", () => {
  assert.throws(() => new Timeline({ frames: [], fps: 0, now: () => 0 }), RangeError);
  assert.throws(() => new Timeline({ frames: [], fps: -25, now: () => 0 }), RangeError);
});
test("Timeline: the default clock works without an injected one", () => {
  const tl = new Timeline({ frames: fiveFrames() });
  tl.play();
  tl.tick();
  assert.ok(tl.position >= 0 && tl.position <= 4);
  tl.pause();
});
