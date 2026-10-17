// Visual-stability checks for the calm-replay pipeline (see
// tasks/visual_stability.md). Runs on the real fixture
// tests/fixtures/erratic_match.jsonl, prints raw and processed numbers,
// plus small synthetic cases. The pipeline under test lives in
// src/sideline/web/lib/replay.js (rejectGlitches / smoothTrack /
// processFrames); the inputs are never modified.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

const lib = await import(
  process.env.REPLAY_LIB ? "file://" + process.env.REPLAY_LIB : "../src/sideline/web/lib/replay.js"
);
const { rejectGlitches, smoothTrack, processFrames } = lib;

const fixturePath = new URL("../tests/fixtures/erratic_match.jsonl", import.meta.url);
const fixture = fs.readFileSync(fixturePath, "utf8").trim().split("\n").map((l) => JSON.parse(l));

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// Nearest-rank percentile of a list of numbers.
function pct(values, p) {
  const s = values.slice().sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
}
const maxOf = (values) => values.reduce((m, v) => (v > m ? v : m), 0);

// Per-person frame-to-frame step (metres per frame of gap) of every drawn
// (non-null, non-hidden) position in a frame list.
function stepsOf(frameList, hidden = new Set()) {
  const byId = new Map();
  for (const f of frameList) {
    for (const p of f.people) {
      if (!p.xy || hidden.has(p.id)) continue;
      if (!byId.has(p.id)) byId.set(p.id, []);
      byId.get(p.id).push({ frame: f.frame, xy: p.xy });
    }
  }
  const steps = [];
  for (const [, pts] of byId) {
    for (let i = 1; i < pts.length; i++) {
      steps.push(dist(pts[i - 1].xy, pts[i].xy) / (pts[i].frame - pts[i - 1].frame));
    }
  }
  return steps;
}

// Whole-frame glitch intervals: frames where a majority of the visible
// people are carried forward (their raw position is garbage).
function glitchIntervals(frameList) {
  const out = new Set();
  for (const f of frameList) {
    let ghost = 0, vis = 0;
    for (const p of f.people) {
      if (!p.xy) continue;
      vis++;
      if (p.ghost) ghost++;
    }
    if (vis >= 8 && ghost * 2 >= vis) out.add(f.frame);
  }
  return out;
}

// Transition window of an accepted relocation for one person: from a bit
// before the jump frame to a bit after the glide (including any chase
// after it) has finished (plus the settle ghost frames around it).
function inTransition(id, frame, relocations, before = 4, after = 10) {
  return relocations.some((r) => {
    const end = r.endFrame ?? r.frame + r.k;
    return r.id === id && frame >= r.jumpFrame - before && frame <= end + after;
  });
}

function runPipeline() {
  const { frames: rejected, relocations: rels } = rejectGlitches(fixture);
  const { frames: processed, relocations, hidden } = processFrames(fixture);
  return { rejected, relocations: rels, processed, hidden };
}

test("(a) processed per-frame step: p99 <= 0.3 m, max <= 0.5 m (raw: 8.85 m, 550 m)", () => {
  const { processed, hidden } = runPipeline();
  const rawSteps = stepsOf(fixture);
  const procSteps = stepsOf(processed, hidden);
  const raw = { p95: pct(rawSteps, 95), p99: pct(rawSteps, 99), max: maxOf(rawSteps) };
  const proc = { p95: pct(procSteps, 95), p99: pct(procSteps, 99), max: maxOf(procSteps) };
  console.log(
    `    raw       n=${rawSteps.length} p95=${raw.p95.toFixed(3)} p99=${raw.p99.toFixed(3)} max=${raw.max.toFixed(1)}`
  );
  console.log(
    `    processed n=${procSteps.length} p95=${proc.p95.toFixed(3)} p99=${proc.p99.toFixed(3)} max=${proc.max.toFixed(3)}`
  );
  assert.ok(proc.p99 <= 0.3 + 1e-9, `processed p99 step ${proc.p99.toFixed(3)} m exceeds 0.3 m`);
  assert.ok(proc.max <= 0.5 + 1e-9, `processed max step ${proc.max.toPrecision(6)} m exceeds 0.5 m`);
});

test("(b) processed p95 step is at most 50% of the raw p95", () => {
  const { processed, hidden } = runPipeline();
  const rawP95 = pct(stepsOf(fixture), 95);
  const procP95 = pct(stepsOf(processed, hidden), 95);
  console.log(`    raw p95=${rawP95.toFixed(3)}  processed p95=${procP95.toFixed(3)} (${((100 * procP95) / rawP95).toFixed(1)}%)`);
  assert.ok(procP95 <= rawP95 / 2, `processed p95 ${procP95.toFixed(3)} exceeds 50% of raw ${rawP95.toFixed(3)}`);
});

test("(c1) processing must not move players: mean displacement per track (>=50 appearances) stays near zero", () => {
  const { rejected, processed, relocations, hidden } = runPipeline();
  const glitches = glitchIntervals(rejected);
  const rawBy = new Map();
  for (const f of fixture) for (const p of f.people) {
    if (!p.xy) continue;
    if (!rawBy.has(p.id)) rawBy.set(p.id, new Map());
    rawBy.get(p.id).set(f.frame, p.xy);
  }
  let worst = { id: -1, disp: -1, n: 0 };
  let bad = [];
  for (const [id, rawMap] of rawBy) {
    if (rawMap.size < 50 || hidden.has(id)) continue;
    let sx = 0, sy = 0, n = 0;
    for (const f of processed) {
      const p = f.people.find((q) => q.id === id);
      if (!p || !p.xy) continue;
      if (glitches.has(f.frame) || inTransition(id, f.frame, relocations)) continue;
      const raw = rawMap.get(f.frame);
      if (!raw) continue;
      sx += p.xy[0] - raw[0];
      sy += p.xy[1] - raw[1];
      n++;
    }
    if (n === 0) continue;
    const disp = Math.hypot(sx / n, sy / n);
    if (disp > worst.disp) worst = { id, disp, n };
    if (disp > 0.4) bad.push(`${id}:${disp.toFixed(2)}`);
  }
  console.log(`    worst mean displacement (excl. transitions/glitches): id ${worst.id} ${worst.disp.toFixed(3)} m over ${worst.n} pts`);
  assert.equal(bad.length, 0, `tracks moved more than 0.4 m on average: ${bad.join(", ")}`);
});

test("(c2) no processed position is >2 m from raw except at accepted relocations / whole-frame glitches", () => {
  const { rejected, processed, relocations, hidden } = runPipeline();
  const glitches = glitchIntervals(rejected);
  const rawPos = new Map();
  for (const f of fixture) for (const p of f.people) if (p.xy) rawPos.set(f.frame + ":" + p.id, p.xy);
  // For every id, the frames where a relocation was accepted (its
  // transition window) or where no relocation ever happened (a candidate
  // farther than maxAutoRelocate was rejected: a wrong match, kept at the
  // last known position).
  let relocationPts = 0, glitchPts = 0, wrongMatchPts = 0, unexplained = 0, total = 0;
  const unexplainedList = [];
  for (const f of processed) {
    for (const p of f.people) {
      if (!p.xy || hidden.has(p.id)) continue;
      total++;
      const raw = rawPos.get(f.frame + ":" + p.id);
      if (!raw) continue;
      const d = dist(p.xy, raw);
      if (d <= 2) continue;
      if (inTransition(p.id, f.frame, relocations)) {
        relocationPts++;
      } else if (glitches.has(f.frame)) {
        glitchPts++;
      } else if (
        p.ghost &&
        d > 20 &&
        !relocations.some((r) => r.id === p.id && r.jumpFrame >= f.frame - 20)
      ) {
        wrongMatchPts++;
      } else {
        unexplained++;
        if (unexplainedList.length < 20) unexplainedList.push(`${p.id}@${f.frame}:${d.toFixed(1)}`);
      }
    }
  }
  const all = relocationPts + glitchPts + wrongMatchPts + unexplained;
  console.log(
    `    >2 m from raw: ${all} pts (${((100 * all) / total).toFixed(1)}% of ${total}) = relocation ${relocationPts} + glitch ${glitchPts} + wrong-match ${wrongMatchPts} + unexplained ${unexplained}`
  );
  if (unexplainedList.length) console.log(`    unexplained sample: ${unexplainedList.join(" ")}`);
  assert.ok(
    unexplained * 100 < total * 5,
    `${unexplained} points are >2 m from raw and not explained by a relocation or glitch`
  );
});

test("(d) processing off returns the raw data unchanged", () => {
  const { frames } = processFrames(fixture, { enabled: false });
  assert.deepEqual(
    JSON.parse(JSON.stringify(frames)),
    JSON.parse(JSON.stringify(fixture))
  );
});

test("(e) the functions never modify their input", () => {
  const before = JSON.parse(JSON.stringify(fixture));
  const points = [
    { frame: 0, xy: [1, 2] },
    { frame: 1, xy: [1.1, 2] },
    { frame: 2, xy: [1.2, 2.1] },
  ];
  const pointsBefore = JSON.parse(JSON.stringify(points));
  rejectGlitches(fixture);
  processFrames(fixture);
  smoothTrack(points);
  assert.deepEqual(fixture, before);
  assert.deepEqual(points, pointsBefore);
});

test("(f1) a player walking at 1.5 m/s with one 40 m glitch frame in the middle", () => {
  const speed = 1.5 / 25; // metres per frame at 25 fps
  const frames = [];
  for (let n = 0; n < 150; n++) {
    let xy = [n * speed, 0];
    if (n === 75) xy = [n * speed + 40, 0]; // one-frame 40 m spike
    frames.push({ frame: n, delay: 0, people: [{ id: 1, role: "player", team: "left", jersey: "1", box: [0, 0, 10, 10], xy }], ball: null, colors: null });
  }
  const { frames: rejected, relocations } = rejectGlitches(frames);
  const g = rejected[75].people[0];
  assert.equal(g.ghost, true, "the glitch frame must be a ghost (carried position)");
  assert.ok(dist(g.xy, [(74) * speed, 0]) < 0.01, "ghost carries the last accepted position");
  const { frames: processed } = processFrames(frames);
  const steps = stepsOf(processed);
  console.log(`    synthetic walk+glitch: p99=${pct(steps, 99).toFixed(3)} max=${maxOf(steps).toFixed(3)}`);
  assert.ok(maxOf(steps) <= 0.5 + 1e-9, "no step may exceed 0.5 m per frame");
  assert.ok(pct(steps, 99) <= 0.3 + 1e-9, "p99 step must stay under 0.3 m per frame");
  const last = processed.at(-1).people[0];
  assert.ok(dist(last.xy, [149 * speed, 0]) < 0.3, "the walk ends where it started heading");
  assert.equal(relocations.length, 0, "a one-frame spike is not a relocation");
});

test("(f2) a synthetic real relocation (everyone shifts 20 m and stays) is accepted after settle frames", () => {
  const frames = [];
  for (let n = 0; n < 130; n++) {
    const people = [];
    for (let i = 0; i < 10; i++) {
      people.push({
        id: i,
        role: "player",
        team: "left",
        jersey: String(i + 1),
        box: [0, 0, 10, 10],
        xy: [i * 3 + (n >= 30 ? 20 : 0), 0],
      });
    }
    frames.push({ frame: n, delay: 0, people, ball: null, colors: null });
  }
  const { relocations } = rejectGlitches(frames);
  assert.equal(relocations.length, 10, "every person gets exactly one relocation");
  for (const r of relocations) {
    assert.equal(r.jumpFrame, 30, "the jump frame is detected");
    assert.ok(r.frame >= 30 + 5 && r.frame <= 30 + 8, `accepted after the settle window (frame ${r.frame})`);
    assert.ok(r.distance > 19, "the full 20 m shift is the relocation");
  }
  const { frames: processed } = processFrames(frames);
  const before = processed[29].people.find((p) => p.id === 3);
  assert.ok(dist(before.xy, [9, 0]) < 0.5, "before the cut the dot is at the old position");
  const after = processed[125].people.find((p) => p.id === 3);
  assert.ok(dist(after.xy, [29, 0]) < 0.5, "after the glide the dot is at the new position");
  const steps = stepsOf(processed);
  console.log(`    synthetic relocation: p99=${pct(steps, 99).toFixed(3)} max=${maxOf(steps).toFixed(3)}`);
  assert.ok(pct(steps, 99) <= 0.3 + 1e-9, "the 20 m cut must be eased in, not teleported");
  assert.ok(maxOf(steps) <= 0.5 + 1e-9, "no step may exceed 0.5 m per frame");
});
