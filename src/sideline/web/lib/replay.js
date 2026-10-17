// Replay helpers for recorded match frames (see docs/WEBAPP_CONTRACT.md).
// Pure functions plus a Timeline playback clock. No DOM access, no imports.

function clamp01(t) {
  return Math.min(1, Math.max(0, t));
}

function lerpRound(a, b, t) {
  return Math.round(a + (b - a) * t);
}

export function frameIndexAt(frames, frameNumber) {
  let lo = 0;
  let hi = frames.length - 1;
  let ans = -1; // empty list (or no frame <= frameNumber): no index
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].frame <= frameNumber) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

function copyEntity(p) {
  const c = { ...p };
  if (c.xy !== null) c.xy = c.xy.slice();
  c.box = c.box.slice();
  return c;
}

// Lerp xy without rounding and box with rounding between two entities.
function interpolateFields(from, to, t) {
  return {
    xy: [from.xy[0] + (to.xy[0] - from.xy[0]) * t, from.xy[1] + (to.xy[1] - from.xy[1]) * t],
    box: from.box.map((v, i) => lerpRound(v, to.box[i], t)),
  };
}

export function interpolatePeople(a, b, t) {
  t = clamp01(t);
  const out = [];
  for (const p of a.people) {
    let q = null;
    for (const r of b.people) {
      if (r.id === p.id) {
        q = r;
        break; // first id in b wins if ids repeat
      }
    }
    if (q === null) continue;
    if (p.xy === null || q.xy === null) {
      out.push(copyEntity(p));
      continue;
    }
    out.push({ ...p, ...interpolateFields(p, q, t) });
  }
  return out;
}

export function interpolateBall(a, b, t) {
  const A = a.ball;
  const B = b.ball;
  if (A === null || B === null) return null;
  t = clamp01(t);
  if (A.xy === null || B.xy === null) return copyEntity(A);
  return { ...A, ...interpolateFields(A, B, t) };
}

export function formatClock(seconds) {
  if (typeof seconds !== "number") {
    throw new RangeError("seconds must be a number");
  }
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0; // spec: non-finite and negative values give 0
  const tenths = Math.floor(seconds * 10 + 1e-9);
  const minutes = Math.floor(tenths / 600);
  const sec = Math.floor((tenths % 600) / 10);
  const tenth = tenths % 10;
  return minutes + ":" + String(sec).padStart(2, "0") + "." + tenth;
}

// ---------------------------------------------------------------------------
// Calm replay: glitch rejection, per-person smoothing, frame processing.
//
// Recorded positions contain calibration glitches: an entire frame's
// positions can jump by metres or hundreds of metres and come back after a
// few frames, a single person can teleport for a few frames, and the
// coordinate frame can re-anchor (all visible people shift together by a
// small amount and stay there). The functions below work per person over
// time, return fresh data and never modify their inputs.

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function medianOf(values) {
  const s = values.slice().sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length === 0 ? null : s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Glide speed (metres per frame) for easing a relocation in: the calm
// rampRate for moves of 8 m or less, and a fixed slightly faster rate
// beyond 8 m so a big camera re-anchor settles in reasonable time.
function glideRate(d, rampRate) {
  return d > 8 ? 0.29 : rampRate;
}

// Decide, for every frame, whether the coordinate frame itself shifted
// coherently: among the people visible in both consecutive frames, a
// majority moved by comparable amounts. A real camera cut or re-anchoring
// moves everyone by the same amount; a chaotic whole-frame calibration
// glitch scatters the jumps by wildly different amounts, so it does not
// count as a shift. Inputs: frames - an array of recorded frames; options -
// minPeople (minimum visible people to judge a frame, default 5) and agree
// (minimum fraction of visible people whose step magnitude lies within a
// factor 2 of the median step, default 0.6). Output: an array parallel to
// frames; entry i (i > 0) is null or {coherent, shift, medianStep} where
// shift is the median step vector [dx, dy] in metres and medianStep its
// magnitude; entry 0 is always null.
function frameShifts(frames, options = {}) {
  const minPeople = options.minPeople ?? 5;
  const agree = options.agree ?? 0.6;
  const out = new Array(frames.length).fill(null);
  for (let i = 1; i < frames.length; i++) {
    const prev = new Map();
    for (const p of frames[i - 1].people) {
      if (p.xy !== null && p.xy !== undefined) prev.set(p.id, p.xy);
    }
    const dxs = [], dys = [], mags = [];
    for (const p of frames[i].people) {
      if (p.xy === null || p.xy === undefined) continue;
      const q = prev.get(p.id);
      if (!q) continue;
      const dx = p.xy[0] - q[0];
      const dy = p.xy[1] - q[1];
      dxs.push(dx);
      dys.push(dy);
      mags.push(Math.hypot(dx, dy));
    }
    if (mags.length < minPeople) continue;
    const medianStep = medianOf(mags);
    const agreeing = mags.filter((m) => m >= medianStep / 2 && m <= medianStep * 2).length;
    out[i] = {
      coherent: agreeing / mags.length >= agree,
      shift: [medianOf(dxs), medianOf(dys)],
      medianStep,
    };
  }
  return out;
}

// Reject position glitches per person over time. Inputs: frames - an array
// of recorded frames in increasing frame order; options - maxStep (implied
// speed limit in metres per frame, default 1.0 = 25 m/s), settle (number
// of consistent frames before a candidate relocation is accepted, default
// 6), settleTol (candidate points must all lie within this many metres of
// each other, given the person's own motion of at most maxStep per frame
// of gap, default 1.5), slowSettle (a non-coherent candidate is still
// accepted after this many consistent frames, default settle * 3), rampRate
// (metres per frame the accepted relocation eases in over, default 0.25;
// relocations of more than 8 m glide at a fixed 0.29 m per frame),
// maxAutoRelocate (a non-coherent candidate is only accepted when the jump
// is at most this many metres, default 25; coherent frame shifts have no
// distance cap), plus the frameShifts options (minPeople, agree). Output: a
// new object {frames, relocations}: frames is a new array of frame copies
// whose people carry the processed position (a new xy array) and ghost -
// true on the frames where the position was carried forward from the last
// accepted point; a person seen for the first time while most visible
// people are glitched gets xy null (no anchor until a trustworthy frame),
// and so does a first appearance on a frame where nobody is anchored yet
// (the head of a freshly loaded window) unless the frame is not a big
// whole-frame shift and the person's own step to or from an adjacent frame
// is at most maxStep (the raw position of a glitch frame is not a safe
// place to start). relocations lists every accepted relocation as
// {id, jumpFrame, frame, from, to, distance, k} where frame is the
// acceptance frame and k the number of frames the glide takes.
// The implied speed of a point is its distance from the last accepted
// position divided by the frame gap since it (a person who was absent can
// legitimately be far away); a point exceeding maxStep is rejected and
// replaced by the carried position. A rejected position that then stays
// consistent for settle frames - every candidate within settleTol of each
// other, allowing the person's own motion at up to maxStep per frame of
// gap - is accepted as a genuine relocation: always when the frame of the
// jump is a coherent frame shift that the person took part in (a real
// camera cut or re-anchoring moves everyone, by any distance), or for
// anyone after slowSettle consistent frames provided the jump is at most
// maxAutoRelocate (a farther non-coherent candidate is a wrong match, not
// a camera cut, and is never accepted). After acceptance the dot glides in
// a straight line from the old position to the new one over
// ceil(distance / rate) frames, so the drawn dot eases into the new
// position instead of teleporting. A person reappearing after an absence
// more than 2 m away but at a plausible speed also glides (the gap frames
// cannot show where they went). The input is never modified.
export function rejectGlitches(frames, options = {}) {
  const maxStep = options.maxStep ?? 1.0;
  const settle = options.settle ?? 6;
  const settleTol = options.settleTol ?? 1.5;
  const slowSettle = options.slowSettle ?? settle * 3;
  const rampRate = options.rampRate ?? 0.25;
  const maxAutoRelocate = options.maxAutoRelocate ?? 25;
  if (!Array.isArray(frames)) throw new TypeError("frames must be an array");
  const shifts = frameShifts(frames, options);
  const states = new Map();
  const relocations = [];
  const out = new Array(frames.length);

  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    // Judge people that already have an anchor first: their ghost count
    // decides whether this frame is trustworthy enough to anchor new ones
    // (a frame where most visible people jumped is a bad frame, and
    // anchoring a first appearance inside it would freeze the person at a
    // glitch position).
    const results = new Map();
    let anchoredVisible = 0;
    let ghosted = 0;
    for (const p of f.people) {
      if (p.xy === null || p.xy === undefined) continue;
      const st = states.get(p.id);
      if (!st) continue;
      anchoredVisible++;
      // An accepted relocation that is still easing in: the dot glides
      // from the old position toward the person's current raw position,
      // in frame time, at a calm rate (even across absences). Each step is
      // capped at the glide rate. When the glide time is up, a small
      // residual is chased once or twice, then normal logic takes over.
      if (st.ramp) {
        const r = st.ramp;
        const tt = (f.frame - r.fromFrame) / r.k;
        if (tt < 1) {
          let xy = [
            r.from[0] + (p.xy[0] - r.from[0]) * tt,
            r.from[1] + (p.xy[1] - r.from[1]) * tt,
          ];
          const s = distance(st.last, xy);
          if (s > r.rate) {
            xy = [
              st.last[0] + (xy[0] - st.last[0]) * (r.rate / s),
              st.last[1] + (xy[1] - st.last[1]) * (r.rate / s),
            ];
          }
          st.last = xy.slice();
          st.lastFrame = f.frame;
          st.lastSeen = f.frame;
          st.win = [];
          results.set(p.id, { kind: "ok", xy, ghost: false });
          continue;
        }
        const d2 = distance(st.last, p.xy);
        if (d2 > maxStep && d2 <= maxAutoRelocate && r.chases < 2) {
          // The player moved on (or was unseen) during the glide: keep
          // gliding after the current position.
          const rate = glideRate(d2, rampRate);
          st.ramp = {
            from: st.last.slice(), fromFrame: f.frame, k: Math.max(1, Math.ceil(d2 / rate)), rate, chases: r.chases + 1,
          };
          st.lastFrame = f.frame;
          st.lastSeen = f.frame;
          st.win = [];
          results.set(p.id, { kind: "ok", xy: st.last.slice(), ghost: false });
          continue;
        } else {
          if (st.rampRec) {
            st.rampRec.endFrame = f.frame;
            st.rampRec = null;
          }
          st.ramp = null;
          // fall through: normal accept/ghost logic (the dot is already
          // close, or a fresh candidate run starts)
        }
      }
      const gap = Math.max(1, f.frame - st.lastFrame);
      const d = distance(st.last, p.xy);
      const absent = f.frame - st.lastSeen > 1; // not visible in between
      const speed = absent ? d / gap : d; // implied speed in m per frame
      if (speed <= maxStep) {
        if (absent && d > 2 && d <= maxAutoRelocate) {
          // Reappearance far away but at a plausible speed: the person was
          // simply unseen for a while. The dot stays put this frame and
          // glides to the new position over the following frames.
          const sinceSeen = Math.max(1, f.frame - st.lastSeen);
          const rate = glideRate(d, rampRate);
          const k = Math.max(1, Math.ceil(d / rate));
          const from = st.last.slice();
          const jumpFrame = st.lastSeen + 1; // first unseen frame
          st.lastSeen = f.frame;
          st.win = [];
          st.ramp = { from, fromFrame: f.frame, k, rate, chases: 0 };
          st.lastFrame = f.frame;
          const rec = {
            id: p.id, jumpFrame, frame: f.frame, from, to: p.xy.slice(), distance: d, k, endFrame: null,
          };
          relocations.push(rec);
          st.rampRec = rec;
          results.set(p.id, { kind: "ok", xy: from.slice(), ghost: false });
          continue;
        }
        st.last = p.xy.slice();
        st.lastFrame = f.frame;
        st.win = [];
        results.set(p.id, { kind: "ok", xy: p.xy.slice(), ghost: false });
        st.lastSeen = f.frame;
        continue;
      }
      // Too fast to be real: a glitch candidate.
      let res = { kind: "ghost", xy: st.last.slice(), ghost: true };
      // Cluster stability: the candidate extends the window only when it
      // is within settleTol of every candidate already in the window,
      // allowing the person's own motion at up to maxStep per frame of gap
      // (the spec's "consecutive steps below maxStep and all within
      // settleTol of each other"); a glitch that wanders, oscillates or
      // drifts faster than a person can move restarts the window instead.
      const win = st.win;
      const grows =
        win.length > 0 &&
        win.every((w) => distance(w.xy, p.xy) <= settleTol + maxStep * (f.frame - w.frame));
      if (grows) {
        win.push({ xy: p.xy.slice(), frame: f.frame });
      } else {
        st.win = [{ xy: p.xy.slice(), frame: f.frame, idx: i }];
      }
      if (st.win.length >= settle) {
        // The jump may have been recorded one or two frames late, so the
        // frame-shift check looks at the window start and the two frames
        // before it.
        const jump = [st.win[0].xy[0] - st.last[0], st.win[0].xy[1] - st.last[1]];
        const jumpDist = distance(st.last, st.win[0].xy);
        let coherent = false;
        for (const oi of [st.win[0].idx, st.win[0].idx - 1, st.win[0].idx - 2]) {
          const info = oi >= 1 ? shifts[oi] : null;
          if (info && info.coherent) {
            const tol = Math.max(3, info.medianStep * 2);
            if (distance(jump, info.shift) <= tol) { coherent = true; break; }
          }
        }
        // A coherent frame shift the person took part in is a genuine
        // relocation (accepted after settle frames, by any distance: a
        // real camera cut or re-anchoring moves everyone). Anything else
        // is accepted only after slowSettle stable frames and only when
        // the jump is at most maxAutoRelocate: a farther non-coherent
        // candidate is a wrong match, not a camera cut, and is never
        // accepted.
        const accept = coherent || (st.win.length >= slowSettle && jumpDist <= maxAutoRelocate);
        if (accept) {
          const from = st.last.slice();
          const to = p.xy.slice();
          const jumpFrame = st.win[0].frame;
          const dRel = distance(from, to);
          const rate = glideRate(dRel, rampRate);
          const k = Math.max(1, Math.ceil(dRel / rate));
          st.last = from.slice(); // the glide starts at the old position
          st.lastFrame = f.frame;
          st.lastSeen = f.frame;
          st.win = [];
          st.ramp = { from, fromFrame: f.frame, k, rate, chases: 0 };
          const rec = { id: p.id, jumpFrame, frame: f.frame, from, to, distance: dRel, k, endFrame: null };
          relocations.push(rec);
          st.rampRec = rec;
          res = { kind: "ok", xy: from.slice(), ghost: false };
        }
      }
      if (res.kind === "ghost") ghosted++;
      st.lastSeen = f.frame;
      results.set(p.id, res);
    }
    // A person who disappears in the middle of a relocation glide: once
    // the glide's time is up, end it here (the dot stays put; the next
    // appearance is a fresh event, judged normally).
    const visibleNow = new Set();
    for (const p of f.people) if (p.xy !== null && p.xy !== undefined) visibleNow.add(p.id);
    for (const [id, st] of states) {
      if (st.ramp && !visibleNow.has(id) && (f.frame - st.ramp.fromFrame) / st.ramp.k >= 1) {
        if (st.rampRec) {
          st.rampRec.endFrame = f.frame;
          st.rampRec = null;
        }
        st.ramp = null;
      }
    }
    const reliable = anchoredVisible > 0 && ghosted * 2 < anchoredVisible;
    // Nobody anchored is visible on this frame (the head of a freshly
    // loaded window): the ghost majority cannot judge it, and the raw
    // positions are not trusted blindly either - a whole-frame glitch at
    // the window head would pin everyone to the glitch position for the
    // whole window (the return to the true track is farther than maxStep,
    // so it is carried as a ghost and never accepted). Such a first
    // appearance is anchored only when the frame is not a big whole-frame
    // shift and the person's own position is continuous with an adjacent
    // frame (step at most maxStep to or from the same person next frame).
    const headBigShift = !!(shifts[i] && shifts[i].medianStep > maxStep);
    const headQuiet = (fr, q) => {
      if (!fr) return false;
      for (const r of fr.people) {
        if (r.id === q.id && r.xy !== null && r.xy !== undefined && distance(r.xy, q.xy) <= maxStep) {
          return true;
        }
      }
      return false;
    };

    const people = f.people.map((p) => {
      if (p.xy === null || p.xy === undefined) {
        return { ...p, xy: null, ghost: false };
      }
      const r = results.get(p.id);
      if (r !== undefined) return { ...p, xy: r.xy, ghost: r.ghost };
      // First appearance of this id: anchor only in a trustworthy frame,
      // otherwise emit no position at all (the raw position of a bad frame
      // is not a safe place to start).
      if (
        reliable ||
        (anchoredVisible === 0 && !headBigShift &&
          (headQuiet(i > 0 ? frames[i - 1] : null, p) ||
            headQuiet(i + 1 < frames.length ? frames[i + 1] : null, p)))
      ) {
        states.set(p.id, { last: p.xy.slice(), lastFrame: f.frame, lastSeen: f.frame, win: [], ramp: null });
        return { ...p, xy: p.xy.slice(), ghost: false };
      }
      return { ...p, xy: null, ghost: false };
    });
    out[i] = { ...f, people };
  }
  return { frames: out, relocations };
}

// Smooth one person's track. Inputs: points - an array of {frame, xy} (and
// any further fields, copied through) in increasing frame order; options -
// span (smoothing window width in frames, about 0.4 s, default 10), maxGap
// (do not smooth across frame gaps larger than this, default 8) and
// maxSpeed (speed clamp in metres per frame, default 0.5). The interior
// points are replaced by the mean of the nearby points (no smoothing across
// gaps larger than maxGap); the first and last points are kept as-is by
// the smoothing; the speed clamp then limits every consecutive pair to
// maxSpeed metres per frame of gap, clamping forward from the first point
// and backward from the last, so both the first and the final point are
// kept as-is and the contraction of an over-fast tail step is distributed
// over the interior points. Returns a new array of the same length with
// new xy arrays; the input is never modified.
export function smoothTrack(points, options = {}) {
  const span = options.span ?? 10;
  const maxGap = options.maxGap ?? 8;
  const maxSpeed = options.maxSpeed ?? 0.5;
  if (!Array.isArray(points)) throw new TypeError("points must be an array");
  const n = points.length;
  const out = points.map((p) => ({ ...p }));
  for (let i = 1; i < n - 1; i++) {
    const half = span / 2;
    const win = [i];
    for (let j = i - 1; j >= 0; j--) {
      if (points[i].frame - points[j].frame > half) break;
      if (points[j + 1].frame - points[j].frame > maxGap) break;
      win.push(j);
    }
    for (let j = i + 1; j < n; j++) {
      if (points[j].frame - points[i].frame > half) break;
      if (points[j].frame - points[j - 1].frame > maxGap) break;
      win.push(j);
    }
    let sx = 0, sy = 0;
    for (const j of win) {
      sx += points[j].xy[0];
      sy += points[j].xy[1];
    }
    out[i].xy = [sx / win.length, sy / win.length];
  }
  for (let i = 1; i < n - 1; i++) {
    const gap = Math.max(1, points[i].frame - points[i - 1].frame);
    const allowed = maxSpeed * gap;
    const a = out[i - 1].xy;
    const b = out[i].xy;
    const d = distance(a, b);
    if (d > allowed && d > 0) {
      const t = allowed / d;
      out[i].xy = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    }
  }
  // Backward pass: the final point is kept as-is, so an over-fast step into
  // it is contracted at the interior point instead (the first point is
  // never touched, it is the anchor of the forward pass).
  for (let i = n - 1; i >= 2; i--) {
    const gap = Math.max(1, points[i].frame - points[i - 1].frame);
    const allowed = maxSpeed * gap;
    const a = out[i - 1].xy;
    const b = out[i].xy;
    const d = distance(a, b);
    if (d > allowed && d > 0) {
      const t = allowed / d;
      out[i - 1].xy = [b[0] + (a[0] - b[0]) * t, b[1] + (a[1] - b[1]) * t];
    }
  }
  return out; // the first and the final point are kept as-is
}

// Full calm-replay pipeline over a list of frames. Inputs: frames - an
// array of recorded frames in increasing frame order; options - every
// rejectGlitches option plus enabled (default true; when false the frames
// are returned as a deep copy, unchanged, with no relocations) and
// minAppearances (people with fewer whole-match appearances than this are
// marked hidden, default 12) and matchAppearances (a Map of id -> raw
// appearances the caller has already counted over the match so far, which
// may include this list; when present, an id is hidden only when the
// whole-match total - the maximum of that count and the appearances in
// this list - is below minAppearances, so a player who appears in fewer
// than minAppearances frames of the current window but often over the
// match is still drawn). Output: {frames, relocations, hidden} where
// frames has the same shape as the rejectGlitches output with each
// person's position additionally smoothed per person and hidden a Set of
// the ids to not draw. The input is never modified.
export function processFrames(frames, options = {}) {
  if (!Array.isArray(frames)) throw new TypeError("frames must be an array");
  const minAppearances = options.minAppearances ?? 12;
  const matchAppearances = options.matchAppearances ?? null;
  if (options.enabled === false) {
    const copy = frames.map((f) => ({
      ...f,
      people: f.people.map((p) => {
        const c = { ...p };
        if (c.xy !== null && c.xy !== undefined) c.xy = c.xy.slice();
        return c;
      }),
    }));
    return { frames: copy, relocations: [], hidden: new Set() };
  }
  const { frames: rejected, relocations } = rejectGlitches(frames, options);
  const counts = new Map();
  const byId = new Map();
  rejected.forEach((f, fi) => {
    f.people.forEach((p, pi) => {
      if (p.xy === null || p.xy === undefined) return;
      counts.set(p.id, (counts.get(p.id) || 0) + 1);
      if (!byId.has(p.id)) byId.set(p.id, []);
      byId.get(p.id).push({ frame: f.frame, xy: p.xy, fi, pi });
    });
  });
  const hidden = new Set();
  for (const [id, count] of counts) {
    const total = matchAppearances ? Math.max(count, matchAppearances.get(id) || 0) : count;
    if (total < minAppearances) hidden.add(id);
  }
  const smoothOpts = { span: options.span, maxGap: options.maxGap, maxSpeed: options.maxSpeed };
  const smoothedXY = new Map(); // id -> Map("fi:pi" -> smoothed xy)
  for (const [id, pts] of byId) {
    const s = smoothTrack(pts, smoothOpts);
    const m = new Map();
    for (let j = 0; j < s.length; j++) m.set(pts[j].fi + ":" + pts[j].pi, s[j].xy);
    smoothedXY.set(id, m);
  }
  const outFrames = rejected.map((f, fi) => ({
    ...f,
    people: f.people.map((p, pi) => {
      if (p.xy === null || p.xy === undefined) return { ...p, hidden: hidden.has(p.id) };
      const m = smoothedXY.get(p.id);
      const xy = m ? m.get(fi + ":" + pi) : null;
      return { ...p, xy: xy !== null && xy !== undefined ? xy : p.xy.slice(), hidden: hidden.has(p.id) };
    }),
  }));
  return { frames: outFrames, relocations, hidden };
}

const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];

export class Timeline {
  constructor({ frames, fps = 25, now = () => performance.now() } = {}) {
    if (typeof fps !== "number" || !Number.isFinite(fps) || fps <= 0) {
      throw new RangeError("fps must be a finite number greater than 0");
    }
    if (frames === undefined) frames = []; // missing value: no recorded frames
    if (!Array.isArray(frames)) {
      throw new RangeError("frames must be an array of recorded frames");
    }
    this.frames = frames;
    this.fps = fps;
    this._now = now;
    this._position = 0;
    this._playing = false;
    this._speed = 1;
    this._last = 0;
  }

  get duration() {
    const f = this.frames;
    if (f.length < 2) return 0; // empty or single-frame list
    return (f[f.length - 1].frame - f[0].frame) / this.fps;
  }

  get position() {
    return this._position;
  }

  get playing() {
    return this._playing;
  }

  get speed() {
    return this._speed;
  }

  _advance() {
    const n = this._now();
    const elapsed = Math.max(0, n - this._last);
    this._last = n;
    if (this._playing) {
      this._position += (elapsed * this._speed) / 1000;
      if (this._position >= this.duration) {
        this._position = this.duration;
        this._playing = false;
      }
    }
  }

  play() {
    if (this._playing) return;
    if (this._position >= this.duration) this._position = 0;
    this._last = this._now();
    this._playing = true;
  }

  pause() {
    this._advance();
    this._playing = false;
  }

  seek(seconds) {
    if (typeof seconds !== "number" || Number.isNaN(seconds)) {
      throw new RangeError("seek target must be a number");
    }
    this._last = this._now();
    this._position = Math.min(Math.max(seconds, 0), this.duration); // clamps past-the-end seeks, incl. +/-Infinity
  }

  setSpeed(s) {
    if (!SPEEDS.includes(s)) {
      throw new RangeError("speed must be one of 0.25, 0.5, 1, 2, 4 or 8");
    }
    this._advance();
    this._speed = s;
  }

  tick() {
    this._advance();
    return this.current();
  }

  current() {
    const f = this.frames;
    if (f.length === 0) return null; // empty list: no view
    let fn = f[0].frame + this._position * this.fps;
    fn = Math.round(fn * 1e6) / 1e6;
    let index = frameIndexAt(f, fn);
    if (index === -1) index = 0;
    const a = f[index];
    const b = index + 1 < f.length ? f[index + 1] : a; // single frame (or at the end): b is a itself
    const t = b === a ? 0 : clamp01((fn - a.frame) / (b.frame - a.frame));
    return { index, t, a, b };
  }
}
