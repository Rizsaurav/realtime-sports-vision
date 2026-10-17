// Pitch mapping, person styling and frame-chunk helpers for the replay panel.
// Pure functions plus a FrameCache. No DOM, no imports.

// SoccerNet pitch dimensions in metres (see docs/WEBAPP_CONTRACT.md): 105 m
// length, 68 m width, 4 m margin around the playing area. Frozen object with
// exactly these three read-only values.
export const PITCH = Object.freeze({ length: 105, width: 68, margin: 4 });

// Compute the canvas transform for the pitch. Inputs: width, height - the
// canvas size in pixels; a non-finite or negative value is treated as 0.
// Output: a new plain {k, ox, oy} object where k is the scale in pixels per
// metre (min(width/113, height/76)) and ox, oy the canvas-pixel offset of
// the centre spot (the canvas middle).
export function pitchGeometry(width, height) {
  const w = Number.isFinite(width) && width >= 0 ? Math.max(0, width) : 0;
  const h = Number.isFinite(height) && height >= 0 ? Math.max(0, height) : 0;
  const k = Math.min(
    w / (PITCH.length + 2 * PITCH.margin),
    h / (PITCH.width + 2 * PITCH.margin)
  );
  return { k, ox: w / 2, oy: h / 2 };
}

// Map a pitch position to canvas pixels. Inputs: xy - [x, y] in metres
// (centre-spot origin, x along the pitch, y across; y is not flipped); geom
// - an object from pitchGeometry. Output: a new [X, Y] array in canvas
// pixels, with no rounding and no clipping, so points off the pitch map off
// the pitch rectangle.
export function pitchToCanvas(xy, geom) {
  return [geom.ox + xy[0] * geom.k, geom.oy + xy[1] * geom.k];
}

// Inverse of pitchToCanvas. Inputs: point - [X, Y] in canvas pixels; geom -
// an object from pitchGeometry. Output: a new [x, y] array in metres; a new
// [0, 0] when geom.k is 0 (instead of dividing by zero).
export function canvasToPitch(point, geom) {
  if (geom.k === 0) return [0, 0];
  return [(point[0] - geom.ox) / geom.k, (point[1] - geom.oy) / geom.k];
}

const BUILTIN = { left: "#e4572e", right: "#3e8ed0" };

function usableColor(v) {
  return typeof v === "string" && v.length > 0;
}

// Resolve the drawing style of one tracked person. Inputs: person -
// {role, team, jersey} (null/undefined behaves like {}); colors - the
// frame's per-side fill colours; fallback - a {left, right} map of hex
// strings; colors and fallback may each be null/undefined. Output: a new
// {fill, stroke, strokeWidth, label} object: fill and stroke are CSS colour
// strings, strokeWidth the outline width in canvas pixels, label the jersey
// as a string ("" when null, undefined or "").
export function personStyle(person, colors, fallback) {
  const p = person !== null && person !== undefined ? person : {};
  const c = colors !== null && colors !== undefined ? colors : {};
  const f = fallback !== null && fallback !== undefined ? fallback : {};
  let fill;
  if (p.role === "referee") {
    fill = "#f2d544";
  } else if (p.team === "left" || p.team === "right") {
    const side = p.team;
    fill = usableColor(c[side]) ? c[side] : usableColor(f[side]) ? f[side] : BUILTIN[side];
  } else {
    fill = "#9aa59f";
  }
  const goalkeeper = p.role === "goalkeeper";
  const jersey = p.jersey;
  const label = jersey === null || jersey === undefined || jersey === "" ? "" : String(jersey);
  return {
    fill,
    stroke: goalkeeper ? "#ffffff" : "rgba(0,0,0,0.55)",
    strokeWidth: goalkeeper ? 2.5 : 1.5,
    label,
  };
}

// Validate size: an integer number of frame numbers per block, >= 1.
// Throws RangeError otherwise.
function checkSize(size) {
  if (!Number.isInteger(size) || size < 1) {
    throw new RangeError("size must be an integer >= 1");
  }
}

// Validate first: an integer frame number. Throws RangeError otherwise.
function checkFirst(first) {
  if (!Number.isInteger(first)) {
    throw new RangeError("first must be an integer");
  }
}

// Split the inclusive frame-number range first..last into fetch blocks.
// Inputs: first, last - integer frame numbers (negative allowed); size -
// integer frame numbers per block, >= 1; any invalid argument throws
// RangeError. Output: a new array of {start, end} objects with inclusive
// frame numbers; the first block starts at first, each next at the previous
// start + size, and only the last block can be shorter than size; [] when
// last < first.
export function chunkRanges(first, last, size) {
  checkSize(size);
  checkFirst(first);
  if (!Number.isInteger(last)) {
    throw new RangeError("last must be an integer");
  }
  const ranges = [];
  for (let start = first; start <= last; start += size) {
    ranges.push({ start, end: Math.min(start + size - 1, last) });
  }
  return ranges;
}

// Start of the block that contains frameNumber. Inputs: frameNumber - any
// number (a frame number; non-integers fine, not validated); first - the
// integer first frame number of the match; size - integer frame numbers per
// block, >= 1; first and size are validated and throw RangeError otherwise.
// Output: the chunk start as a frame number:
// first + floor((frameNumber - first) / size) * size.
export function chunkStartFor(frameNumber, first, size) {
  checkSize(size);
  checkFirst(first);
  return first + Math.floor((frameNumber - first) / size) * size;
}

// Least-recently-used cache of frame chunks. A chunk is an array of frame
// objects in increasing `frame` order (frame numbers, integers), keyed by
// its start frame number; one chunk covers chunkSize frame numbers,
// start .. start + chunkSize - 1.
export class FrameCache {
  // Inputs: maxChunks - the maximum number of chunks held (integer >= 1);
  // chunkSize - frame numbers covered by one chunk (integer >= 1, default
  // 250); otherwise RangeError.
  constructor(maxChunks, chunkSize = 250) {
    if (!Number.isInteger(maxChunks) || maxChunks < 1) {
      throw new RangeError("maxChunks must be an integer >= 1");
    }
    if (!Number.isInteger(chunkSize) || chunkSize < 1) {
      throw new RangeError("chunkSize must be an integer >= 1");
    }
    this.maxChunks = maxChunks;
    this.chunkSize = chunkSize;
    this._chunks = new Map(); // start -> frames (stored by reference)
    this._order = []; // start values, least recently used first
  }

  // Number of chunks currently held (getter, not a method).
  get size() {
    return this._order.length;
  }

  // True when a chunk is stored under exactly this start (frame number),
  // else false. Does not change the recency order.
  has(start) {
    return this._chunks.has(start);
  }

  // Store frames (an array, kept by reference, empty allowed) under start
  // (an integer frame number), as the most recently used chunk; an existing
  // start is replaced and refreshed. Least recently used chunks beyond
  // maxChunks are evicted, never the chunk just put. RangeError for a
  // non-integer start, TypeError for a non-array frames; nothing changes on
  // error. Returns undefined.
  put(start, frames) {
    if (!Number.isInteger(start)) {
      throw new RangeError("start must be an integer");
    }
    if (!Array.isArray(frames)) {
      throw new TypeError("frames must be an array");
    }
    const i = this._order.indexOf(start);
    if (i !== -1) this._order.splice(i, 1);
    this._chunks.set(start, frames);
    this._order.push(start);
    while (this._order.length > this.maxChunks) {
      this._chunks.delete(this._order.shift());
    }
  }

  // The last frame in the cached chunk covering frameNumber (a finite
  // number) whose frame is <= frameNumber, as the original object
  // reference; null when frameNumber is not finite, no chunk covers it, or
  // no frame in it qualifies. A covering chunk becomes the most recently
  // used.
  frameAt(frameNumber) {
    if (typeof frameNumber !== "number" || !Number.isFinite(frameNumber)) return null;
    const end = frameNumber;
    let best = null;
    for (const start of this._order) {
      if (start <= end && end <= start + this.chunkSize - 1) {
        if (best === null || start > best) best = start;
      }
    }
    if (best === null) return null;
    const i = this._order.indexOf(best);
    this._order.splice(i, 1);
    this._order.push(best);
    const frames = this._chunks.get(best);
    let lo = 0, hi = frames.length - 1, ans = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (frames[mid].frame <= frameNumber) {
        ans = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return ans === -1 ? null : frames[ans];
  }

  // A new array of the cached chunk start frame numbers, ordered from least
  // recently used to most recently used.
  starts() {
    return this._order.slice();
  }

  // Remove all chunks. Returns undefined.
  clear() {
    this._chunks.clear();
    this._order.length = 0;
  }
}
