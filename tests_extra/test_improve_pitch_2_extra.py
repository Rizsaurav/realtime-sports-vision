"""Extra adversarial tests for the replay view task (tasks/replay_view.md).

The part-1 deliverable, ``src/sideline/web/lib/pitch.js``, is a plain ES
module with no Python counterpart, so these tests pin down the part of the
specification that a behavioural JS suite does not (or only partially)
check:

* the module-purity rules from the spec header ("A plain ES module, named
  exports only, no ``import`` statements, no DOM, no canvas, no
  ``window``/``document``, no timers, no network") and the exact export
  list, checked statically against the source text;
* the exact worked examples of the specification (geometry, mapping,
  person styling, chunk ranges, chunk starts, the FrameCache example and
  its walk-through), executed through Node so they fail if the
  implementation drifts from the spec.

The behavioural checks are the spec's own examples; they are independent
of the JS test suites and stay fast (a couple of short Node invocations).
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PITCH_JS = REPO_ROOT / "src" / "sideline" / "web" / "lib" / "pitch.js"

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node executable not available")

# --------------------------------------------------------------------------
# Static module-purity rules (spec header of Part 1)
# --------------------------------------------------------------------------

def _source():
    return PITCH_JS.read_text(encoding="utf-8")


def test_no_import_statements_or_commonjs_or_dynamic_imports():
    # "A plain ES module, named exports only, no `import` statements"
    src = _source()
    assert not re.search(r"^\s*import\b", src, flags=re.MULTILINE), \
        "pitch.js must not contain import statements"
    assert "require(" not in src, "pitch.js must not use CommonJS require"
    assert "import.meta" not in src, "pitch.js must not use import.meta"
    assert "import(" not in src, "pitch.js must not use dynamic import"


def test_no_dom_timers_or_network_apis():
    # "no DOM, no canvas, no window/document, no timers, no network"
    # (the word "canvas" may appear in comments; runtime APIs may not)
    src = _source()
    for token in (
        "document",
        "window",
        "navigator",
        "screen",
        "localStorage",
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
        "setTimeout",
        "setInterval",
        "requestAnimationFrame",
        "queueMicrotask",
        "performance",
        "new Worker",
    ):
        assert token not in src, f"pitch.js must not reference {token!r}"


def test_named_exports_are_exactly_the_spec_list():
    # "named exports only" with exactly the eight names from the spec.
    src = _source()
    names = re.findall(r"^\s*export\s+(?:const|function|class)\s+([A-Za-z_$][\w$]*)",
                       src, flags=re.MULTILINE)
    assert sorted(names) == sorted([
        "PITCH", "pitchGeometry", "pitchToCanvas", "canvasToPitch",
        "personStyle", "chunkRanges", "chunkStartFor", "FrameCache",
    ]), f"unexpected export names: {names}"
    assert "export default" not in src, "pitch.js must not have a default export"
    assert not re.search(r"^\s*export\s*\{", src, flags=re.MULTILINE), \
        "pitch.js must use inline named exports only"


# --------------------------------------------------------------------------
# Spec worked examples, executed through Node
# --------------------------------------------------------------------------

def _run_node(script: str) -> None:
    with tempfile.TemporaryDirectory(prefix="pitch_extra_") as tmp:
        script_path = Path(tmp) / "check.mjs"
        script_path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            [NODE, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert proc.returncode == 0, (
        f"node check failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
    assert "PITCH_EXTRA_OK" in proc.stdout


def test_spec_examples_pure_functions_via_node():
    _run_node(f"""
import assert from "node:assert/strict";
const m = await import({str(PITCH_JS)!r});

// PITCH: exactly these values, frozen
assert.deepEqual(m.PITCH, {{ length: 105, width: 68, margin: 4 }});
assert.ok(Object.isFrozen(m.PITCH));
assert.deepEqual(Object.keys(m.PITCH).sort(), ["length", "margin", "width"]);

// pitchGeometry: the spec's examples
assert.deepEqual(m.pitchGeometry(1130, 760), {{ k: 10, ox: 565, oy: 380 }});
assert.deepEqual(m.pitchGeometry(1130, 380), {{ k: 5, ox: 565, oy: 190 }});
assert.deepEqual(m.pitchGeometry(226, 1000), {{ k: 2, ox: 113, oy: 500 }});
assert.deepEqual(m.pitchGeometry(0, 500), {{ k: 0, ox: 0, oy: 250 }});
assert.deepEqual(m.pitchGeometry(NaN, 500), {{ k: 0, ox: 0, oy: 250 }});
assert.deepEqual(m.pitchGeometry(-10, 500), {{ k: 0, ox: 0, oy: 250 }});

// pitchToCanvas / canvasToPitch: the spec's examples
const G = {{ k: 10, ox: 565, oy: 380 }};
assert.deepEqual(m.pitchToCanvas([0, 0], G), [565, 380]);
assert.deepEqual(m.pitchToCanvas([52.5, 34], G), [1090, 720]);
assert.deepEqual(m.pitchToCanvas([-52.5, -34], G), [40, 40]);
assert.deepEqual(m.pitchToCanvas([1.25, -2], G), [577.5, 360]);
assert.deepEqual(m.canvasToPitch([565, 380], G), [0, 0]);
assert.deepEqual(m.canvasToPitch([1090, 720], G), [52.5, 34]);
assert.deepEqual(m.canvasToPitch([577.5, 360], G), [1.25, -2]);

// personStyle: the spec's examples
const COLORS = {{ left: "#d03030", right: "#2050c0" }};
assert.deepEqual(
  m.personStyle({{ role: "player", team: "left", jersey: "10" }}, COLORS),
  {{ fill: "#d03030", stroke: "rgba(0,0,0,0.55)", strokeWidth: 1.5, label: "10" }});
assert.equal(
  m.personStyle({{ role: "player", team: "left", jersey: "10" }}, null).fill, "#e4572e");
assert.equal(
  m.personStyle({{ role: "player", team: "left", jersey: "10" }}, null, {{ left: "#111111" }}).fill,
  "#111111");
assert.equal(
  m.personStyle({{ role: "player", team: "right", jersey: "3" }}, null, {{ left: "#111111" }}).fill,
  "#3e8ed0");
assert.deepEqual(
  m.personStyle({{ role: "goalkeeper", team: "right", jersey: null }}, COLORS),
  {{ fill: "#2050c0", stroke: "#ffffff", strokeWidth: 2.5, label: "" }});
assert.deepEqual(
  m.personStyle({{ role: "referee", team: null, jersey: null }}, COLORS),
  {{ fill: "#f2d544", stroke: "rgba(0,0,0,0.55)", strokeWidth: 1.5, label: "" }});
assert.equal(
  m.personStyle({{ role: "player", team: "left" }}, {{ left: "", right: "#2050c0" }}).fill,
  "#e4572e");
console.log("PITCH_EXTRA_OK");
""")


def test_spec_examples_chunks_and_cache_via_node():
    _run_node(f"""
import assert from "node:assert/strict";
const m = await import({str(PITCH_JS)!r});

// chunkRanges: the spec's examples
assert.deepEqual(m.chunkRanges(0, 749, 250),
  [{{ start: 0, end: 249 }}, {{ start: 250, end: 499 }}, {{ start: 500, end: 749 }}]);
assert.deepEqual(m.chunkRanges(0, 600, 250),
  [{{ start: 0, end: 249 }}, {{ start: 250, end: 499 }}, {{ start: 500, end: 600 }}]);
assert.deepEqual(m.chunkRanges(100, 100, 250), [{{ start: 100, end: 100 }}]);
assert.deepEqual(m.chunkRanges(10, 5, 250), []);
assert.deepEqual(m.chunkRanges(5, 14, 5), [{{ start: 5, end: 9 }}, {{ start: 10, end: 14 }}]);
assert.equal(m.chunkRanges(0, 9, 1).length, 10);
assert.throws(() => m.chunkRanges(0, 10, 0), RangeError);
assert.throws(() => m.chunkRanges(0, 10, 2.5), RangeError);
assert.throws(() => m.chunkRanges(0, 10.5, 5), RangeError);

// chunkStartFor: the spec's examples
assert.equal(m.chunkStartFor(0, 0, 250), 0);
assert.equal(m.chunkStartFor(249, 0, 250), 0);
assert.equal(m.chunkStartFor(250, 0, 250), 250);
assert.equal(m.chunkStartFor(499.9, 0, 250), 250);
assert.equal(m.chunkStartFor(1000, 0, 250), 1000);
assert.equal(m.chunkStartFor(100, 100, 250), 100);
assert.equal(m.chunkStartFor(349, 100, 250), 100);
assert.equal(m.chunkStartFor(350, 100, 250), 350);
assert.equal(m.chunkStartFor(99, 100, 250), -150);

// FrameCache: the spec's example (frames {{frame: n}})
const c = new m.FrameCache(2);
const f0 = {{ frame: 0 }}, f1 = {{ frame: 5 }}, f2 = {{ frame: 10 }};
c.put(0, [f0, f1, f2]);
assert.equal(c.frameAt(7), f1);
assert.equal(c.frameAt(10), f2);
assert.equal(c.frameAt(-1), null);
assert.equal(c.frameAt(250), null);
assert.equal(c.frameAt(249), f2);
assert.equal(c.has(0), true);

// FrameCache: the spec's walk-through
const w = new m.FrameCache(2);
w.put(0, [f0]);
w.put(250, [{{ frame: 250 }}]);
assert.deepEqual(w.starts(), [0, 250]);
w.frameAt(10);
assert.deepEqual(w.starts(), [250, 0]);
w.put(500, [{{ frame: 500 }}]);
assert.deepEqual(w.starts(), [0, 500]);
assert.equal(w.size, 2);
assert.equal(w.has(250), false);
w.put(0, [{{ frame: 0 }}]);
assert.deepEqual(w.starts(), [500, 0]);

// determinism: the same walk-through repeated gives the same result
function walk() {{
  const x = new m.FrameCache(2);
  x.put(0, [f0]);
  x.put(250, [{{ frame: 250 }}]);
  x.frameAt(10);
  x.put(500, [{{ frame: 500 }}]);
  return x.starts();
}}
assert.deepEqual(walk(), walk());
assert.deepEqual(walk(), [0, 500]);
console.log("PITCH_EXTRA_OK");
""")
