"""Measured behaviour of the guard on real keyframe homographies.

Fixture: tests/fixtures/real_homographies.json (5 clips, ~75 keyframes each,
1920x1080).  This file checks the v2 acceptance goals:

  (a) at least 90% of all real keyframes are accepted by the guard, and no
      clip is below 80%;
  (b) no clip ends with zero acceptances (a usable calibration is reached);
  (c) the largest view_distance between two consecutive accepted keyframes
      is reported;
  (d) with one keyframe in every ten replaced by a 40 m glitch, at least
      95% of the injected glitches are rejected and at most 5% of the
      genuine keyframes are rejected;
  (e) with a 40 m camera cut at frame 350, the cut is accepted within at
      most 3 keyframes.
"""
import json
from pathlib import Path

import numpy as np

from sideline.calibration_guard import CalibrationGuard, view_distance

FIXTURE = Path(__file__).parent / "fixtures" / "real_homographies.json"


def _load():
    d = json.loads(FIXTURE.read_text())
    size = tuple(d["size"])
    clips = {}
    for name, entries in d["clips"].items():
        clips[name] = [(int(e["frame"]), np.array(e["H"], float).reshape(3, 3))
                       for e in entries]
    return size, clips


_CUT = np.array([[1.0, 0, 40.0], [0, 1, 0], [0, 0, 1]])   # a 40 m camera cut
_GLITCH = _CUT                                             # the injected glitch


def test_a_most_real_keyframes_are_accepted():
    size, clips = _load()
    total = accepted = 0
    for name, Hs in clips.items():
        g = CalibrationGuard()
        cur = None
        acc = n = 0
        for _, H in Hs:
            if not np.isfinite(H).all():
                continue
            cur, dec = g.update(H, cur, size)
            n += 1
            acc += dec.startswith("accepted")
        total += n
        accepted += acc
        pct = 100.0 * acc / n
        print(f"{name}: {acc}/{n} = {pct:.1f}%")
        assert pct >= 80.0, f"{name} below the per-clip minimum"
    pct = 100.0 * accepted / total
    print(f"total: {accepted}/{total} = {pct:.1f}%")
    assert pct >= 90.0


def test_b_no_clip_ends_uncalibrated():
    size, clips = _load()
    for name, Hs in clips.items():
        g = CalibrationGuard()
        cur = None
        acc = 0
        for _, H in Hs:
            if not np.isfinite(H).all():
                continue
            cur, dec = g.update(H, cur, size)
            acc += dec.startswith("accepted")
        assert acc > 0, f"{name}: zero accepted keyframes"
        assert cur is not None, f"{name}: no usable calibration by the end"


def test_c_largest_accepted_pair_jump_is_reported():
    size, clips = _load()
    largest = 0.0
    where = None
    for name, Hs in clips.items():
        g = CalibrationGuard()
        cur = None
        prev = None
        for _, H in Hs:
            if not np.isfinite(H).all():
                continue
            cur, dec = g.update(H, cur, size)
            if dec.startswith("accepted") and prev is not None:
                d = view_distance(cur, prev, size)
                if d > largest:
                    largest, where = d, name
            if dec.startswith("accepted"):
                prev = cur
    print(f"largest view_distance between consecutive accepted keyframes: "
          f"{largest:.2f} m (in {where})")
    assert np.isfinite(largest)
    assert largest > 0.0


def _is_glitch(i):
    # one keyframe in every ten is a glitch; the first keyframe of each
    # clip stays genuine so the lenient first acceptance is not spent on
    # an injected frame
    return i % 10 == 1


def test_d_injected_glitches_are_rejected():
    size, clips = _load()
    glitch_n = glitch_rej = genuine_n = genuine_rej = 0
    for name, Hs in clips.items():
        g = CalibrationGuard()
        cur = None
        for i, (frame, H) in enumerate(Hs):
            if not np.isfinite(H).all():
                continue
            glitch = _is_glitch(i)
            Hc = _GLITCH @ H if glitch else H
            cur, dec = g.update(Hc, cur, size)
            if glitch:
                glitch_n += 1
                glitch_rej += not dec.startswith("accepted")
            else:
                genuine_n += 1
                genuine_rej += not dec.startswith("accepted")
    gj = 100.0 * glitch_rej / glitch_n
    gr = 100.0 * genuine_rej / genuine_n
    print(f"glitches rejected: {glitch_rej}/{glitch_n} = {gj:.1f}%")
    print(f"genuine keyframes rejected: {genuine_rej}/{genuine_n} = {gr:.1f}%")
    assert gj >= 95.0
    assert gr <= 5.0


def test_e_injected_camera_cut_accepted_within_three_keyframes():
    size, clips = _load()
    for name, Hs in clips.items():
        g = CalibrationGuard()
        cur = None
        cut_idx = None
        acc_idx = None
        for i, (frame, H) in enumerate(Hs):
            if not np.isfinite(H).all():
                continue
            Hc = _CUT @ H if frame >= 350 else H
            cur, dec = g.update(Hc, cur, size)
            if frame >= 350 and cut_idx is None:
                cut_idx = i
            if frame >= 350 and dec.startswith("accepted") and acc_idx is None:
                acc_idx = i
        assert cut_idx is not None, f"{name}: no keyframe at/after frame 350"
        assert acc_idx is not None, f"{name}: the cut was never accepted"
        lag = acc_idx - cut_idx
        print(f"{name}: cut accepted {lag} keyframe(s) after the cut")
        assert lag <= 3, f"{name}: cut took {lag} keyframes"
