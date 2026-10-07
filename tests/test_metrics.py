"""Unit tests for src/benchmark/metrics.py.

Covers: the `_iou` util, `compute_detection_metrics` (greedy matching +
11-point mAP), and `compute_tracking_metrics` — both the motmetrics path
and the no-motmetrics fallback.
"""

import sys

import pytest

from benchmark.metrics import (
    _iou,
    compute_detection_metrics,
    compute_tracking_metrics,
)


# --- _iou ---------------------------------------------------------------

def test_iou_identical_boxes():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)


def test_iou_partial_overlap():
    # intersection=1, union=4+4-1=7
    assert _iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)


def test_iou_zero_area_box():
    assert _iou((5, 5, 5, 5), (0, 0, 10, 10)) == pytest.approx(0.0)


# --- mAP ----------------------------------------------------------------

GT_ONE = {1: [(1, 0, 0, 10, 10)]}


def test_map_perfect_prediction():
    r = compute_detection_metrics({1: [(0, 0, 10, 10, 0.9, 0)]}, GT_ONE)
    assert r["map"] == pytest.approx(1.0)
    assert (r["n_gt"], r["n_pred"]) == (1, 1)


def test_map_no_predictions():
    r = compute_detection_metrics({}, GT_ONE)
    assert r["map"] == pytest.approx(0.0)
    assert (r["n_gt"], r["n_pred"]) == (1, 0)


def test_map_empty_inputs():
    r = compute_detection_metrics({}, {})
    assert r["map"] == pytest.approx(0.0)
    assert (r["n_gt"], r["n_pred"]) == (0, 0)


def test_map_partial_recall():
    # one of two GT boxes found -> recall caps at 0.5 -> AP = 6/11
    gt = {1: [(1, 0, 0, 10, 10), (2, 50, 50, 60, 60)]}
    r = compute_detection_metrics({1: [(0, 0, 10, 10, 0.9, 0)]}, gt)
    assert r["map"] == pytest.approx(6 / 11)
    assert (r["n_gt"], r["n_pred"]) == (2, 1)


def test_map_averages_over_frames():
    gt = {1: [(1, 0, 0, 10, 10)], 2: [(1, 0, 0, 10, 10)]}
    preds = {1: [(0, 0, 10, 10, 0.9, 0)]}  # frame 2 missed entirely
    r = compute_detection_metrics(preds, gt)
    assert r["map"] == pytest.approx(0.5)


def test_map_predictions_ranked_by_score():
    # the lower-scored near-duplicate must not steal the GT match
    preds = {1: [(0, 0, 10, 10, 0.9, 0), (1, 1, 9, 9, 0.1, 0)]}
    r = compute_detection_metrics(preds, GT_ONE)
    assert r["map"] == pytest.approx(1.0)
    assert r["n_pred"] == 2


# --- tracking: motmetrics path ------------------------------------------

GT_TRACK = {1: [(1, 0, 0, 10, 10)], 2: [(1, 0, 0, 10, 10)]}


def _perfect_tracks():
    return [(1, [(0, 0, 10, 10, 1, 0)]), (2, [(0, 0, 10, 10, 1, 0)])]


def test_tracking_mota_perfect_motmetrics():
    pytest.importorskip("motmetrics")
    r = compute_tracking_metrics(_perfect_tracks(), GT_TRACK)
    assert r["mota"] == pytest.approx(1.0)
    assert r["id_switches"] == 0
    # motmetrics MOTP is a *distance* (1 - IoU): 0.0 is perfect
    assert r["motp"] == pytest.approx(0.0)


def test_tracking_mota_all_misses_motmetrics():
    pytest.importorskip("motmetrics")
    r = compute_tracking_metrics([(1, []), (2, [])], GT_TRACK)
    assert r["mota"] == pytest.approx(0.0)
    assert r["id_switches"] == 0


# --- tracking: no-motmetrics fallback -----------------------------------

def _disable_motmetrics(monkeypatch):
    # makes `import motmetrics` raise ImportError inside metrics.py
    monkeypatch.setitem(sys.modules, "motmetrics", None)


def test_tracking_mota_fallback_perfect(monkeypatch):
    _disable_motmetrics(monkeypatch)
    r = compute_tracking_metrics(_perfect_tracks(), GT_TRACK)
    assert r["mota"] == pytest.approx(1.0)
    assert r["motp"] is None
    assert r["id_switches"] == 0


def test_tracking_mota_fallback_misses_and_id_switch(monkeypatch):
    _disable_motmetrics(monkeypatch)
    gt = {1: [(1, 0, 0, 10, 10), (2, 50, 50, 60, 60)],
          2: [(1, 0, 0, 10, 10), (2, 50, 50, 60, 60)]}
    # frame 1: GT track 1 carried by hyp id 5; frame 2: by hyp id 7
    # (one ID switch); GT track 2 missed on both frames
    tracks = [(1, [(0, 0, 10, 10, 5, 0)]),
              (2, [(0, 0, 10, 10, 7, 0)])]
    r = compute_tracking_metrics(tracks, gt)
    assert r["id_switches"] == 1
    # misses=2, fp=0, switches=1, total_gt=4 -> 1 - 3/4
    assert r["mota"] == pytest.approx(0.25)
