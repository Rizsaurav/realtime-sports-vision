"""Unit tests for the built-in greedy IoU fallback tracker
(src/tracking/tracker.py::_GreedyIoUTracker)."""

from tracking.tracker import _GreedyIoUTracker


def _det(box=(0, 0, 10, 10), score=0.9, cls=0):
    x1, y1, x2, y2 = box
    return (x1, y1, x2, y2, score, cls)


def test_id_stable_across_frames():
    t = _GreedyIoUTracker()
    ids = [t.update([_det()])[0][4] for _ in range(5)]
    assert ids == [1] * 5


def test_distinct_detections_get_distinct_ids():
    t = _GreedyIoUTracker()
    out = t.update([_det((0, 0, 10, 10)), _det((50, 50, 60, 60), score=0.8)])
    assert sorted(r[4] for r in out) == [1, 2]


def test_reacquire_after_short_gap_keeps_id():
    t = _GreedyIoUTracker(max_age=5)
    first = t.update([_det()])[0][4]
    t.update([])  # one missed frame, within max_age
    assert t.update([_det()])[0][4] == first


def test_aged_out_track_gets_new_id():
    t = _GreedyIoUTracker(max_age=1)
    first = t.update([_det()])[0][4]
    t.update([])  # age 1: kept
    t.update([])  # age 2 > max_age: dropped
    assert t.update([_det()])[0][4] != first


def test_reset_restarts_ids():
    t = _GreedyIoUTracker()
    t.update([_det()])
    t.reset()
    assert t.update([_det()])[0][4] == 1
