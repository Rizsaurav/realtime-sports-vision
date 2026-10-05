"""Unit tests for src/data/sportsmot.py using a synthetic gt.txt fixture
(tests/fixtures/sportsmot/train/seq01/)."""

from pathlib import Path

import pytest

from data.sportsmot import SportsMOT

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sportsmot"


@pytest.fixture()
def ds():
    return SportsMOT(root=str(FIXTURE_ROOT), split="train")


def test_parses_gt_txt(ds):
    gt = ds.ground_truth("seq01")
    assert gt[1] == [(1, 100.0, 200.0, 150.0, 320.0),
                    (2, 300.0, 250.0, 360.0, 380.0)]
    assert gt[2] == [(1, 105.0, 205.0, 155.0, 325.0),
                    (2, 305.0, 255.0, 365.0, 385.0)]
    assert gt[3] == [(1, 110.0, 210.0, 160.0, 330.0)]


def test_xywh_converted_to_x1y1x2y2(ds):
    _tid, x1, y1, x2, y2 = ds.ground_truth("seq01")[1][0]
    assert (x2 - x1, y2 - y1) == pytest.approx((50.0, 120.0))


def test_missing_gt_returns_empty(ds):
    assert ds.ground_truth("no_such_seq") == {}


def test_sequences_lists_img1_dirs(ds):
    assert ds.sequences() == ["seq01"]


def test_seq_info(ds):
    info = ds.seq_info("seq01")
    assert info["seqLength"] == "3"
    assert info["frameRate"] == "25"


def test_missing_root_raises():
    with pytest.raises(FileNotFoundError):
        SportsMOT(root=str(FIXTURE_ROOT), split="no_such_split")
