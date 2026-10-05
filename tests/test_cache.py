"""Unit tests for src/streaming/cache.py (FrameFeatureCache)."""

import numpy as np
import pytest

from streaming.cache import FrameFeatureCache


def test_keyframe_schedule():
    c = FrameFeatureCache(keyframe_interval=3)
    seen = []
    for _ in range(7):
        seen.append(c.is_keyframe())
        c.step()
    assert seen == [True, False, False, True, False, False, True]


def test_update_and_get():
    c = FrameFeatureCache()
    feats = np.ones(8)
    c.update(feats)
    assert c.get() is feats


def test_hard_replace_on_keyframe():
    c = FrameFeatureCache(ema_decay=0.0)
    c.update(np.ones(4))
    c.update(np.zeros(4))
    np.testing.assert_array_equal(c.get(), np.zeros(4))


def test_ema_blend():
    c = FrameFeatureCache(ema_decay=0.5)
    c.update(np.ones(4))
    c.update(np.zeros(4))
    np.testing.assert_allclose(c.get(), np.full(4, 0.5))


def test_none_update_is_safe():
    c = FrameFeatureCache()
    c.update(None)  # no features yet: must not crash (Week-2 fallback)
    assert c.get() is None
    assert c.keyframes_used == 1


def test_hit_rate_stats():
    c = FrameFeatureCache(keyframe_interval=3)
    for _ in range(6):
        if c.is_keyframe():
            c.update(np.ones(2))
        c.step()
    s = c.stats()
    assert (s["frames"], s["keyframes"]) == (6, 2)
    assert s["cache_hit_rate"] == pytest.approx(2 / 3)


def test_reset():
    c = FrameFeatureCache()
    c.update(np.ones(2))
    c.step()
    c.reset()
    assert c.get() is None
    assert c.keyframes_used == 0
    assert c.is_keyframe()
