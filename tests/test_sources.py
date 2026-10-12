"""Video sources: working rate, working size, image folders and video files."""
import cv2
import numpy as np
import pytest

from sideline.sources import SIZE, Source


def _write_images(d, n, size=(640, 360)):
    d.mkdir(parents=True)
    for i in range(n):
        img = np.full((size[1], size[0], 3), i * 10 % 255, np.uint8)
        cv2.imwrite(str(d / f"{i:06d}.jpg"), img)


def test_image_folder_frames_are_resized_and_numbered(tmp_path):
    _write_images(tmp_path / "clip" / "img1", 5)
    src = Source(tmp_path / "clip")
    got = list(src.frames())
    assert [i for i, _ in got] == [0, 1, 2, 3, 4]
    assert got[0][1].shape == (SIZE[1], SIZE[0], 3)


def test_image_folder_loop_keeps_indices_increasing(tmp_path):
    _write_images(tmp_path / "img1", 3)
    src = Source(tmp_path / "img1", loop=True, max_frames=8)
    idx = [i for i, _ in src.frames()]
    assert len(idx) == 8 and idx == sorted(idx) and len(set(idx)) == 8


def test_fast_video_is_resampled_to_25_fps(tmp_path):
    path = str(tmp_path / "v.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 50.0, (320, 180))
    if not w.isOpened():
        pytest.skip("no MJPG writer")
    for i in range(100):                                    # 2 s at 50 FPS
        w.write(np.full((180, 320, 3), i % 255, np.uint8))
    w.release()
    idx = [i for i, _ in Source(path).frames()]
    assert 48 <= len(idx) <= 51
    assert idx[-1] - idx[0] == len(idx) - 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Source(tmp_path / "nope.mp4")
