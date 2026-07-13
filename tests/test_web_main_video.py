from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import supervision as sv


def _make_video(path: str, frames: int = 30, fps: float = 30.0, w: int = 64, h: int = 64):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for _ in range(frames):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()


@pytest.fixture
def video_file(tmp_path):
    p = str(tmp_path / "input.mp4")
    _make_video(p)
    return p


def _patch_ml(tmp_path):
    fake_pt = tmp_path / "fake.pt"
    fake_pt.touch()
    mock_tracker = MagicMock()
    mock_tracker.count.return_value = 5
    return [
        patch("web_main._model_path", new=fake_pt),
        patch("web_main._seg_model_path", new=tmp_path / "no_seg.pt"),
        patch("web_main.get_sahi", return_value=MagicMock()),
        patch("web_main.apply_sahi", return_value=sv.Detections.empty()),
        patch("web_main.apply_clahe", side_effect=lambda *a, **k: a[0] if a else k["frame"]),
        patch("web_main.GlobalTracker", return_value=mock_tracker),
    ]


def test_run_creates_output_video(tmp_path, video_file):
    output_path = str(tmp_path / "output.mp4")
    patches = _patch_ml(tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        from web_main import run
        count = run(video_file, output_path=output_path)
    assert count == 5
    assert Path(output_path).exists()


def test_run_without_output_path_returns_count(tmp_path, video_file):
    patches = _patch_ml(tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        from web_main import run
        count = run(video_file)
    assert count == 5


def test_run_reports_progress(tmp_path, video_file):
    patches = _patch_ml(tmp_path)
    reported = []
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        from web_main import run
        count = run(video_file, progress=reported.append)
    assert count == 5
    assert reported, "progress callback was never called"
    assert reported == sorted(reported)
    assert reported[-1] == 100
