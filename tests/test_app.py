import uuid
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app

client = TestClient(app)


def test_get_index_returns_html():
    # templates/index.html must exist for this test
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_count_rejects_non_video():
    response = client.post(
        "/count",
        files={"file": ("photo.jpg", BytesIO(b"fake"), "image/jpeg")},
    )
    assert response.status_code == 400
    assert "video" in response.json()["detail"].lower()


def test_count_returns_count_on_success():
    with patch("app.web_main.run", return_value=42):
        response = client.post(
            "/count",
            files={"file": ("clip.mp4", BytesIO(b"fake video bytes"), "video/mp4")},
        )
    assert response.status_code == 200
    assert response.json() == {"count": 42}


def test_count_returns_zero_and_error_on_failure():
    with patch("app.web_main.run", side_effect=FileNotFoundError("model missing")):
        response = client.post(
            "/count",
            files={"file": ("clip.mp4", BytesIO(b"fake video bytes"), "video/mp4")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert "model missing" in data["error"]


def test_count_returns_video_id_on_success():
    def mock_run_with_output(video_path, output_path=None):
        if output_path:
            Path(output_path).write_bytes(b"fake mp4")
        return 7

    with patch("app.web_main.run", side_effect=mock_run_with_output):
        response = client.post(
            "/count",
            files={"file": ("clip.mp4", BytesIO(b"fake video bytes"), "video/mp4")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 7
    assert "video_id" in data
    try:
        uuid.UUID(data["video_id"])
    except ValueError:
        pytest.fail("video_id is not a valid UUID")


def test_count_no_video_id_on_error():
    with patch("app.web_main.run", side_effect=RuntimeError("boom")):
        response = client.post(
            "/count",
            files={"file": ("clip.mp4", BytesIO(b"fake video bytes"), "video/mp4")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert "video_id" not in data


def test_get_video_serves_and_deletes_file():
    video_id = str(uuid.uuid4())
    video_file = app_module._temp_dir / f"{video_id}_output.mp4"
    video_file.write_bytes(b"fake mp4 content")

    response = client.get(f"/video/{video_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert video_file.exists()
    video_file.unlink(missing_ok=True)


def test_get_video_returns_404_for_unknown_id():
    response = client.get(f"/video/{uuid.uuid4()}")
    assert response.status_code == 404


def test_get_video_returns_400_for_invalid_id():
    response = client.get("/video/not-a-uuid")
    assert response.status_code == 400


def test_delete_video_removes_file():
    video_id = str(uuid.uuid4())
    video_file = app_module._temp_dir / f"{video_id}_output.mp4"
    video_file.write_bytes(b"fake mp4 content")

    response = client.delete(f"/video/{video_id}")
    assert response.status_code == 204
    assert not video_file.exists()


def test_delete_video_returns_404_for_unknown_id():
    response = client.delete(f"/video/{uuid.uuid4()}")
    assert response.status_code == 404


def test_delete_video_returns_400_for_invalid_id():
    response = client.delete("/video/not-a-uuid")
    assert response.status_code == 400
