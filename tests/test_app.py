from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient

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
