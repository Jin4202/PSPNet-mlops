import io

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from tests.conftest import FakeSegModel


@pytest.fixture
def client(monkeypatch, cfg):
    monkeypatch.setattr(main_module, "load_config", lambda: cfg)
    monkeypatch.setattr(
        main_module, "load_model", lambda cfg, device: (FakeSegModel(cfg["model"]["num_classes"]), "checkpoint:fake.pth")
    )
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_source"] == "checkpoint:fake.pth"


def test_predict_valid_image_returns_mask(client, sample_image):
    buf = io.BytesIO()
    sample_image.save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/predict", files={"file": ("test.png", buf, "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["width"] == sample_image.width
    assert body["height"] == sample_image.height
    assert isinstance(body["inference_time_ms"], float)
    assert isinstance(body["mask_png_base64"], str) and len(body["mask_png_base64"]) > 0


def test_predict_rejects_non_image_content_type(client):
    response = client.post(
        "/predict", files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    )

    assert response.status_code == 400


def test_predict_rejects_corrupt_image_bytes(client):
    response = client.post(
        "/predict", files={"file": ("test.png", io.BytesIO(b"garbage-not-a-real-png"), "image/png")}
    )

    assert response.status_code == 400


def test_health_when_no_request_yet_uses_lifespan_state(client):
    # lifespan has already populated state by the time the client is usable
    response = client.get("/health")
    assert response.json()["device"] == "cpu"
