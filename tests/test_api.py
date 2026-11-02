import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setattr("app.main.MODEL_DIR", str(model_dir))
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["domains"] == 2


def test_predict(client):
    r = client.post("/predict", json={"text": "turn the volume up a bit"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "volume_up"
    assert body["domain"] == "media"
    assert body["latency_ms"] >= 0


def test_predict_rejects_blank_text(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"texts": ["send money to john", "play jazz"]})
    assert [x["intent"] for x in r.json()] == ["transfer", "play_music"]
