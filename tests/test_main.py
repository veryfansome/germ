from fastapi.testclient import TestClient
from bot.main import bot

client = TestClient(bot)


def test_main():
    response = client.get("/")
    assert response.status_code == 200


def test_main_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "OK"


def test_chat_basic():
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "Hello?"}]})
    assert response.status_code == 200
    assert "message_received_id" in response.json()
    assert "message_sent_id" in response.json()
    assert "response" in response.json()
