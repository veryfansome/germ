from fastapi.testclient import TestClient
from bot.main import bot

client = TestClient(bot)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "OK"


def test_chat_page_load():
    response = client.get("/")
    assert response.status_code == 200
