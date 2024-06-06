from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}


def test_chat():
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "Hello"}]})
    assert response.status_code == 200
    assert "response" in response.json()
