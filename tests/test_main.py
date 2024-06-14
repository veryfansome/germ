from fastapi.testclient import TestClient
from bot.main import bot

client = TestClient(bot)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "OK"


def test_chat_pageload():
    response = client.get("/")
    assert response.status_code == 200


def test_fetching_bookmarks():
    response = client.get("/chat/bookmarks")
    assert response.status_code == 200


def test_basic_interactions():
    chat_response = client.post("/chat", json={"messages": [
        {"role": "user", "content": "Hello?"},
    ]})
    assert chat_response.status_code == 200
    chat_data = chat_response.json()
    assert "message_received_id" in chat_data
    assert "message_replied_id" in chat_data
    assert "response" in chat_data

    # Bookmark it
    bookmark_response = client.post("/chat/bookmark", json={
        'message_received_id': chat_data["message_received_id"],
        'message_replied_id': chat_data["message_replied_id"],
        'message_replied_content': chat_data["response"]["choices"][0]['message']['content'],
    })
    assert bookmark_response.status_code == 200
    bookmark_data = bookmark_response.json()
    assert "bookmark_id" in bookmark_data

    # Thumbs down
    thumbs_down_response = client.post("/chat/thumbs-down", json={
        'message_received_id': chat_data["message_received_id"],
        'message_replied_id': chat_data["message_replied_id"],
    })
    assert thumbs_down_response.status_code == 200
    thumbs_down_data = thumbs_down_response.json()
    assert "thumbs_down_id" in thumbs_down_data

    # TODO: test fetching messages
    pass


def test_postgres_pageload():
    response = client.get("/postgres.html")
    assert response.status_code == 200


def test_postgres_query():
    response = client.post("/postgres/germ/query", json={"sql": "\\dt"})
    assert response.status_code == 200


