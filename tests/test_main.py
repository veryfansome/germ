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
    bookmark_post_response = client.post("/chat/bookmark", json={
        'message_received_id': chat_data["message_received_id"],
        'message_replied_id': chat_data["message_replied_id"],
        'message_replied_content': chat_data["response"]["choices"][0]['message']['content'],
    })
    assert bookmark_post_response.status_code == 200
    bookmark_post_data = bookmark_post_response.json()
    assert "id" in bookmark_post_data

    # Retrieve bookmarked messages
    bookmark_get_response = client.get(f"/chat/bookmark/{bookmark_post_data['id']}")
    assert bookmark_get_response.status_code == 200
    bookmark_get_data = bookmark_get_response.json()
    assert "id" in bookmark_get_data
    assert "message_summary" in bookmark_get_data
    assert "message_received" in bookmark_get_data
    assert "id" in bookmark_get_data["message_received"]
    assert bookmark_get_data["message_received"]["id"] == chat_data["message_received_id"]
    assert "message_replied" in bookmark_get_data
    assert "id" in bookmark_get_data["message_replied"]
    assert bookmark_get_data["message_replied"]["id"] == chat_data["message_replied_id"]


def test_postgres_page_load():
    response = client.get("/postgres.html")
    assert response.status_code == 200


def test_postgres_query():
    response = client.post("/postgres/germ/query", json={"sql": "\\dt"})
    assert response.status_code == 200


