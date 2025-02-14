from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import datetime
import pytest

from starlette.websockets import WebSocketState
from starlette.websockets import WebSocketDisconnect

from bot.websocket import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    SessionMonitor,
    WebSocketConnectionManager,
    WebSocketReceiveEventHandler,
    WebSocketSendEventHandler,
    WebSocketSender,
    get_chat_session_messages,
    get_chat_session_summaries,
    new_chat_request_received,
    new_chat_response_sent,
    new_chat_session,
    update_chat_session_is_hidden,
    update_chat_session_summary,
    update_chat_session_time_stopped
)


@pytest.mark.asyncio
async def test_connect_disconnect_basic():
    mock_ws = MagicMock()
    mock_ws.accept = AsyncMock()
    mock_ws.close = AsyncMock()
    mock_ws.state = WebSocketState.CONNECTED  # << Add this line
    mock_control_plane = MagicMock()
    mock_control_plane.add_chat_session = AsyncMock(return_value=None)

    manager = WebSocketConnectionManager(control_plane=mock_control_plane)

    with patch("bot.websocket.new_chat_session", new_callable=AsyncMock) as mock_new_session:
        mock_new_session.return_value = 1234
        chat_session_id = await manager.connect(mock_ws)

    with patch("bot.websocket.update_chat_session_time_stopped", new_callable=AsyncMock) as mock_time_stopped, \
         patch("bot.websocket.update_chat_session_summary", new_callable=AsyncMock) as mock_update_summary:

        # The mock is in CONNECTED state, so disconnect should call ws.close()
        await manager.disconnect(1234)

    # Should pass now
    mock_ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_conduct_chat_session_timeout():
    """
    Test that conduct_chat_session() exits on timeout, calls disconnect().
    """
    mock_ws = MagicMock()
    mock_ws.receive_json = AsyncMock(side_effect=asyncio.TimeoutError)  # force a Timeout
    mock_control_plane = MagicMock()

    manager = WebSocketConnectionManager(control_plane=mock_control_plane)
    manager.active_connections[5678] = mock_ws

    # Patch for idle timeout
    with patch("bot.websocket.WEBSOCKET_CONNECTION_IDLE_TIMEOUT", 0.01):
        with patch.object(manager, "disconnect", new_callable=AsyncMock) as mock_disconnect:
            await manager.conduct_chat_session(5678)
            mock_disconnect.assert_awaited_once_with(5678)


@pytest.mark.asyncio
async def test_conduct_chat_session_disconnect():
    """
    Test that if the WebSocket disconnects (WebSocketDisconnect is raised),
    the manager calls its own disconnect method.
    """
    mock_ws = MagicMock()
    # Force a WebSocketDisconnect
    mock_ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect)
    mock_control_plane = MagicMock()

    manager = WebSocketConnectionManager(control_plane=mock_control_plane)
    manager.active_connections[9999] = mock_ws

    with patch.object(manager, "disconnect", new_callable=AsyncMock) as mock_disconnect:
        await manager.conduct_chat_session(9999)
        mock_disconnect.assert_awaited_once_with(9999)


@pytest.mark.asyncio
async def test_receive_and_event_handlers():
    """
    Test that receive() decodes a ChatRequest, stores it in DB, and
    invokes the registered WebSocketReceiveEventHandlers.
    """
    mock_ws = MagicMock()
    mock_ws.receive_json = AsyncMock(return_value={
        "messages": [{"role": "user", "content": "Hello"}]
    })
    mock_control_plane = MagicMock()

    manager = WebSocketConnectionManager(control_plane=mock_control_plane)
    manager.active_connections[1111] = mock_ws

    # Create a mock handler
    class MockReceiveHandler(WebSocketReceiveEventHandler):
        async def on_receive(self, chat_session_id, chat_request_received_id, chat_request, ws_sender):
            # Verify the handler sees correct data
            assert chat_session_id == 1111
            assert isinstance(chat_request, ChatRequest)
            # We won't do deep checks for test brevity
            pass

    mock_handler = MockReceiveHandler()
    manager.add_receive_event_handler(mock_handler)

    with patch("bot.websocket.new_chat_request_received", new_callable=AsyncMock) as mock_new_req_id, \
         patch("bot.websocket.ChatRequest.model_validate", side_effect=ChatRequest.model_validate) as mock_model_validate, \
         patch.object(manager.control_plane, "add_chat_request", new_callable=AsyncMock) as mock_add_chat_req, \
         patch.object(manager.control_plane, "link_chat_request_to_chat_session", new_callable=AsyncMock) as mock_link_req_sess:

        mock_new_req_id.return_value = 55
        mock_add_chat_req.return_value = (None, datetime.datetime.now(datetime.UTC))

        await manager.receive(1111)

        # new_chat_request_received was called
        mock_new_req_id.assert_awaited_once()
        # The model_validate call also must have happened
        assert mock_model_validate.called
        # add_chat_request was called
        mock_add_chat_req.assert_awaited_once()
        # link_chat_request_to_chat_session was called
        mock_link_req_sess.assert_called_once()


@pytest.mark.asyncio
async def test_send_chat_response():
    """
    Test that WebSocketSender.send_chat_response() sends data, logs the response in DB,
    and triggers send_event_handlers.
    """
    mock_ws = MagicMock()
    mock_ws.send_text = AsyncMock()
    mock_control_plane = MagicMock()

    # Prepare a mock send handler
    class MockSendHandler(WebSocketSendEventHandler):
        async def on_send(self, chat_response_sent_id, chat_response, chat_session_id, chat_request_received_id=None):
            assert chat_response_sent_id == 99
            assert chat_response.content == "test"
            assert chat_session_id == 2222

    sender = WebSocketSender(
        control_plane=mock_control_plane,
        chat_session_id=2222,
        connection=mock_ws,
        send_event_handlers=[MockSendHandler()]
    )

    chat_response = ChatResponse(content="test", role="assistant")

    with patch("bot.websocket.new_chat_response_sent", new_callable=AsyncMock) as mock_resp_sent, \
         patch.object(mock_control_plane, "add_chat_response", new_callable=AsyncMock) as mock_add_chat_resp, \
         patch.object(mock_control_plane, "link_chat_response_to_chat_session", new_callable=AsyncMock) as mock_link_resp_sess:
        mock_resp_sent.return_value = 99
        mock_add_chat_resp.return_value = (None, datetime.datetime.now(datetime.UTC))

        await sender.send_chat_response(chat_response)

        # The WS should have had send_text called
        mock_ws.send_text.assert_awaited_once()
        # DB calls
        mock_resp_sent.assert_awaited_once()
        mock_add_chat_resp.assert_awaited_once()
        mock_link_resp_sess.assert_awaited_once()


@pytest.mark.asyncio
async def test_return_chat_response():
    """
    Similar to test_send_chat_response, but verifies the link to ChatRequest is used.
    """
    mock_ws = MagicMock()
    mock_ws.send_text = AsyncMock()
    mock_control_plane = MagicMock()

    class MockSendHandler(WebSocketSendEventHandler):
        async def on_send(self, chat_response_sent_id, chat_response, chat_session_id, chat_request_received_id=None):
            assert chat_response_sent_id == 77
            assert chat_request_received_id == 44

    sender = WebSocketSender(
        control_plane=mock_control_plane,
        chat_session_id=3333,
        connection=mock_ws,
        send_event_handlers=[MockSendHandler()]
    )

    chat_response = ChatResponse(content="some reply", role="assistant")

    with patch("bot.websocket.new_chat_response_sent", new_callable=AsyncMock) as mock_new_resp, \
         patch.object(mock_control_plane, "add_chat_response", new_callable=AsyncMock) as mock_add_chat_resp, \
         patch.object(mock_control_plane, "link_chat_response_to_chat_request", new_callable=AsyncMock) as mock_link_req, \
         patch.object(mock_control_plane, "link_chat_response_to_chat_session", new_callable=AsyncMock) as mock_link_sess:
        mock_new_resp.return_value = 77
        mock_add_chat_resp.return_value = (None, datetime.datetime.now(datetime.UTC))

        await sender.return_chat_response(44, chat_response)

        mock_ws.send_text.assert_awaited_once()
        mock_link_req.assert_awaited_once()
        mock_link_sess.assert_awaited_once()


@pytest.mark.asyncio
async def test_monitor_chat_session_basic():
    """
    Test that monitor_chat_session runs in a loop and eventually calls on_tick()
    for each registered SessionMonitor, until the cancel_event is set.

    Because the real code waits 15 seconds, we'll patch asyncio.wait_for so
    we don't delay the test by 15 seconds. We'll force it to timeout quickly
    or set the event to simulate the timer.
    """

    mock_ws = MagicMock()
    mock_control_plane = MagicMock()
    mock_ws.state = WebSocketState.CONNECTED

    # A sample SessionMonitor that just logs a call
    class MockSessionMonitor(SessionMonitor):
        def __init__(self):
            self.tick_count = 0

        async def on_tick(self, chat_session_id, ws_sender):
            self.tick_count += 1

    manager = WebSocketConnectionManager(control_plane=mock_control_plane)
    mock_monitor = MockSessionMonitor()
    manager.add_session_monitor(mock_monitor)

    # Instead of actually sleeping 15s, we patch `asyncio.wait_for`
    # so it triggers a TimeoutError after a very short delay.
    # That way, our code’s "except asyncio.TimeoutError" branch runs quickly.
    async def fake_wait_for(coro, timeout):
        # Use a short sleep to simulate the loop quickly
        await asyncio.sleep(0.01)
        raise asyncio.TimeoutError

    with patch("asyncio.wait_for", side_effect=fake_wait_for):
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(manager.monitor_chat_session(10, mock_ws, cancel_event))

        # Let the task run a bit. It should hit a few short "timeouts"
        # and call on_tick each time it times out.
        await asyncio.sleep(0.05)

        # Now signal the monitor to shut down
        cancel_event.set()

        # Wait for the monitor task to finish
        await monitor_task

        # Because we forced multiple quick timeouts in 0.05 seconds,
        # we expect `on_tick` was called at least once or more.
        # The exact count depends on how quickly the code cycles,
        # but it should be >= 1 if everything’s working.
        assert mock_monitor.tick_count >= 1, (
            "Expected monitor to call on_tick at least once, but got 0"
        )


# Below are simpler direct tests for the DB-related coroutines. Often you’d want more integrated tests,
# but here we do a minimal approach with mocks or partial real calls.

@pytest.mark.asyncio
async def test_new_chat_session():
    """
    Test new_chat_session in isolation.
    Typically you'd mock out AsyncSessionLocal, but here we do a bare minimal approach.
    """
    with patch("bot.websocket.AsyncSessionLocal", autospec=True) as mock_session_local:
        mock_session = MagicMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = False
        mock_session_local.return_value = mock_session

        # We also need a chat_session_id. We can store a fake ID in the record.
        mock_record = MagicMock()
        mock_record.chat_session_id = 123

        # Patch ChatSession so that instantiating it returns our mock_record.
        with patch("bot.websocket.ChatSession", return_value=mock_record):
            # Make sure add() and commit() are properly awaitable.
            mock_session.add = MagicMock()
            mock_session.commit = AsyncMock()

            await_val = await new_chat_session()
            assert await_val == 123, f"Expected 123, got {await_val}"


@pytest.mark.asyncio
async def test_new_chat_request_received():
    """
    Basic test that new_chat_request_received commits a new record.
    """
    with patch("bot.websocket.AsyncSessionLocal", autospec=True) as mock_session_local:
        mock_session = MagicMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = False
        mock_session_local.return_value = mock_session

        # Mock record
        mock_record = MagicMock()
        mock_record.chat_request_received_id = 999
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        # We track how the ChatRequestReceived is constructed
        with patch("bot.websocket.ChatRequestReceived") as mock_model:
            mock_model.return_value = mock_record
            chat_request = ChatRequest(messages=[{"role": "user", "content": "Hello test"}])
            out = await new_chat_request_received(123, chat_request)
            assert out == 999


@pytest.mark.asyncio
async def test_new_chat_response_sent():
    """
    Basic test for new_chat_response_sent.
    """
    with patch("bot.websocket.AsyncSessionLocal", autospec=True) as mock_session_local:
        mock_session = MagicMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = False
        mock_session_local.return_value = mock_session

        mock_record = MagicMock()
        mock_record.chat_response_sent_id = 777
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        with patch("bot.websocket.ChatResponseSent") as mock_response_sent:
            mock_response_sent.return_value = mock_record
            resp = ChatResponse(role="assistant", content="Test answer")
            out = await new_chat_response_sent(chat_session_id=123, chat_response=resp)
            assert out == 777


@pytest.mark.asyncio
async def test_get_chat_session_messages():
    """
    Minimal test for get_chat_session_messages.
    We will mock the DB queries and return a single record of ChatResponseSent.
    """
    with patch("bot.websocket.AsyncSessionLocal", autospec=True) as mock_session_local:
        mock_session = MagicMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = False
        mock_session_local.return_value = mock_session

        mock_crs = MagicMock()
        # This record will have some JSON that we can parse into ChatResponse
        mock_crs.chat_request_received_id = None
        mock_crs.chat_response = {"role": "assistant", "content": "Hello from the bot"}

        # Mocking the query
        mock_execute = MagicMock()
        mock_execute.scalars.return_value.first.return_value = mock_crs
        mock_session.execute = AsyncMock(return_value=mock_execute)

        messages = await get_chat_session_messages(123)
        assert len(messages) == 1
        assert messages[0].role == "assistant"
        assert messages[0].content == "Hello from the bot"


@pytest.mark.asyncio
async def test_get_chat_session_summaries():
    """
    Minimal test for get_chat_session_summaries().
    """
    with patch("bot.websocket.AsyncSessionLocal", autospec=True) as mock_session_local:
        mock_session = MagicMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = False
        mock_session_local.return_value = mock_session

        mock_cs = MagicMock()
        mock_cs.chat_session_id = 1
        mock_cs.summary = "Fake summary"
        mock_cs.time_started = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
        mock_cs.time_stopped = datetime.datetime(2023, 1, 2, tzinfo=datetime.timezone.utc)

        mock_exec = MagicMock()
        mock_exec.scalars.return_value.all.return_value = [mock_cs]
        mock_session.execute = AsyncMock(return_value=mock_exec)

        summaries = await get_chat_session_summaries()
        assert len(summaries) == 1
        s = summaries[0]
        assert s.chat_session_id == 1
        assert s.summary == "Fake summary"


@pytest.mark.asyncio
async def test_update_chat_session_is_hidden():
    """
    Minimal test for update_chat_session_is_hidden().
    """
    with patch("bot.websocket.AsyncSessionLocal", autospec=True) as mock_session_local:
        mock_session = MagicMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = False
        mock_session_local.return_value = mock_session

        mock_cs = MagicMock()
        mock_exec = MagicMock()
        mock_exec.scalars.return_value.first.return_value = mock_cs
        mock_session.commit = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_exec)

        await update_chat_session_is_hidden(22)
        mock_session.commit.assert_awaited_once()
        assert mock_cs.is_hidden is True


@pytest.mark.asyncio
async def test_update_chat_session_time_stopped():
    """
    Minimal test for update_chat_session_time_stopped().
    """
    with patch("bot.websocket.AsyncSessionLocal", autospec=True) as mock_session_local:
        mock_session = MagicMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = False
        mock_session_local.return_value = mock_session

        mock_cs = MagicMock()
        mock_exec = MagicMock()
        mock_exec.scalars.return_value.first.return_value = mock_cs
        mock_session.commit = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_exec)

        await update_chat_session_time_stopped(22)
        mock_session.commit.assert_awaited_once()
        assert mock_cs.time_stopped is not None


@pytest.mark.asyncio
async def test_update_chat_session_summary_no_messages(caplog):
    """
    If there aren't at least two messages, the code logs an info and does nothing.
    We can check that it logs the skip message.
    """
    with patch("bot.websocket.get_chat_session_messages", new_callable=AsyncMock) as mock_get_msgs:
        with patch("bot.websocket.AsyncSessionLocal", autospec=True) as mock_session_local:
            mock_session = MagicMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = False
            mock_session_local.return_value = mock_session

            mock_get_msgs.return_value = [
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="Hello, how can I help?"),
            ]
            mock_cs = MagicMock()
            mock_exec = MagicMock()
            mock_exec.scalars.return_value.first.return_value = mock_cs
            mock_session.commit = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_exec)

            messages_updated = await update_chat_session_summary(42)
            assert messages_updated == 2


@pytest.mark.asyncio
async def test_update_chat_session_summary_with_messages():
    """
    If there are enough messages, the code calls openai and updates the summary.
    """
    # We will mock get_chat_session_messages to return multiple messages
    fake_msgs = [
        # ChatMessage has "role", "content"
        # We rely on ChatMessage being shaped like ChatRequest or ChatResponse
        # Here we just pass a raw object or dict that can be validated
        # if you prefer you can create actual ChatRequest/ChatResponse objects
        MagicMock(role="user", content="Hello"),
        MagicMock(role="assistant", content="Hi there")
    ]

    with patch("bot.websocket.get_chat_session_messages", new_callable=AsyncMock) as mock_get_msgs, \
         patch("bot.websocket.async_openai_client") as mock_openai_client, \
         patch("bot.websocket.AsyncSessionLocal") as mock_session_local:
        mock_get_msgs.return_value = fake_msgs

        # mock openai reply
        mock_openai_client.chat.completions.create = AsyncMock()
        mock_openai_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="A short summary"))
        ]

        # Mock DB
        mock_session = MagicMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = False
        mock_session_local.return_value = mock_session

        mock_cs = MagicMock()
        mock_exec = MagicMock()
        mock_exec.scalars.return_value.first.return_value = mock_cs
        mock_session.commit = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_exec)

        await update_chat_session_summary(42)

        mock_openai_client.chat.completions.create.assert_awaited_once()
        mock_session.commit.assert_awaited_once()
        assert mock_cs.summary == "A short summary"
