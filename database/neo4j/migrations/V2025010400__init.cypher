
//
CREATE CONSTRAINT     FOR (chatRequest:ChatRequest)     REQUIRE chatRequest.chat_request_received_id        IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatResponse:ChatResponse)   REQUIRE chatResponse.chat_response_sent_id          IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatSession:ChatSession)     REQUIRE chatSession.chat_session_id                 IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatUser:ChatUser)           REQUIRE chatUser.user_id                            IS UNIQUE;

