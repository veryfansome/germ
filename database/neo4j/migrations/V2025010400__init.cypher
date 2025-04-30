
//
CREATE CONSTRAINT     FOR (chatMessage:ChatMessage)     REQUIRE chatMessage.message_id          IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatSession:ChatSession)     REQUIRE chatSession.session_id          IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatUser:ChatUser)           REQUIRE chatUser.user_id                IS UNIQUE;

