
//
CREATE CONSTRAINT   FOR (m:ChatMessage)     REQUIRE (m.conversation_id, m.dt_created)       IS UNIQUE;
//
CREATE CONSTRAINT   FOR (u:ChatUser)        REQUIRE u.user_id                               IS UNIQUE;
//
CREATE CONSTRAINT   FOR (c:Conversation)    REQUIRE c.conversation_id                       IS UNIQUE;

