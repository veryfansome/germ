
//
CREATE CONSTRAINT                     FOR (m:ChatMessage)     REQUIRE (m.conversation_id, m.dt_created)       IS UNIQUE;
//
CREATE CONSTRAINT                     FOR (u:ChatUser)        REQUIRE u.user_id                               IS UNIQUE;
//
CREATE CONSTRAINT                     FOR (c:Conversation)    REQUIRE c.conversation_id                       IS UNIQUE;
//
CREATE CONSTRAINT                     FOR (d:Definition)      REQUIRE d.definition                            IS UNIQUE;
CREATE FULLTEXT INDEX definitionText  FOR (d:Definition)      ON EACH [d.definition];
//
CREATE CONSTRAINT                     FOR (s:Synset)          REQUIRE (s.lemma, s.pos, s.sense)               IS UNIQUE;
CREATE INDEX posIndex                 FOR (s:Synset)          ON (s.pos);
CREATE INDEX lemmaIndex               FOR (s:Synset)          ON (s.lemma);
