
//
CREATE CONSTRAINT                       FOR (m:ChatMessage)         REQUIRE (m.conversation_id, m.dt_created)   IS UNIQUE;
//
CREATE CONSTRAINT                       FOR (u:ChatUser)            REQUIRE u.user_id                           IS UNIQUE;
//
CREATE CONSTRAINT                       FOR (c:Conversation)        REQUIRE c.conversation_id                   IS UNIQUE;
//
CREATE CONSTRAINT                       FOR (s:Synset)              REQUIRE (s.lemma, s.pos, s.sense)           IS UNIQUE;
CREATE INDEX synsetPosIndex             FOR (s:Synset)              ON (s.pos);
CREATE INDEX synsetLemmaIndex           FOR (s:Synset)              ON (s.lemma);
CREATE INDEX synsetLemmasIndex          FOR (s:Synset)              ON (s.lemmas);
//
CREATE CONSTRAINT                       FOR (d:SynsetDefinition)    REQUIRE d.text                              IS UNIQUE;
CREATE FULLTEXT INDEX synsetDefText     FOR (d:SynsetDefinition)    ON EACH [d.text];
