

CREATE CONSTRAINT                       FOR (m:ChatMessage)         REQUIRE (m.conversation_id, m.dt_created)   IS UNIQUE;

CREATE CONSTRAINT                       FOR (u:ChatUser)            REQUIRE u.user_id                           IS UNIQUE;

CREATE CONSTRAINT                       FOR (c:Conversation)        REQUIRE c.conversation_id                   IS UNIQUE;

// Intent

CREATE CONSTRAINT                       FOR (s:Intent)              REQUIRE (s.text)                            IS UNIQUE;
CREATE CONSTRAINT                       FOR (s:IntentCategory)      REQUIRE (s.text)                            IS UNIQUE;

// SearchQuery nodes

CREATE CONSTRAINT                       FOR (s:SearchQuery)         REQUIRE (s.text)                            IS UNIQUE;
CREATE FULLTEXT INDEX searchQueryText   FOR (s:SearchQuery)         ON EACH [s.text];
CREATE VECTOR INDEX searchQueryVector   FOR (s:SearchQuery)         ON (s.embedding)
OPTIONS {
    indexConfig: {
          `vector.dimensions`:          1024        // Based on dimension of embedding model
        , `vector.similarity_function`: 'cosine'
    }
};

// Summary nodes

CREATE CONSTRAINT                       FOR (s:Summary)             REQUIRE s.text                              IS UNIQUE;
CREATE FULLTEXT INDEX summaryText       FOR (s:Summary)             ON EACH [s.text];
CREATE VECTOR INDEX summaryVector       FOR (s:Summary)             ON (s.embedding)
OPTIONS {
    indexConfig: {
          `vector.dimensions`:          1024        // Based on dimension of embedding model
        , `vector.similarity_function`: 'cosine'
    }
};

// Website nodes

CREATE CONSTRAINT                       FOR (w:Website)             REQUIRE w.domain_name                       IS UNIQUE;
