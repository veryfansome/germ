
//
CREATE CONSTRAINT     FOR (adj:Adjective)               REQUIRE adj.text                                    IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatRequest:ChatRequest)     REQUIRE chatRequest.chat_request_received_id        IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatResponse:ChatResponse)   REQUIRE chatResponse.chat_response_sent_id          IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatSession:ChatSession)     REQUIRE chatSession.chat_session_id                 IS UNIQUE;
//
CREATE CONSTRAINT     FOR (code:CodeBlock)              REQUIRE (code.text_block_id, code.time_occurred)    IS UNIQUE;
//
CREATE CONSTRAINT     FOR (domainName:DomainName)       REQUIRE domainName.name                             IS UNIQUE;
//
CREATE CONSTRAINT     FOR (noun:Noun)                   REQUIRE (noun.text, noun.text_block_id)               IS UNIQUE;
//
CREATE CONSTRAINT     FOR (para:Paragraph)              REQUIRE (para.text_block_id, para.time_occurred)     IS UNIQUE;
//
CREATE CONSTRAINT     FOR (sentence:Sentence)           REQUIRE sentence.text                               IS UNIQUE;

//
CREATE INDEX          FOR (adj:Adjective)               ON (adj.forms);
//
CREATE INDEX          FOR (noun:Noun)                   ON (noun.forms, noun.text_block_id);

