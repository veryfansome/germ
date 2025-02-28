
//
CREATE CONSTRAINT     FOR (chatRequest:ChatRequest)            REQUIRE chatRequest.chat_request_received_id  IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatResponse:ChatResponse)          REQUIRE chatResponse.chat_response_sent_id    IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatSession:ChatSession)            REQUIRE chatSession.chat_session_id           IS UNIQUE;
//
CREATE CONSTRAINT     FOR (codeBlock:CodeBlock)                REQUIRE codeBlock.code_block_id               IS UNIQUE;
//
CREATE CONSTRAINT     FOR (domainName:DomainName)              REQUIRE domainName.name                       IS UNIQUE;
// Corresponds with NN, singular noun
CREATE CONSTRAINT     FOR (noun:Noun)                          REQUIRE noun.text                             IS UNIQUE;
//
CREATE CONSTRAINT     FOR (paragraph:Paragraph)                REQUIRE paragraph.paragraph_id                IS UNIQUE;
// Corresponds with NNP, singular proper noun
CREATE CONSTRAINT     FOR (properNoun:ProperNoun)              REQUIRE properNoun.text                       IS UNIQUE;
//
CREATE CONSTRAINT     FOR (sentence:Sentence)                  REQUIRE sentence.text                         IS UNIQUE;

//
CREATE INDEX          FOR (noun:Noun)                          ON noun.forms;

