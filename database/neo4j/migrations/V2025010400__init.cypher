
// Corresponds with JJ, standard adjective
CREATE CONSTRAINT     FOR (adjective:Adjective)                REQUIRE adjective.text                        IS UNIQUE;
// Corresponds with RB, standard adverb
CREATE CONSTRAINT     FOR (adverb:Adverb)                      REQUIRE adverb.text                           IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatRequest:ChatRequest)            REQUIRE chatRequest.chat_request_received_id  IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatResponse:ChatResponse)          REQUIRE chatResponse.chat_response_sent_id    IS UNIQUE;
//
CREATE CONSTRAINT     FOR (chatSession:ChatSession)            REQUIRE chatSession.chat_session_id           IS UNIQUE;
//
CREATE CONSTRAINT     FOR (codeBlock:CodeBlock)                REQUIRE codeBlock.code_block_id               IS UNIQUE;
//
CREATE CONSTRAINT     FOR (codeSnippet:CodeSnippet)            REQUIRE codeSnippet.text                      IS UNIQUE;
//
CREATE CONSTRAINT     FOR (document:Document)                  REQUIRE document.name                         IS UNIQUE;
//
CREATE CONSTRAINT     FOR (header:Header)                      REQUIRE header.text                           IS UNIQUE;
// Corresponds with NN, singular noun
CREATE CONSTRAINT     FOR (noun:Noun)                          REQUIRE noun.text                             IS UNIQUE;
//
CREATE CONSTRAINT     FOR (paragraph:Paragraph)                REQUIRE paragraph.paragraph_id                IS UNIQUE;
//
CREATE CONSTRAINT     FOR (partOfSpeech:PartOfSpeech)          REQUIRE partOfSpeech.tag                      IS UNIQUE;
//
CREATE CONSTRAINT     FOR (phrase:Phrase)                      REQUIRE phrase.text                           IS UNIQUE;
// Corresponds with NNP, singular proper noun
CREATE CONSTRAINT     FOR (properNoun:ProperNoun)              REQUIRE properNoun.text                       IS UNIQUE;
//
CREATE CONSTRAINT     FOR (semanticCategory:SemanticCategory)  REQUIRE semanticCategory.text                 IS UNIQUE;
//
CREATE CONSTRAINT     FOR (sentence:Sentence)                  REQUIRE sentence.text                         IS UNIQUE;
// Corresponds with VB, base form
CREATE CONSTRAINT     FOR (verb:Verb)                          REQUIRE verb.text                             IS UNIQUE;

//
CREATE INDEX          FOR (noun:Noun)                          ON noun.forms;

