//
CREATE CONSTRAINT     FOR (document:Document)             REQUIRE document.name         IS UNIQUE;
//
CREATE CONSTRAINT     FOR (noun:Noun)                     REQUIRE noun.text             IS UNIQUE;
//
CREATE CONSTRAINT     FOR (phrase:Phrase)                 REQUIRE phrase.text           IS UNIQUE;
//
CREATE CONSTRAINT     FOR (sentence:Sentence)             REQUIRE sentence.text         IS UNIQUE;

//
CREATE INDEX          FOR (noun:Noun)                     ON noun.plural_forms;

