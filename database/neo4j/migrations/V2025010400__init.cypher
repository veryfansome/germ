
// Corresponds with JJ, standard adjective
CREATE CONSTRAINT     FOR (adjective:Adjective)                     REQUIRE adjective.text              IS UNIQUE;
// Corresponds with RB, standard adverb
CREATE CONSTRAINT     FOR (adverb:Adverb)                           REQUIRE adverb.text                 IS UNIQUE;
//
CREATE CONSTRAINT     FOR (document:Document)                       REQUIRE document.name               IS UNIQUE;
// Corresponds with NN, singular noun
CREATE CONSTRAINT     FOR (noun:Noun)                               REQUIRE noun.text                   IS UNIQUE;
//
CREATE CONSTRAINT     FOR (phrase:Phrase)                           REQUIRE phrase.text                 IS UNIQUE;
//
CREATE CONSTRAINT     FOR (pos:PartOfSpeech)                        REQUIRE pos.tag                     IS UNIQUE;
// Corresponds with NNP, singular proper noun
CREATE CONSTRAINT     FOR (properNoun:ProperNoun)                   REQUIRE properNoun.text             IS UNIQUE;
//
CREATE CONSTRAINT     FOR (semanticCategory:SemanticCategory)       REQUIRE semanticCategory.text       IS UNIQUE;
//
CREATE CONSTRAINT     FOR (sentence:Sentence)                       REQUIRE sentence.text               IS UNIQUE;
// Corresponds with VB, base form
CREATE CONSTRAINT     FOR (verb:Verb)                               REQUIRE verb.text                   IS UNIQUE;

//
CREATE INDEX          FOR (noun:Noun)                               ON noun.forms;

