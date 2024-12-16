import re
from db.neo4j import Neo4jDriver

from bot.lang.utils import findall_words
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)


def add_sentence_to_graph(connection, sentence):
    # TODO: we probably need more concepts like named entities, phrases, etc.
    words = findall_words(sentence.strip().lower())

    # Check if the sentence already exists
    existing_sentences = connection.query("MATCH (s:Sentence {text: $text}) RETURN s", {"text": sentence})

    if not existing_sentences:  # If the sentence does not exist
        # Create a node for the sentence
        connection.query("CREATE (s:Sentence {text: $text})", {"text": sentence})

        for word in words:
            # Create a node for the word if it doesn't exist
            connection.query("MERGE (w:Word {text: $text})", {"text": word})
            # Create a relationship between the word and the sentence
            connection.query("MATCH (w:Word {text: $word}), (s:Sentence {text: $text}) "
                             "CREATE (w)-[:CONTAINS]->(s)", {"word": word, "text": sentence})
    else:
        print(f"The sentence '{sentence}' already exists in the graph.")


def get_sentences_related_to_word(connection, word):
    query = """
    MATCH (w:Word {text: $word})-[:CONTAINS]->(s:Sentence)
    RETURN s.text AS sentence
    """
    return connection.query(query, {"word": word})


def get_sentences_related_to_any_of_the_words(connection, words):
    # Create a parameterized query to match any of the words
    query = """
    MATCH (w:Word)-[:CONTAINS]->(s:Sentence)
    WHERE w.text IN $words
    RETURN DISTINCT s.text AS sentence
    """
    return connection.query(query, {"words": words})


if __name__ == '__main__':
    setup_logging()

    driver = Neo4jDriver()
    #driver.delete_all_data()

    # Add sentences to the graph
    test_set = [
        "The cat sat on the mat.",
        "The dog barked at the cat.",
        "Cats and dogs are common pets.",
        "The mat is where the cat sleeps."
    ]

    for test in test_set:
        add_sentence_to_graph(driver, test)

    # Query related sentences
    related_sentences = get_sentences_related_to_word(driver, "cat")
    for relation in related_sentences:
        print(relation['sentence'])

    driver.close()
