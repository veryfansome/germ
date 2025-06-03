import asyncio
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

from germ.bot.db.neo4j import AsyncNeo4jDriver


async def create_definition_node(tx, definition):
    query = """
    MERGE (d:Definition {definition: $definition})
    """
    await tx.run(query, definition=definition)


async def create_word_node(tx, word, pos):
    query = """
    MERGE (w:Word {word: $word, pos: $pos})
    """
    await tx.run(query, word=word, pos=pos)


async def create_means_relationship(tx, word, pos, definition):
    query = """
    MATCH (w:Word {word: $word, pos: $pos})
    MATCH (d:Definition {definition: $definition})
    MERGE (w)-[:MEANS]->(d)
    """
    await tx.run(query, word=word, pos=pos, definition=definition)


async def create_opposite_of_relationship(tx, word1, pos1, word2, pos2):
    query = """
    MATCH (w1:Word {word: $word1, pos: $pos1})
    MATCH (w2:Word {word: $word2, pos: $pos2})
    MERGE (w1)<-[:ANTONYM]->(w2)
    """
    await tx.run(query, word1=word1, pos1=pos1, word2=word2, pos2=pos2)


async def create_topic_domain_relationship(tx, word, pos, definition):
    query = """
    MATCH (w:Word {word: $word, pos: $pos})
    MATCH (d:Definition {definition: $definition})
    MERGE (w)-[:TOPIC_DOMAIN]->(d)
    """
    await tx.run(query, word=word, pos=pos, definition=definition)


async def create_usage_domain_relationship(tx, word, pos, definition):
    query = """
    MATCH (w:Word {word: $word, pos: $pos})
    MATCH (d:Definition {definition: $definition})
    MERGE (w)-[:USAGE_DOMAIN]->(d)
    """
    await tx.run(query, word=word, pos=pos, definition=definition)


async def main():
    driver = AsyncNeo4jDriver()
    all_synsets = [s for s in wn.all_synsets()]
    all_synsets_len = len(all_synsets)
    synset_idx = 0
    while synset_idx < all_synsets_len:
        async with driver.driver.session() as session:
            for _ in range(250):
                if synset_idx >= all_synsets_len:
                    break

                synset = all_synsets[synset_idx]
                definition = synset.definition()
                pos = synset.pos()

                for topic in synset.topic_domains():
                    topic_name = topic.name().split('.')[0]
                    topic_pos = topic.pos()
                    await session.execute_write(create_word_node, topic_name, topic_pos)
                    await session.execute_write(
                        create_topic_domain_relationship,
                        topic_name,
                        topic_pos,
                        definition,
                    )

                for usage in synset.usage_domains():
                    usage_name = usage.name().split('.')[0]
                    print(f"{synset.name()} {usage_name}")
                    usage_pos = usage.pos()
                    await session.execute_write(create_word_node, usage_name, usage_pos)
                    await session.execute_write(
                        create_usage_domain_relationship,
                        usage_name,
                        usage_pos,
                        definition,
                    )

                await session.execute_write(create_definition_node, synset.definition())
                for lemma in synset.lemmas():
                    lemma_name = lemma.name().replace("_", " ")
                    antonyms = lemma.antonyms()
                    await session.execute_write(create_word_node, lemma_name, pos)
                    await session.execute_write(
                        create_means_relationship,
                        lemma_name,
                        pos,
                        definition,
                    )
                    for antonym in antonyms:
                        antonym_synset = antonym.synset()
                        antonym_definition = antonym_synset.definition()
                        antonym_name = antonym.name().replace("_", " ")
                        antonym_pos = antonym_synset.pos()
                        await session.execute_write(create_definition_node, antonym.synset().definition())
                        await session.execute_write(
                            create_means_relationship,
                            antonym_name,
                            antonym_pos,
                            antonym_definition,
                        )
                        await session.execute_write(
                            create_opposite_of_relationship,
                            lemma_name,
                            pos,
                            antonym_name,
                            antonym_pos,
                        )
                synset_idx += 1
            print(f"Processed {synset_idx + 1} synsets")

    await driver.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
