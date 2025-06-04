import asyncio
import nltk

from germ.database.neo4j import new_async_driver


async def create_antonym_relationship(tx, word1, pos1, word2, pos2):
    query = """
    MATCH (w1:Word {word: $word1, pos: $pos1})
    MATCH (w2:Word {word: $word2, pos: $pos2})
    MERGE (w1)-[:ANTONYM]->(w2)
    """
    await tx.run(query, word1=word1, pos1=pos1, word2=word2, pos2=pos2)


async def create_hypernym_relationship(tx, hypernym_word, hypernym_pos, hyponym_word, hyponym_pos):
    query = """
    MATCH (hyper:Word {word: $hypernym_word, pos: $hypernym_pos})
    MATCH (hypo:Word {word: $hyponym_word, pos: $hyponym_pos})
    MERGE (hyper)-[:HYPERNYM]->(hypo)
    """
    await tx.run(query,
                 hypernym_word=hypernym_word, hypernym_pos=hypernym_pos,
                 hyponym_word=hyponym_word, hyponym_pos=hyponym_pos)


async def create_definition_node(tx, definition):
    query = """
    MERGE (d:Definition {definition: $definition})
    """
    await tx.run(query, definition=definition)


async def create_means_relationship(tx, word, pos, definition):
    query = """
    MATCH (w:Word {word: $word, pos: $pos})
    MATCH (d:Definition {definition: $definition})
    MERGE (w)-[:MEANS]->(d)
    """
    await tx.run(query, word=word, pos=pos, definition=definition)


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


async def create_word_node(tx, word, pos):
    query = """
    MERGE (w:Word {word: $word, pos: $pos})
    """
    await tx.run(query, word=word, pos=pos)


async def main(batch_size: int = 250):
    driver = new_async_driver()
    all_synsets = [s for s in nltk.corpus.wordnet.all_synsets()]
    all_synsets_len = len(all_synsets)
    synset_idx = 0
    max_concurrent_batches = 10
    pending_batches = []
    while synset_idx < all_synsets_len:
        batch = []
        for _ in range(batch_size):
            if synset_idx >= all_synsets_len:
                break
            batch.append(all_synsets[synset_idx])
            synset_idx += 1
        if len(pending_batches) >= max_concurrent_batches:
            await asyncio.gather(*pending_batches)
            pending_batches = []
        else:
            pending_batches.append(asyncio.create_task(process_synset_batch(driver, batch)))
        if synset_idx > 0 and synset_idx % 2500 == 0:
            print(f"Processed {synset_idx} synsets")
    if pending_batches:
        await asyncio.gather(*pending_batches)
    await driver.close()


async def process_synset_batch(driver, batch):
    async with driver.session() as session:
        for synset in batch:
            synset_definition = synset.definition()
            synset_name = synset.name().split('.')[0].replace('_', ' ')
            synset_pos = synset.pos()
            await session.execute_write(create_definition_node, synset_definition)
            await session.execute_write(create_word_node, synset_name, synset_pos)
            await session.execute_write(
                create_means_relationship,
                synset_name, synset_pos, synset_definition)

            for hyponym in synset.hyponyms():
                hyponym_name = hyponym.name().split('.')[0].replace('_', ' ')
                hyponym_pos = hyponym.pos()
                await session.execute_write(create_word_node, hyponym_name, hyponym_pos)
                await session.execute_write(
                    create_hypernym_relationship,
                    synset_name, synset_pos, hyponym_name, hyponym_pos)

            for topic in synset.topic_domains():
                topic_name = topic.name().split('.')[0].replace('_', ' ')
                topic_pos = topic.pos()
                await session.execute_write(create_word_node, topic_name, topic_pos)
                await session.execute_write(
                    create_topic_domain_relationship,
                    topic_name, topic_pos, synset_definition)

            for usage in synset.usage_domains():
                usage_name = usage.name().split('.')[0].replace('_', ' ')
                usage_pos = usage.pos()
                await session.execute_write(create_word_node, usage_name, usage_pos)
                await session.execute_write(
                    create_usage_domain_relationship,
                    usage_name, usage_pos, synset_definition)

            for lemma in synset.lemmas():
                lemma_name = lemma.name().split('.')[0].replace("_", " ")
                antonyms = lemma.antonyms()
                await session.execute_write(create_word_node, lemma_name, synset_pos)
                await session.execute_write(
                    create_means_relationship,
                    lemma_name, synset_pos, synset_definition)
                for antonym in antonyms:
                    antonym_synset = antonym.synset()
                    antonym_definition = antonym_synset.definition()
                    antonym_name = antonym.name().replace("_", " ")
                    antonym_pos = antonym_synset.pos()
                    await session.execute_write(create_definition_node, antonym.synset().definition())
                    await session.execute_write(
                        create_means_relationship,
                        antonym_name, antonym_pos, antonym_definition)
                    await session.execute_write(
                        create_antonym_relationship,
                        lemma_name, synset_pos, antonym_name, antonym_pos)


if __name__ == '__main__':
    nltk.download('wordnet')
    asyncio.run(main())
