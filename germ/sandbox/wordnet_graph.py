import asyncio
import nltk

from germ.database.neo4j import new_async_driver

pos2id = {
    "a": 1,  # Adjective
    "n": 2,  # Noun
    "r": 3,  # Adverb
    "s": 4,  # Adjective Satellite
    "v": 5,  # Verb
}
id2pos = {v: k for k, v in pos2id.items()}


async def create_antonym_relationship(tx, word1, pos1, sense1, word2, pos2, sense2):
    query = """
    MATCH (w1:Word {word: $word1, pos: $pos1, sense: $sense1})
    MATCH (w2:Word {word: $word2, pos: $pos2, sense: $sense2})
    MERGE (w1)-[:ANTONYM]->(w2)
    """
    await tx.run(query, word1=word1, pos1=pos1, sense1=sense1, word2=word2, pos2=pos2, sense2=sense2)


async def create_definition_node(tx, definition):
    query = """
    MERGE (d:Definition {definition: $definition})
    """
    await tx.run(query, definition=definition)


async def create_kind_of_relationship(tx, hypernym_word, hypernym_pos, hypernym_sense,
                                       hyponym_word, hyponym_pos, hyponym_sense):
    query = """
    MATCH (hyper:Word {word: $hypernym_word, pos: $hypernym_pos, sense: $hypernym_sense})
    MATCH (hypo:Word {word: $hyponym_word, pos: $hyponym_pos, sense: $hyponym_sense})
    MERGE (hypo)-[:KIND_OF]->(hyper)
    """
    await tx.run(query,
                 hypernym_word=hypernym_word, hypernym_pos=hypernym_pos, hypernym_sense=hypernym_sense,
                 hyponym_word=hyponym_word, hyponym_pos=hyponym_pos, hyponym_sense=hyponym_sense)


async def create_means_relationship(tx, word, pos, sense, definition):
    query = """
    MATCH (w:Word {word: $word, pos: $pos, sense: $sense})
    MATCH (d:Definition {definition: $definition})
    MERGE (w)-[:MEANS]->(d)
    """
    await tx.run(query, word=word, pos=pos, sense=sense, definition=definition)


async def create_member_of_relationship(tx, member_word, member_pos, member_sense,
                                       group_word, group_pos, group_sense):
    query = """
    MATCH (m:Word {word: $member_word, pos: $member_pos, sense: $member_sense})
    MATCH (g:Word {word: $group_word, pos: $group_pos, sense: $group_sense})
    MERGE (m)-[:MEMBER_OF]->(g)
    """
    await tx.run(query,
                 member_word=member_word, member_pos=member_pos, member_sense=member_sense,
                 group_word=group_word, group_pos=group_pos, group_sense=group_sense)


async def create_topic_domain_relationship(tx, word, pos, sense, definition):
    query = """
    MATCH (w:Word {word: $word, pos: $pos, sense: $sense})
    MATCH (d:Definition {definition: $definition})
    MERGE (d)-[:TOPIC_DOMAIN]->(w)
    """
    await tx.run(query, word=word, pos=pos, sense=sense, definition=definition)


async def create_usage_domain_relationship(tx, word, pos, sense, domain_word, domain_pos, domain_sense):
    query = """
    MATCH (w:Word {word: $word, pos: $pos, sense: $sense})
    MATCH (d:Word {word: $domain_word, pos: $domain_pos, sense: $domain_sense})
    MERGE (w)-[:USAGE_DOMAIN]->(d)
    """
    await tx.run(query, word=word, pos=pos, sense=sense, domain_word=domain_word, domain_pos=domain_pos, domain_sense=domain_sense)


async def create_word_node(tx, word, pos, sense):
    query = """
    MERGE (w:Word {word: $word, pos: $pos, sense: $sense})
    """
    await tx.run(query, word=word, pos=pos, sense=sense)


async def main(batch_size: int = 250):
    driver = new_async_driver()
    all_synsets = [s for s in nltk.corpus.wordnet.all_synsets()]
    all_synsets_len = len(all_synsets)
    synset_idx = 0
    max_concurrent_batches = 10
    pending_batches = []
    foo = set()
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
            pending_batches.append(asyncio.create_task(process_synset_batch(driver, batch, foo)))
        if synset_idx > 0 and synset_idx % 2500 == 0:
            print(f"Processed {synset_idx} synsets")
    if pending_batches:
        await asyncio.gather(*pending_batches)
    await driver.close()


async def process_synset_batch(driver, batch, foo):
    async with driver.session() as session:
        for synset in batch:
            synset_definition = synset.definition()
            synset_name, synset_pos, synset_sense = tokenize_synset_name(synset.name())

            await session.execute_write(create_definition_node, synset_definition)
            if synset.name() not in foo:
                await session.execute_write(create_word_node, synset_name, synset_pos, synset_sense)
                foo.add(synset.name())
            await session.execute_write(
                create_means_relationship,
                synset_name, synset_pos, synset_sense, synset_definition)

            for holonym in synset.member_holonyms():
                holonym_name, holonym_pos, holonym_sense = tokenize_synset_name(holonym.name())
                if holonym.name() not in foo:
                    await session.execute_write(create_word_node, holonym_name, holonym_pos, holonym_sense)
                    foo.add(holonym.name())
                rel_key = f"{synset.name()}_member_of_{holonym.name()}"
                if rel_key not in foo:
                    await session.execute_write(
                        create_member_of_relationship,
                        synset_name, synset_pos, synset_sense,
                        holonym_name, holonym_pos, holonym_sense)
                    foo.add(rel_key)
            for meronym in synset.member_meronyms():
                meronym_name, meronym_pos, meronym_sense = tokenize_synset_name(meronym.name())
                if meronym.name() not in foo:
                    await session.execute_write(create_word_node, meronym_name, meronym_pos, meronym_sense)
                    foo.add(meronym.name())
                rel_key = f"{meronym.name()}_member_of_{synset.name()}"
                if rel_key not in foo:
                    await session.execute_write(
                        create_member_of_relationship,
                        meronym_name, meronym_pos, meronym_sense,
                        synset_name, synset_pos, synset_sense)
                    foo.add(rel_key)

            for hyponym in synset.hyponyms():
                hyponym_name, hyponym_pos, hyponym_sense = tokenize_synset_name(hyponym.name())
                if hyponym.name() not in foo:
                    await session.execute_write(create_word_node, hyponym_name, hyponym_pos, hyponym_sense)
                    foo.add(hyponym.name())
                rel_key = f"{synset.name()}_kind_of_{hyponym.name()}"
                if rel_key not in foo:
                    await session.execute_write(
                        create_kind_of_relationship,
                        synset_name, synset_pos, synset_sense,
                        hyponym_name, hyponym_pos, hyponym_sense)
                    foo.add(rel_key)
            for hypernym in synset.hypernyms():
                hypernym_name, hypernym_pos, hypernym_sense = tokenize_synset_name(hypernym.name())
                if hypernym.name() not in foo:
                    await session.execute_write(create_word_node, hypernym_name, hypernym_pos, hypernym_sense)
                    foo.add(hypernym.name())
                rel_key = f"{hypernym.name()}_kind_of_{synset.name()}"
                if rel_key not in foo:
                    await session.execute_write(
                        create_kind_of_relationship,
                        hypernym_name, hypernym_pos, hypernym_sense,
                        synset_name, synset_pos, synset_sense)
                    foo.add(rel_key)

            for topic in synset.topic_domains():
                topic_name, topic_pos, topic_sense = tokenize_synset_name(topic.name())
                if topic.name() not in foo:
                    await session.execute_write(create_word_node, topic_name, topic_pos, topic_sense)
                    foo.add(topic.name())
                await session.execute_write(
                    create_topic_domain_relationship,
                    topic_name, topic_pos, topic_sense, synset_definition)

            for usage in synset.usage_domains():
                usage_name, usage_pos, usage_sense = tokenize_synset_name(usage.name())
                if usage.name() not in foo:
                    await session.execute_write(create_word_node, usage_name, usage_pos, usage_sense)
                    foo.add(usage.name())
                await session.execute_write(
                    create_usage_domain_relationship,
                    synset_name, synset_pos, synset_sense,
                    usage_name, usage_pos, usage_sense)

            for lemma in synset.lemmas():
                lemma_name = lemma.name()
                if lemma_name == synset_name:
                    antonyms = lemma.antonyms()
                    for antonym in antonyms:
                        antonym_synset = antonym.synset()
                        antonym_name, antonym_pos, antonym_sense = tokenize_synset_name(antonym_synset.name())
                        if antonym_synset.name() not in foo:
                            await session.execute_write(create_word_node, antonym_name, antonym_pos, antonym_sense)
                            foo.add(antonym_synset.name())
                        await session.execute_write(
                            create_antonym_relationship,
                            synset_name, synset_pos, synset_sense, antonym_name, antonym_pos, antonym_sense)


def tokenize_synset_name(synset_name: str):
    components = synset_name.split('.')
    synset_sense = components.pop()
    synset_pos = components.pop()
    synset_name = '.'.join(components).replace('_', ' ')
    return synset_name, pos2id[synset_pos], int(synset_sense)


if __name__ == '__main__':
    nltk.download('wordnet')
    asyncio.run(main())
