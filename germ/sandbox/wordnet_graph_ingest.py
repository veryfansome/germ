import asyncio
import nltk

from germ.database.neo4j import new_async_driver

# ---------------------------
# Batch creation queries
# ---------------------------

async def batch_create_words(tx, words):
    """
    words is a list of 2-tuples or dicts like:
    [ (word, pos), ... ] or [ {'word': w, 'pos': p}, ... ]
    Here we'll assume 2-tuples and convert them.
    """
    query = """
    UNWIND $words AS w
    MERGE (node:Word {word: w.word, pos: w.pos})
    """
    # Convert the list of tuples into a list of dicts for the query
    words_dicts = [{'word': w[0], 'pos': w[1]} for w in set(words)]
    await tx.run(query, words=words_dicts)

async def batch_create_definitions(tx, definitions):
    """
    definitions is a list of strings.
    """
    query = """
    UNWIND $defs AS def
    MERGE (d:Definition {definition: def})
    """
    # Use set to avoid duplicates
    unique_defs = list(set(definitions))
    await tx.run(query, defs=unique_defs)

async def batch_create_means_relationships(tx, means_rels):
    """
    means_rels is a list of 3-tuples like
    [ (word, pos, definition), ... ]
    """
    query = """
    UNWIND $rels AS rel
    MATCH (w:Word {word: rel.word, pos: rel.pos})
    MATCH (d:Definition {definition: rel.definition})
    MERGE (w)-[:MEANS]->(d)
    """
    # Convert to dicts
    rels_dicts = [
        {
            'word': r[0],
            'pos': r[1],
            'definition': r[2]
        }
        for r in set(means_rels)
    ]
    await tx.run(query, rels=rels_dicts)

async def batch_create_hypernym_relationships(tx, hypernyms):
    """
    hypernyms is a list of tuples:
    [ (hypernym_word, hypernym_pos, hyponym_word, hyponym_pos), ... ]
    """
    query = """
    UNWIND $rels AS rel
    MATCH (hyper:Word {word: rel.hyper_word, pos: rel.hyper_pos})
    MATCH (hypo:Word {word: rel.hypo_word, pos: rel.hypo_pos})
    MERGE (hyper)-[:HYPERNYM]->(hypo)
    """
    rels_dicts = [
        {
            'hyper_word': r[0],
            'hyper_pos': r[1],
            'hypo_word': r[2],
            'hypo_pos': r[3],
        }
        for r in set(hypernyms)
    ]
    await tx.run(query, rels=rels_dicts)

async def batch_create_topic_domain_relationships(tx, topic_rels):
    """
    topic_rels is a list of tuples:
    [ (word, pos, definition), ... ]
    indicating that Word -> TOPIC_DOMAIN -> Definition.
    """
    query = """
    UNWIND $rels AS rel
    MATCH (w:Word {word: rel.word, pos: rel.pos})
    MATCH (d:Definition {definition: rel.definition})
    MERGE (w)-[:TOPIC_DOMAIN]->(d)
    """
    rels_dicts = [
        {
            'word': r[0],
            'pos': r[1],
            'definition': r[2]
        }
        for r in set(topic_rels)
    ]
    await tx.run(query, rels=rels_dicts)

async def batch_create_usage_domain_relationships(tx, usage_rels):
    """
    usage_rels is a list of tuples:
    [ (word, pos, definition), ... ]
    indicating that Word -> USAGE_DOMAIN -> Definition.
    """
    query = """
    UNWIND $rels AS rel
    MATCH (w:Word {word: rel.word, pos: rel.pos})
    MATCH (d:Definition {definition: rel.definition})
    MERGE (w)-[:USAGE_DOMAIN]->(d)
    """
    rels_dicts = [
        {
            'word': r[0],
            'pos': r[1],
            'definition': r[2]
        }
        for r in set(usage_rels)
    ]
    await tx.run(query, rels=rels_dicts)

async def batch_create_antonym_relationships(tx, antonyms):
    """
    antonyms is a list of tuples:
    [ (word1, pos1, word2, pos2), ... ]
    indicating that Word1 -> ANTONYM -> Word2.
    """
    query = """
    UNWIND $rels AS rel
    MATCH (w1:Word {word: rel.word1, pos: rel.pos1})
    MATCH (w2:Word {word: rel.word2, pos: rel.pos2})
    MERGE (w1)-[:ANTONYM]->(w2)
    """
    rels_dicts = [
        {
            'word1': r[0],
            'pos1': r[1],
            'word2': r[2],
            'pos2': r[3]
        }
        for r in set(antonyms)
    ]
    await tx.run(query, rels=rels_dicts)

# ---------------------------
# Main processing
# ---------------------------

async def process_synset_batch(driver, batch):
    # We'll gather everything here first
    words = []
    definitions = []
    means_relationships = []
    hypernym_relationships = []
    topic_domain_relationships = []
    usage_domain_relationships = []
    antonym_relationships = []

    for synset in batch:
        synset_definition = synset.definition()
        synset_name = synset.name().split('.')[0].replace('_', ' ')
        synset_pos = synset.pos()

        # Collect the node & definition
        definitions.append(synset_definition)
        words.append((synset_name, synset_pos))
        means_relationships.append((synset_name, synset_pos, synset_definition))

        # Hyponyms -> hypernym relationships
        for hyponym in synset.hyponyms():
            hyponym_name = hyponym.name().split('.')[0].replace('_', ' ')
            hyponym_pos = hyponym.pos()
            words.append((hyponym_name, hyponym_pos))
            hypernym_relationships.append((synset_name, synset_pos, hyponym_name, hyponym_pos))

        # Topic domains
        for topic in synset.topic_domains():
            topic_name = topic.name().split('.')[0].replace('_', ' ')
            topic_pos = topic.pos()
            words.append((topic_name, topic_pos))
            topic_domain_relationships.append((topic_name, topic_pos, synset_definition))

        # Usage domains
        for usage in synset.usage_domains():
            usage_name = usage.name().split('.')[0].replace('_', ' ')
            usage_pos = usage.pos()
            words.append((usage_name, usage_pos))
            usage_domain_relationships.append((usage_name, usage_pos, synset_definition))

        # Lemmas and antonyms
        for lemma in synset.lemmas():
            lemma_name = lemma.name().split('.')[0].replace('_', ' ')
            words.append((lemma_name, synset_pos))
            means_relationships.append((lemma_name, synset_pos, synset_definition))

            # Antonyms
            for antonym in lemma.antonyms():
                antonym_synset = antonym.synset()
                antonym_name = antonym.name().replace('_', ' ')
                antonym_pos = antonym_synset.pos()
                antonym_definition = antonym_synset.definition()

                # We'll also need the antonym node + definition
                words.append((antonym_name, antonym_pos))
                definitions.append(antonym_definition)
                means_relationships.append((antonym_name, antonym_pos, antonym_definition))
                antonym_relationships.append((lemma_name, synset_pos, antonym_name, antonym_pos))

    # Now perform all merges in a few bulk queries:
    async with driver.session() as session:
        await session.execute_write(batch_create_words, words)
        await session.execute_write(batch_create_definitions, definitions)
        await session.execute_write(batch_create_means_relationships, means_relationships)
        await session.execute_write(batch_create_hypernym_relationships, hypernym_relationships)
        await session.execute_write(batch_create_topic_domain_relationships, topic_domain_relationships)
        await session.execute_write(batch_create_usage_domain_relationships, usage_domain_relationships)
        await session.execute_write(batch_create_antonym_relationships, antonym_relationships)

async def main(batch_size: int = 250):
    driver = new_async_driver()
    all_synsets = list(nltk.corpus.wordnet.all_synsets())
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

        # Enforce a maximum number of concurrent batch tasks
        if len(pending_batches) >= max_concurrent_batches:
            await asyncio.gather(*pending_batches)
            pending_batches = []

        pending_batches.append(asyncio.create_task(process_synset_batch(driver, batch)))

        # Optional progress logging
        if synset_idx > 0 and synset_idx % 2500 == 0:
            print(f"Processed {synset_idx} synsets")

    # Process any leftover batches
    if pending_batches:
        await asyncio.gather(*pending_batches)

    await driver.close()

if __name__ == '__main__':
    nltk.download('wordnet')
    asyncio.run(main())
