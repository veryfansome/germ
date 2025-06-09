import asyncio
import logging
from nltk.corpus.reader import WordNetCorpusReader

from germ.database.neo4j import new_async_driver

logger = logging.getLogger(__name__)

oewn_path = "data/oewn2024"
pos2id = {
    "a": 1,  # Adjective
    "n": 2,  # Noun
    "r": 3,  # Adverb
    "s": 4,  # Adjective Satellite
    "v": 5,  # Verb
}
id2pos = {v: k for k, v in pos2id.items()}

# TODO:
#   - Verify bidirectional antonyms
#   - Dedupe senses with identical connections

async def main():
    driver = new_async_driver()
    reader = WordNetCorpusReader(oewn_path, omw_reader=None)
    all_synsets = [s for s in reader.all_synsets()]
    #foo_cnt = 0
    #for synset in all_synsets:
    #    foo = synset.substance_meronyms()
    #    if foo:
    #        print(f"{synset.name()} -> {foo}")
    #        foo_cnt += 1
    #print(f"{foo_cnt} foo synsets")
    #exit()
    logger.info(f"Processing {len(all_synsets)} synsets")
    processors = [
        process_synset_and_definition_batch,
        process_also_see_batch,
        process_antonym_batch,
        process_attribute_batch,
        process_cause_batch,
        process_entailment_batch,
        process_hypernym_batch,
        process_instance_hypernym_batch,
        process_meronym_batch,
        process_pertainym_batch,
        process_region_domain_batch,
        process_related_form_batch,
        process_root_hypernym_batch,
        #process_substance_meronym_batch,  # TODO: has to be object
        process_topic_domain_batch,
        process_usage_domain_batch,
        process_verb_group_batch,
    ]
    for processor in processors:
        async with driver.session() as session:
            await session.execute_write(processor, all_synsets)


async def process_also_see_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (r:Synset {lemma: in_struct.relation_lemma, pos: in_struct.relation_pos, sense: in_struct.relation_sense})
    MERGE (r)-[:ALSO_SEE]->(s)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for relation in synset.also_sees():
            relation_lemma, relation_pos, relation_sense = tokenize_synset_name(relation.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "relation_lemma": relation_lemma, "relation_pos": relation_pos, "relation_sense": relation_sense,
            })
    logger.info(f"Merged {len(in_struct)} also-sees relationships")
    await tx.run(query, in_struct=in_struct)


async def process_antonym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (a:Synset {lemma: in_struct.antonym_lemma, pos: in_struct.antonym_pos, sense: in_struct.antonym_sense})
    MERGE (s)-[:ANTONYM_OF]->(a)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for lemma in synset.lemmas():
            # Each synset can have many lemmas. Currently, we focus on the main lemma that's used in the synset name
            # because individual lemmas are not associated with senses, which we require.
            if lemma.name() == synset_lemma:
                for antonym in lemma.antonyms():
                    antonym_lemma, antonym_pos, antonym_sense = tokenize_synset_name(antonym.synset().name())
                    in_struct.append({
                        "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                        "antonym_lemma": antonym_lemma, "antonym_pos": antonym_pos, "antonym_sense": antonym_sense,
                    })
    logger.info(f"Merged {len(in_struct)} antonym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_attribute_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (a:Synset {lemma: in_struct.attribute_lemma, pos: in_struct.attribute_pos, sense: in_struct.attribute_sense})
    MERGE (a)-[:ATTRIBUTE_OF]->(s)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for attribute in synset.attributes():
            attribute_lemma, attribute_pos, attribute_sense = tokenize_synset_name(attribute.name())
            # In the data, these relationships are bidirectional but attributes are adjectives that convey a
            # property or condition related to some noun word. For example:
            #
            #   presence.n.01 -> [Synset('absent.a.01'), Synset('present.a.02')]
            #
            if id2pos[synset_pos] == "n":
                in_struct.append({
                    "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                    "attribute_lemma": attribute_lemma, "attribute_pos": attribute_pos, "attribute_sense": attribute_sense,
                })
    logger.info(f"Merged {len(in_struct)} attribute relationships")
    await tx.run(query, in_struct=in_struct)


async def process_cause_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (e:Synset {lemma: in_struct.effect_lemma, pos: in_struct.effect_pos, sense: in_struct.effect_sense})
    MERGE (s)-[:CAUSES]->(e)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for effect in synset.causes():
            effect_lemma, effect_pos, effect_sense = tokenize_synset_name(effect.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "effect_lemma": effect_lemma, "effect_pos": effect_pos, "effect_sense": effect_sense,
            })
    logger.info(f"Merged {len(in_struct)} cause relationships")
    await tx.run(query, in_struct=in_struct)


async def process_entailment_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (i:Synset {lemma: in_struct.imp_lemma, pos: in_struct.imp_pos, sense: in_struct.imp_sense})
    MERGE (s)-[:IMPLIES]->(i)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for implication in synset.entailments():
            implication_lemma, implication_pos, implication_sense = tokenize_synset_name(implication.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "imp_lemma": implication_lemma, "imp_pos": implication_pos, "imp_sense": implication_sense,
            })
    logger.info(f"Merged {len(in_struct)} entailment relationships")
    await tx.run(query, in_struct=in_struct)


async def process_hypernym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (hpr:Synset {lemma: in_struct.hypernym_lemma, pos: in_struct.hypernym_pos, sense: in_struct.hypernym_sense})
    MATCH (hpo:Synset {lemma: in_struct.hyponym_lemma, pos: in_struct.hyponym_pos, sense: in_struct.hyponym_sense})
    MERGE (hpo)-[:IS_A]->(hpr)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for hypernym in synset.hypernyms():
            hypernym_lemma, hypernym_pos, hypernym_sense = tokenize_synset_name(hypernym.name())
            in_struct.append({
                "hypernym_lemma": hypernym_lemma, "hypernym_pos": hypernym_pos, "hypernym_sense": hypernym_sense,
                "hyponym_lemma": synset_lemma, "hyponym_pos": synset_pos, "hyponym_sense": synset_sense,
            })
    logger.info(f"Merged {len(in_struct)} hypernym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_instance_hypernym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (hpr:Synset {lemma: in_struct.hypernym_lemma, pos: in_struct.hypernym_pos, sense: in_struct.hypernym_sense})
    MATCH (hpo:Synset {lemma: in_struct.hyponym_lemma, pos: in_struct.hyponym_pos, sense: in_struct.hyponym_sense})
    MERGE (hpo)-[:INSTANCE_IS_A]->(hpr)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for hypernym in synset.instance_hypernyms():
            hypernym_lemma, hypernym_pos, hypernym_sense = tokenize_synset_name(hypernym.name())
            in_struct.append({
                "hypernym_lemma": hypernym_lemma, "hypernym_pos": hypernym_pos, "hypernym_sense": hypernym_sense,
                "hyponym_lemma": synset_lemma, "hyponym_pos": synset_pos, "hyponym_sense": synset_sense,
            })
    logger.info(f"Merged {len(in_struct)} instance hypernym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_meronym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (m:Synset {lemma: in_struct.member_lemma, pos: in_struct.member_pos, sense: in_struct.member_sense})
    MATCH (g:Synset {lemma: in_struct.group_lemma, pos: in_struct.group_pos, sense: in_struct.group_sense})
    MERGE (m)-[:MEMBER_OF]->(g)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for meronym in synset.member_meronyms():
            meronym_lemma, meronym_pos, meronym_sense = tokenize_synset_name(meronym.name())
            in_struct.append({
                "member_lemma": meronym_lemma, "member_pos": meronym_pos, "member_sense": meronym_sense,
                "group_lemma": synset_lemma, "group_pos": synset_pos, "group_sense": synset_sense,
            })
    logger.info(f"Merged {len(in_struct)} meronym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_pertainym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (p:Synset {lemma: in_struct.pertainym_lemma, pos: in_struct.pertainym_pos, sense: in_struct.pertainym_sense})
    MERGE (s)-[:FORM_OF]->(p)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for lemma in synset.lemmas():
            if lemma.name() == synset_lemma:
                for pertainym in lemma.pertainyms():
                    pertainym_lemma, pertainym_pos, pertainym_sense = tokenize_synset_name(pertainym.synset().name())
                    in_struct.append({
                        "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                        "pertainym_lemma": pertainym_lemma, "pertainym_pos": pertainym_pos, "pertainym_sense": pertainym_sense,
                    })
    logger.info(f"Merged {len(in_struct)} pertainym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_region_domain_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (rd:Synset {lemma: in_struct.domain_lemma, pos: in_struct.domain_pos, sense: in_struct.domain_sense})
    MERGE (s)-[:OF_REGION]->(rd)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for domain in synset.region_domains():
            domain_lemma, domain_pos, domain_sense = tokenize_synset_name(domain.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "domain_lemma": domain_lemma, "domain_pos": domain_pos, "domain_sense": domain_sense,
            })
    logger.info(f"Merged {len(in_struct)} region domain relationships")
    await tx.run(query, in_struct=in_struct)


async def process_related_form_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (r:Synset {lemma: in_struct.related_lemma, pos: in_struct.related_pos, sense: in_struct.related_sense})
    MERGE (s)-[:FORM_OF]->(r)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for lemma in synset.lemmas():
            # Each synset can have many lemmas. Currently, we focus on the main lemma that's used in the synset name
            # because individual lemmas are not associated with senses, which we require.
            if lemma.name() == synset_lemma:
                for derived in lemma.derivationally_related_forms():
                    derived_lemma, derived_pos, derived_sense = tokenize_synset_name(derived.synset().name())
                    in_struct.append({
                        "lemma": derived_lemma, "pos": derived_pos, "sense": derived_sense,
                        "related_lemma": synset_lemma, "related_pos": synset_pos, "related_sense": synset_sense,
                    })
    logger.info(f"Merged {len(in_struct)} related form relationships")
    await tx.run(query, in_struct=in_struct)


async def process_root_hypernym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (hpr:Synset {lemma: in_struct.hypernym_lemma, pos: in_struct.hypernym_pos, sense: in_struct.hypernym_sense})
    MATCH (hpo:Synset {lemma: in_struct.hyponym_lemma, pos: in_struct.hyponym_pos, sense: in_struct.hyponym_sense})
    MERGE (hpo)-[:ROOT_IS_A]->(hpr)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for hypernym in synset.root_hypernyms():
            hypernym_lemma, hypernym_pos, hypernym_sense = tokenize_synset_name(hypernym.name())
            if hypernym_lemma != synset_lemma:
                in_struct.append({
                    "hypernym_lemma": hypernym_lemma, "hypernym_pos": hypernym_pos, "hypernym_sense": hypernym_sense,
                    "hyponym_lemma": synset_lemma, "hyponym_pos": synset_pos, "hyponym_sense": synset_sense,
                })
    logger.info(f"Merged {len(in_struct)} root hypernym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_substance_meronym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (sub:Synset {lemma: in_struct.sub_lemma, pos: in_struct.sub_pos, sense: in_struct.sub_sense})
    MERGE (s)-[:MADE_OF]->(sub)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for substance in synset.member_meronyms():
            substance_lemma, substance_pos, substance_sense = tokenize_synset_name(substance.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "sub_lemma": substance_lemma, "sub_pos": substance_pos, "sub_sense": substance_sense,
            })
    logger.info(f"Merged {len(in_struct)} substance meronym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_synset_and_definition_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MERGE (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MERGE (d:SynsetDefinition {text: in_struct.definition})
    MERGE (d)-[:DEFINES]->(s)
    """
    in_struct = [
        {
            "lemma": pair[0][0],
            "pos": pair[0][1],
            "sense": pair[0][2],
            "definition": pair[1].definition(),
        }
        for pair in [(tokenize_synset_name(s.name()), s) for s in synsets]
    ]
    logger.info(f"Merged {len(in_struct)} synsets")
    await tx.run(query, in_struct=in_struct)


async def process_topic_domain_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (td:Synset {lemma: in_struct.domain_lemma, pos: in_struct.domain_pos, sense: in_struct.domain_sense})
    MERGE (s)-[:OF_TOPIC]->(td)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for domain in synset.topic_domains():
            domain_lemma, domain_pos, domain_sense = tokenize_synset_name(domain.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "domain_lemma": domain_lemma, "domain_pos": domain_pos, "domain_sense": domain_sense,
            })
    logger.info(f"Merged {len(in_struct)} topic domain relationships")
    await tx.run(query, in_struct=in_struct)


async def process_usage_domain_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (ud:Synset {lemma: in_struct.domain_lemma, pos: in_struct.domain_pos, sense: in_struct.domain_sense})
    MERGE (s)-[:USED_AS]->(ud)
    """
    # Convert the list of tuples into a list of dicts for the query
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for domain in synset.usage_domains():
            domain_lemma, domain_pos, domain_sense = tokenize_synset_name(domain.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "domain_lemma": domain_lemma, "domain_pos": domain_pos, "domain_sense": domain_sense,
            })
    logger.info(f"Merged {len(in_struct)} usage domain relationships")
    await tx.run(query, in_struct=in_struct)


async def process_verb_group_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (p:Synset {lemma: in_struct.peer_lemma, pos: in_struct.peer_pos, sense: in_struct.peer_sense})
    MERGE (s)-[:VERB_GROUP_WITH]->(p)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for peer in synset.verb_groups():
            peer_lemma, peer_pos, peer_sense = tokenize_synset_name(peer.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "peer_lemma": peer_lemma, "peer_pos": peer_pos, "peer_sense": peer_sense,
            })
    logger.info(f"Merged {len(in_struct)} verb group relationships")
    await tx.run(query, in_struct=in_struct)


def tokenize_synset_name(synset_name: str):
    components = synset_name.split('.')
    synset_sense = components.pop()
    synset_pos = components.pop()
    synset_name = '.'.join(components).replace('_', ' ')
    return synset_name, pos2id[synset_pos], int(synset_sense)


if __name__ == '__main__':
    import nltk
    from germ.observability.logging import setup_logging

    nltk.download('wordnet')
    setup_logging()
    asyncio.run(main())
