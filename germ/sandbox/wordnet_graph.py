import asyncio
import logging
from difflib import SequenceMatcher
from nltk.corpus.reader import WordNetCorpusReader

from germ.database.neo4j import new_async_driver

logger = logging.getLogger(__name__)

OEWN_PATH = "data/oewn2024"

# TODO:
#   - Verify bidirectional antonyms
#   - Dedupe senses with identical connections

async def main():
    driver = new_async_driver()
    reader = WordNetCorpusReader(OEWN_PATH, omw_reader=None)
    all_synsets = [s for s in reader.all_synsets()]

    #foo_cnt = 0
    #for synset in all_synsets:
    #    foo = synset.similar_tos()
    #    if foo:
    #        print(f"{synset.name()} -> {foo}")
    #        foo_cnt += 1
    #    synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
    #    #for lemma in synset.lemmas():
    #    #    #if lemma.name() == synset_lemma:
    #    #    for derived_form in lemma.derivationally_related_forms():
    #    #        if lemma.name() != derived_form.name():
    #    #            print(f"{lemma.name()} -> {derived_form}")
    #print(f"{foo_cnt} foo synsets")
    #exit()

    logger.info(f"Processing {len(all_synsets)} synsets")
    processors = [
        process_synset_and_definition_batch,
        process_also_see_batch,
        process_antonym_batch,
        process_attribute_batch,
        process_cause_batch,
        process_derived_from_batch,
        process_entailment_batch,
        process_hypernym_batch,
        process_instance_hypernym_batch,
        process_member_meronym_batch,
        process_part_meronym_batch,
        process_pertainym_batch,
        process_region_domain_batch,
        process_root_hypernym_batch,
        process_substance_meronym_batch,
        process_topic_domain_batch,
        process_usage_domain_batch,
        process_verb_group_batch,
    ]
    for processor in processors:
        async with driver.session() as session:
            await session.execute_write(processor, all_synsets)


async def _process_also_see_batch(tx, in_struct):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (r:Synset {lemma: in_struct.relation_lemma, pos: in_struct.relation_pos, sense: in_struct.relation_sense})
    MERGE (r)-[:ALSO_SEE]-(s)
    """
    logger.info(f"Merging {len(in_struct)} also-sees relationships")
    await tx.run(query, in_struct=in_struct)


async def process_also_see_batch(tx, synsets):
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for relation in (synset.also_sees() + synset.similar_tos()):
            relation_lemma, relation_pos, relation_sense = tokenize_synset_name(relation.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "relation_lemma": relation_lemma, "relation_pos": relation_pos, "relation_sense": relation_sense,
            })
    await _process_also_see_batch(tx, in_struct)


async def process_antonym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (a:Synset {lemma: in_struct.antonym_lemma, pos: in_struct.antonym_pos, sense: in_struct.antonym_sense})
    MERGE (s)-[:ANTONYM_OF {pair: in_struct.pair}]-(a)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                antonym_lemma, antonym_pos, antonym_sense = tokenize_synset_name(antonym.synset().name())
                in_struct.append({
                    "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                    "antonym_lemma": antonym_lemma, "antonym_pos": antonym_pos, "antonym_sense": antonym_sense,
                    # Non-directional cyper so sorted to dedupe bidirectional relationships from WordNet
                    "pair": " <> ".join(sorted([f"{synset_pos}." + lemma.name().replace('_', ' '),
                                                f"{antonym_pos}." + antonym.name().replace('_', ' ')])),
                })
    logger.info(f"Merging {len(in_struct)} antonym relationships")
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
            # Attributes are adjectives that convey a property or condition related to some noun word. For example:
            #
            #   presence.n.01 -> [Synset('absent.a.01'), Synset('present.a.02')]
            #
            if synset_pos == "n":
                in_struct.append({
                    "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                    "attribute_lemma": attribute_lemma, "attribute_pos": attribute_pos, "attribute_sense": attribute_sense,
                })
    logger.info(f"Merging {len(in_struct)} attribute relationships")
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
    logger.info(f"Merging {len(in_struct)} cause relationships")
    await tx.run(query, in_struct=in_struct)


async def process_derived_from_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (d:Synset {lemma: in_struct.derived_lemma, pos: in_struct.derived_pos, sense: in_struct.derived_sense})
    MERGE (d)-[:DERIVED_FROM {pair: in_struct.pair}]->(s)
    """
    also_see_struct = []
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for lemma in synset.lemmas():
            lemma_name = lemma.name()
            for derived in lemma.derivationally_related_forms():
                derived_lemma, derived_pos, derived_sense = tokenize_synset_name(derived.synset().name())
                derived_name = derived.name()
                if lemma_name == derived_name:
                    pass  # Avoid cases of same lemma but different pos
                elif lemma_name in derived_name or (
                        len(lemma_name) < len(derived_name)
                        and similar_char_ratio(lemma.name(), derived.name(), threshold=0.7)):
                    # Focus on morphologies
                    in_struct.append({
                        "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                        "derived_lemma": derived_lemma, "derived_pos": derived_pos, "derived_sense": derived_sense,
                        "pair": " >> ".join([f"{derived_pos}." + derived.name().replace('_', ' '),
                                             f"{synset_pos}." + lemma.name().replace('_', ' ')]),
                    })
                else:
                    # If not morphology, mark as ALSO_SEE
                    also_see_struct.append({
                        "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                        "relation_lemma": derived_lemma, "relation_pos": derived_pos, "relation_sense": derived_sense,
                    })
    logger.info(f"Merging {len(in_struct)} derived from and {len(also_see_struct)} also-see relationships")
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
    logger.info(f"Merging {len(in_struct)} entailment relationships")
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
    logger.info(f"Merging {len(in_struct)} hypernym relationships")
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
    logger.info(f"Merging {len(in_struct)} instance hypernym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_member_meronym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (m:Synset {lemma: in_struct.member_lemma, pos: in_struct.member_pos, sense: in_struct.member_sense})
    MERGE (m)-[:MEMBER_OF]->(s)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for member in synset.member_meronyms():
            member_lemma, member_pos, member_sense = tokenize_synset_name(member.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "member_lemma": member_lemma, "member_pos": member_pos, "member_sense": member_sense,
            })
    logger.info(f"Merging {len(in_struct)} member meronym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_part_meronym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (p:Synset {lemma: in_struct.part_lemma, pos: in_struct.part_pos, sense: in_struct.part_sense})
    MERGE (p)-[:PART_OF]->(s)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for part in synset.part_meronyms():
            part_lemma, part_pos, part_sense = tokenize_synset_name(part.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "part_lemma": part_lemma, "part_pos": part_pos, "part_sense": part_sense,
            })
    logger.info(f"Merging {len(in_struct)} part meronym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_pertainym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (p:Synset {lemma: in_struct.pertainym_lemma, pos: in_struct.pertainym_pos, sense: in_struct.pertainym_sense})
    MERGE (p)-[:PERTAINYM_OF {pair: in_struct.pair}]->(s)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for lemma in synset.lemmas():
            # Pertainyms are adjectives that are derived to form some adverb. For example:
            #
            #   insufferably.r.02.insufferably -> [Lemma('insufferable.s.01.insufferable')]
            #
            if synset_pos == "r":  # The WordNet data deviates from this pattern, but we enforce it.
                # TODO: Send rest to ALSO_SEE?
                for pertainym in lemma.pertainyms():
                    pertainym_lemma, pertainym_pos, pertainym_sense = tokenize_synset_name(pertainym.synset().name())
                    in_struct.append({
                        "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                        "pertainym_lemma": pertainym_lemma, "pertainym_pos": pertainym_pos, "pertainym_sense": pertainym_sense,
                        "pair": " >> ".join([f"{synset_pos}." + lemma.name().replace('_', ' '),
                                             f"{pertainym_pos}." + pertainym.name().replace('_', ' ')]),
                    })
    logger.info(f"Merging {len(in_struct)} pertainym relationships")
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
    logger.info(f"Merging {len(in_struct)} region domain relationships")
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
    logger.info(f"Merging {len(in_struct)} root hypernym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_substance_meronym_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (sub:Synset {lemma: in_struct.sub_lemma, pos: in_struct.sub_pos, sense: in_struct.sub_sense})
    MERGE (sub)-[:SUBSTANCE_IN]->(s)
    """
    in_struct = []
    for synset in synsets:
        synset_lemma, synset_pos, synset_sense = tokenize_synset_name(synset.name())
        for substance in synset.substance_meronyms():
            substance_lemma, substance_pos, substance_sense = tokenize_synset_name(substance.name())
            in_struct.append({
                "lemma": synset_lemma, "pos": synset_pos, "sense": synset_sense,
                "sub_lemma": substance_lemma, "sub_pos": substance_pos, "sub_sense": substance_sense,
            })
    logger.info(f"Merging {len(in_struct)} substance meronym relationships")
    await tx.run(query, in_struct=in_struct)


async def process_synset_and_definition_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MERGE (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    WITH in_struct, s
    SET s.lemmas = in_struct.lemmas
    WITH in_struct, s
    MERGE (d:SynsetDefinition {text: in_struct.definition})
    MERGE (d)-[:DEFINES]->(s)
    """
    in_struct = [
        {
            "lemma": pair[0][0],
            "lemmas": [l.name().replace("_", " ") for l in pair[1].lemmas()],
            "pos": pair[0][1],
            "sense": pair[0][2],
            "definition": pair[1].definition(),
        }
        for pair in [(tokenize_synset_name(s.name()), s) for s in synsets]
    ]
    logger.info(f"Merging {len(in_struct)} synsets")
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
    logger.info(f"Merging {len(in_struct)} topic domain relationships")
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
    logger.info(f"Merging {len(in_struct)} usage domain relationships")
    await tx.run(query, in_struct=in_struct)


async def process_verb_group_batch(tx, synsets):
    query = """
    UNWIND $in_struct AS in_struct
    MATCH (s:Synset {lemma: in_struct.lemma, pos: in_struct.pos, sense: in_struct.sense})
    MATCH (p:Synset {lemma: in_struct.peer_lemma, pos: in_struct.peer_pos, sense: in_struct.peer_sense})
    MERGE (s)-[:VERB_GROUP_WITH]-(p)
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
    logger.info(f"Merging {len(in_struct)} verb group relationships")
    await tx.run(query, in_struct=in_struct)


def similar_char_ratio(a: str, b: str, *, threshold: float = 0.5) -> bool:
    """
    Returns True if *either* string contains a run of consecutive characters
    that covers more than `threshold` · min(len(a), len(b)).
    """
    # Longest common *consecutive* substring length
    longest = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b)
    ).size
    return longest / max(len(a), len(b)) > threshold


def tokenize_synset_name(synset_name: str):
    components = synset_name.split('.')
    synset_sense = components.pop()
    synset_pos = components.pop()
    synset_name = '.'.join(components).replace('_', ' ')
    return synset_name, synset_pos, int(synset_sense)


if __name__ == '__main__':
    import nltk
    from germ.observability.logging import setup_logging

    nltk.download('wordnet')
    setup_logging()
    asyncio.run(main())
