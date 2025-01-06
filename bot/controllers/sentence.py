import asyncio
import inflect
from starlette.concurrency import run_in_threadpool

from bot.graph.idea import SentenceMergeEventHandler, idea_graph
from bot.lang.classifiers import (
    ADJECTIVE_POS_TAGS, ADVERB_POS_TAGS, NOUN_POS_TAGS, PRONOUN_POS_TAGS, VERB_POS_TAGS,
    get_flair_pos_tags, get_noun_modifier_classifier, get_sentence_nouns_classifier, get_single_noun_classifier)
from observability.logging import logging, setup_logging

inflect_engine = inflect.engine()
logger = logging.getLogger(__name__)


class SentenceController(SentenceMergeEventHandler):
    def __init__(self, interval_seconds: int = 30):
        self.interval_seconds = interval_seconds
        self.sentence_merge_artifacts = []

    async def on_periodic_run(self):
        logger.info(f"on_periodic_run: {self.sentence_merge_artifacts}")

    async def on_sentence_merge(self, sentence: str, sentence_id: int, sentence_parameters):
        logger.info(f"on_sentence_merge: sentence_id={sentence_id}, {sentence_parameters}, {sentence}")

        # Get POS in the background
        pos_task = asyncio.create_task(run_in_threadpool(get_flair_pos_tags, sentence))

        # TODO: is sorting by connections enough or should we also care about how many are recent?
        # Query semantic categories from graph, sorted by most connections
        semantic_categories = [t["semanticCategory"]["text"] for t in await idea_graph.get_semantic_category_desc_by_connections()]
        logger.debug(f"semantic_categories: {semantic_categories}")

        # "artifacts" that can be cached or persisted
        artifacts = {
            # Used noun types from query as hints to classify sentence
            #"openai_nouns": await run_in_threadpool(get_sentence_nouns_classifier(semantic_categories).classify,
            #                                        sentence, review=False, review_json=None),
            "pos_tags": await pos_task,
            "semantic_categories": semantic_categories,
            "sentence": sentence,
            "sentence_id": sentence_id,
            "sentence_parameters": sentence_parameters,
        }
        logger.info(f"pos_tags: {artifacts["pos_tags"]}")

        # Nouns and POS:
        # Start with nouns because they're the things we talk about. Logic dealing with pronouns and adjectives both
        # depend on their nouns and verbs don't make sense without nouns either so nouns is a good place to start.
        # We'll need to iterate through all the words at this point, so it makes sense to also do all the POS vertexes
        # here as well.
        nouns = []
        last_pos_tag = None
        for idx, word in enumerate(artifacts["pos_tags"]):
            text, tag = word
            await idea_graph.add_part_of_speech(tag)
            if last_pos_tag is not None:
                task = asyncio.create_task(  # Link in background
                    idea_graph.link_pos_tag_to_last_pos_tag(tag, idx, last_pos_tag, idx-1, sentence_id))
            last_pos_tag = tag

            if tag not in NOUN_POS_TAGS:
                continue
            elif idx > 0 and artifacts["pos_tags"][idx-1][1] in ["''", '""']:
                nouns.append((f"{artifacts["pos_tags"][idx-1][0]}{text}", tag))
            else:
                nouns.append(word)
        logger.info(f"nouns: {nouns}")
        for noun, tag in nouns:
            if tag == "NN":  # Common singular
                await idea_graph.add_noun_form(noun, noun)
                await idea_graph.link_noun_form_to_sentence(noun, noun, tag, sentence_id)
            elif tag == "NNP":  # Proper singular
                await idea_graph.add_proper_noun_form(noun, noun)
                await idea_graph.link_proper_noun_form_to_sentence(noun, noun, tag, sentence_id)
            elif tag == "NNS":  # Common plural
                singular = await run_in_threadpool(inflect_engine.singular_noun, noun)
                await idea_graph.add_noun_form(singular, noun)
                await idea_graph.link_noun_form_to_sentence(singular, noun, tag, sentence_id)
            else:
                # TODO: flesh out plural proper noun handling since we can't use inflect
                logger.info(f"plural proper noun handling is WIP: {noun},{tag}")

        pronouns = [word for word in artifacts["pos_tags"] if word[1] in PRONOUN_POS_TAGS]
        logger.info(f"pronouns: {pronouns}")

        adjectives = [word for word in artifacts["pos_tags"] if word[1] in ADJECTIVE_POS_TAGS]
        logger.info(f"adjectives: {adjectives}")

        verbs = [word for word in artifacts["pos_tags"] if word[1] in VERB_POS_TAGS]
        logger.info(f"verbs: {verbs}")

        adverbs = [word for word in artifacts["pos_tags"] if word[1] in ADVERB_POS_TAGS]
        logger.info(f"adverbs: {adverbs}")

        #logger.info(f"openai_nouns: {artifacts["openai_nouns"]}")
        tasks_to_await = []
        #if "nouns" not in artifacts["openai_nouns"]:
        #    logger.warning(f"expected `nouns` field is missing in: {artifacts["openai_nouns"]}")
        #else:
        #    for noun in artifacts["openai_nouns"]["nouns"]:
        #        try:
        #            logger.info(f"noun_text: {noun["noun"]}")
        #            tasks_to_await.append(asyncio.create_task(modifier_peeler(
        #                noun["noun"], noun["plurality"], sentence, sentence_id, semantic_categories,
        #                semantic_category=noun["semanticCategory"])))
        #        except Exception as e:
        #            logger.warning(f"failed to add noun: {noun}", e)
        self.sentence_merge_artifacts.append(artifacts)


async def add_noun(noun_text: str, semantic_category: str, plurality: str, sentence_id: int):
    if plurality == "plural":
        singular_form = inflect_engine.singular_noun(noun_text)
        if not singular_form:
            logger.warning(f"inflect and classifier don't agree: {noun_text}, classifier {plurality}")
        else:
            # Add and link singular form
            noun_record, semantic_category_record = await asyncio.gather(*[
                idea_graph.add_noun(singular_form),
                idea_graph.add_semantic_category(semantic_category)])
            await idea_graph.link_noun_to_semantic_category(singular_form, semantic_category, sentence_id)
            await idea_graph.link_noun_to_sentence(singular_form, sentence_id, plurality=plurality)
            return noun_record, semantic_category_record
    else:
        noun_record, semantic_category_record = await asyncio.gather(*[
            idea_graph.add_noun(noun_text),
            idea_graph.add_semantic_category(semantic_category)])
        await idea_graph.link_noun_to_semantic_category(noun_text, semantic_category, sentence_id)
        await idea_graph.link_noun_to_sentence(noun_text, sentence_id, plurality=plurality)
        return noun_record, semantic_category_record


def get_noun_with_sentence_blob(noun: str, sentence: str):
    return f"""
    Noun: {noun}
    Text: {sentence}
    """.strip()


async def modifier_peeler(noun_text: str, plurality: str,
                          sentence: str, sentence_id: int, semantic_categories: list[str],
                          semantic_category: str = None, limit_additional_recursion: bool = False):
    noun_words = noun_text.split()
    if len(noun_words) > 1:
        # For phrases, see if there are modifier words that wrap root words.
        modifiers = await run_in_threadpool(get_noun_modifier_classifier(semantic_categories).classify,
                                            get_noun_with_sentence_blob(noun_text, sentence),
                                            review=False, review_json=None)
        logger.info(f"modifiers: {modifiers}")
        single_semantic_category_classifier = get_single_noun_classifier(semantic_categories)
        for modifier in modifiers["modifiers"]:
            logger.info(f"modifier: {modifier}")
            if modifier["noun"] == noun_text:

                logger.info(f"modified noun('{modifier["noun"]}') == i noun('{noun_text}'), adding multi-word noun")
                noun_record, semantic_category_record = await add_noun(
                    modifier["noun"], semantic_category, plurality, sentence_id)
                await idea_graph.add_modifier(modifier["modifier"])
                await idea_graph.link_noun_to_modifier(
                    noun_record[0]["noun"]["text"], modifier["modifier"], sentence_id)

            elif modifier["noun"] not in noun_text:
                # If the thing being modified isn't in noun_text words, we're likely looking at some
                # noun/modifier pair the classifier found that is related to noun_text, but possibly something
                # totally different.

                # Do a type check
                new_semantic_category = await run_in_threadpool(
                    single_semantic_category_classifier.classify,
                    get_noun_with_sentence_blob(modifier["noun"], sentence),
                    review=False, review_json=None)
                logger.info(f"semantic_category: {semantic_category}, new_semantic_category: {new_semantic_category}")
                new_noun_words = modifier["noun"].split()
                if len(new_noun_words) > 1:

                    # If the new thing being modified is also multi-word, to avoid possibly recursing forever, we
                    # recurse only once.
                    if limit_additional_recursion:
                        logger.info(f"'{modifier["noun"]}' is not in {noun_text}, but not doing more")
                        continue
                    else:
                        task = asyncio.create_task(
                            modifier_peeler(modifier["noun"], new_semantic_category["plurality"],
                                            sentence, sentence_id, semantic_categories,
                                            semantic_category=new_semantic_category["semanticCategory"],
                                            limit_additional_recursion=True))
                else:
                    noun_record, semantic_category_record = await add_noun(
                        modifier["noun"], new_semantic_category["semanticCategory"], new_semantic_category["plurality"], sentence_id)
                    await idea_graph.add_modifier(modifier["modifier"])
                    await idea_graph.link_noun_to_modifier(
                        noun_record[0]["noun"]["text"], modifier["modifier"], sentence_id)

            else:

                logger.info(f"'{modifier["noun"]}' is in {noun_text}")
                root_words = modifier["noun"].split()
                if len(root_words) > 1:
                    # If we're still dealing with a multi-word string, dig deeper.
                    logger.info(f"'{modifier["noun"]}' has more than one word")
                    task = asyncio.create_task(
                        modifier_peeler(modifier["noun"], plurality,
                                        sentence, sentence_id, semantic_categories,
                                        semantic_category=semantic_category))
                else:
                    noun_record, semantic_category_record = await add_noun(
                        modifier["noun"], semantic_category, plurality, sentence_id)
                    await idea_graph.add_modifier(modifier["modifier"])
                    await idea_graph.link_noun_to_modifier(
                        noun_record[0]["noun"]["text"], modifier["modifier"], sentence_id)
    else:
        await add_noun(noun_text, semantic_category, plurality, sentence_id)


sentence_controller = SentenceController()


async def main():
    await sentence_controller.on_periodic_run()


if __name__ == "__main__":
    setup_logging()
    while True:
        asyncio.run(main())
