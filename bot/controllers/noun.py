import asyncio
import inflect
from starlette.concurrency import run_in_threadpool

from bot.graph.idea import SentenceMergeEventHandler, idea_graph
from bot.lang.classifiers import (get_semantic_categories_classifier, get_noun_modifier_classifier,
                                  get_single_semantic_category_classifier)
from observability.logging import logging, setup_logging

inflect_engine = inflect.engine()
logger = logging.getLogger(__name__)


class NounController(SentenceMergeEventHandler):
    def __init__(self, interval_seconds: int = 30):
        self.interval_seconds = interval_seconds
        self.sentence_merge_artifacts = []

    async def on_periodic_run(self):
        logger.info(f"on_periodic_run: {self.sentence_merge_artifacts}")

    async def on_sentence_merge(self, sentence: str, sentence_id: int, sentence_parameters):
        logger.info(f"on_sentence_merge: sentence_id={sentence_id}, {sentence_parameters}, {sentence}")

        # TODO: is sorting by connections enough or should we also care about how many are recent?
        # Query noun types from graph, sorted by most connections
        semantic_categories = [t["semantic_category"]["text"] for t in await idea_graph.get_semantic_category_desc_by_connections()]
        logger.debug(f"semantic_categories: {semantic_categories}")
        artifacts = {
            "semantic_categories": semantic_categories,
            # Used noun types from query as hints to classify sentence
            "openai_entities": await run_in_threadpool(get_semantic_categories_classifier(semantic_categories).classify,
                                                       sentence, review=False, review_json=None),
            "sentence": sentence,
            "sentence_id": sentence_id,
            "sentence_parameters": sentence_parameters,
        }
        logger.info(f"openai_entities: {artifacts["openai_entities"]}")
        tasks_to_await = []
        if "entities" not in artifacts["openai_entities"]:
            logger.warning(f"expected `entities` field is missing in: {artifacts["openai_entities"]}")
        else:
            for noun in artifacts["openai_entities"]["entities"]:
                try:
                    logger.info(f"noun_name: {noun["noun"]}")
                    tasks_to_await.append(asyncio.create_task(modifier_peeler(
                        noun["noun"], noun["plurality"], sentence, sentence_id, semantic_categories,
                        semantic_category=noun["semantic_category"])))
                except Exception as e:
                    logger.warning(f"failed to add noun: {noun}", e)
        self.sentence_merge_artifacts.append(artifacts)


async def add_noun(noun_name: str, semantic_category: str, plurality: str, sentence_id: int):
    if plurality == "plural":
        singular_form = inflect_engine.singular_noun(noun_name)
        if not singular_form:
            logger.warning(f"inflect and classifier don't agree: {noun_name}, classifier {plurality}")
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
            idea_graph.add_noun(noun_name),
            idea_graph.add_semantic_category(semantic_category)])
        await idea_graph.link_noun_to_semantic_category(noun_name, semantic_category, sentence_id)
        await idea_graph.link_noun_to_sentence(noun_name, sentence_id, plurality=plurality)
        return noun_record, semantic_category_record


def get_noun_with_sentence_blob(noun: str, sentence: str):
    return f"""
    Noun: {noun}
    Text: {sentence}
    """.strip()


async def modifier_peeler(noun_name: str, plurality: str,
                          sentence: str, sentence_id: int, semantic_categories: list[str],
                          semantic_category: str = None, limit_additional_recursion: bool = False):
    noun_words = noun_name.split()
    if len(noun_words) > 1:
        # For phrases, see if there are modifier words that wrap root words.
        modifiers = await run_in_threadpool(get_noun_modifier_classifier(semantic_categories).classify,
                                            get_noun_with_sentence_blob(noun_name, sentence),
                                            review=False, review_json=None)
        logger.info(f"modifiers: {modifiers}")
        single_semantic_category_classifier = get_single_semantic_category_classifier(semantic_categories)
        for modifier in modifiers["modifiers"]:
            logger.info(f"modifier: {modifier}")
            if modifier["modifies"] == noun_name:

                logger.info(f"modifies('{modifier["modifies"]}') == noun('{noun_name}'), adding multi-word noun")
                noun_record, semantic_category_record = await add_noun(
                    modifier["modifies"], semantic_category, plurality, sentence_id)
                await idea_graph.add_modifier(modifier["modifier"])
                await idea_graph.link_noun_to_modifier(
                    noun_record[0]["noun"]["text"], modifier["modifier"], sentence_id)

            elif modifier["modifies"] not in noun_name:
                # If the thing being modified isn't in noun_name words, we're likely looking at some
                # modifies/modifier pair the classifier found that is related to noun_name, but possibly something
                # totally different.

                # Do a type check
                new_semantic_category = await run_in_threadpool(
                    single_semantic_category_classifier.classify,
                    get_noun_with_sentence_blob(modifier["modifies"], sentence),
                    review=False, review_json=None)
                logger.info(f"semantic_category: {semantic_category}, new_semantic_category: {new_semantic_category}")
                new_noun_words = modifier["modifies"].split()
                if len(new_noun_words) > 1:

                    # If the new thing being modified is also multi-word, to avoid possibly recursing forever, we
                    # recurse only once.
                    if limit_additional_recursion:
                        logger.info(f"'{modifier["modifies"]}' is not in {noun_name}, but not doing more")
                        continue
                    else:
                        task = asyncio.create_task(
                            modifier_peeler(modifier["modifies"], new_semantic_category["plurality"],
                                            sentence, sentence_id, semantic_categories,
                                            semantic_category=new_semantic_category["semantic_category"],
                                            limit_additional_recursion=True))
                else:
                    noun_record, semantic_category_record = await add_noun(
                        modifier["modifies"], new_semantic_category["semantic_category"], new_semantic_category["plurality"], sentence_id)
                    await idea_graph.add_modifier(modifier["modifier"])
                    await idea_graph.link_noun_to_modifier(
                        noun_record[0]["noun"]["text"], modifier["modifier"], sentence_id)

            else:

                logger.info(f"'{modifier["modifies"]}' is in {noun_name}")
                root_words = modifier["modifies"].split()
                if len(root_words) > 1:
                    # If we're still dealing with a multi-word string, dig deeper.
                    logger.info(f"'{modifier["modifies"]}' has more than one word")
                    task = asyncio.create_task(
                        modifier_peeler(modifier["modifies"], plurality,
                                        sentence, sentence_id, semantic_categories,
                                        semantic_category=semantic_category))
                else:
                    noun_record, semantic_category_record = await add_noun(
                        modifier["modifies"], semantic_category, plurality, sentence_id)
                    await idea_graph.add_modifier(modifier["modifier"])
                    await idea_graph.link_noun_to_modifier(
                        noun_record[0]["noun"]["text"], modifier["modifier"], sentence_id)
    else:
        await add_noun(noun_name, semantic_category, plurality, sentence_id)


noun_controller = NounController()


async def main():
    await noun_controller.on_periodic_run()


if __name__ == "__main__":
    setup_logging()
    while True:
        asyncio.run(main())
