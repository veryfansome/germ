import asyncio
import inflect
from starlette.concurrency import run_in_threadpool

from bot.graph.idea import SentenceMergeEventHandler, idea_graph
from bot.lang.classifiers import (get_entity_type_classifier, get_entity_modifier_classifier,
                                  get_single_entity_type_classifier)
from observability.logging import logging, setup_logging

inflect_engine = inflect.engine()
logger = logging.getLogger(__name__)


class EntityController(SentenceMergeEventHandler):
    def __init__(self, interval_seconds: int = 30):
        self.interval_seconds = interval_seconds
        self.sentence_merge_artifacts = []

    async def on_periodic_run(self):
        logger.info(f"on_periodic_run: {self.sentence_merge_artifacts}")

    async def on_sentence_merge(self, sentence: str, node_type: str, sentence_id: int, sentence_parameters):
        logger.info(f"on_sentence_merge: {node_type}, sentence_id={sentence_id}, {sentence_parameters}, {sentence}")

        # TODO: is sorting by connections enough or should we also care about how many are recent?
        # Query entity types from graph, sorted by most connections
        entity_types = [t["entity_type"]["text"] for t in await idea_graph.get_entity_type_desc_by_connections()]
        logger.debug(f"entity_types: {entity_types}")
        artifacts = {
            "entity_types": entity_types,
            "node_type": node_type,
            # Used entity types from query as hints to classify sentence
            "openai_entities": await run_in_threadpool(get_entity_type_classifier(entity_types).classify,
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
            for entity in artifacts["openai_entities"]["entities"]:
                try:
                    logger.info(f"entity_name: {entity["entity"]}")
                    tasks_to_await.append(asyncio.create_task(modifier_peeler(
                        entity["entity"], entity["plurality"],
                        sentence, node_type, sentence_id, entity_types,
                        entity_type=entity["entity_type"])))
                except Exception as e:
                    logger.warning(f"failed to add entity: {entity}", e)
        self.sentence_merge_artifacts.append(artifacts)


async def add_entity(entity_name: str, entity_type: str, plurality: str,
                     node_type: str, sentence_id: int):

    if plurality == "plural":
        singular_form = inflect_engine.singular_noun(entity_name)
        if not singular_form:
            logger.warning(f"inflect and classifier don't agree: {entity_name}, classifier {plurality}")
        else:
            # Add and link singular form
            entity_record, entity_type_record = await asyncio.gather(*[
                idea_graph.add_entity(singular_form),
                idea_graph.add_entity_type(entity_type)])
            await idea_graph.link_entity_to_entity_type(singular_form, entity_type, sentence_id)
            await idea_graph.link_entity_to_sentence(singular_form, sentence_id, node_type, plurality=plurality)
            return entity_record, entity_type_record
    else:
        entity_record, entity_type_record = await asyncio.gather(*[
            idea_graph.add_entity(entity_name),
            idea_graph.add_entity_type(entity_type)])
        await idea_graph.link_entity_to_entity_type(entity_name, entity_type, sentence_id)
        await idea_graph.link_entity_to_sentence(entity_name, sentence_id, node_type, plurality=plurality)
        return entity_record, entity_type_record


def get_entity_with_sentence_blob(entity: str, sentence: str):
    return f"""
    Entity: {entity}
    Text: {sentence}
    """.strip()


async def modifier_peeler(entity_name: str, plurality: str,
                          sentence: str, node_type: str, sentence_id: int, entity_types: list[str],
                          entity_type: str = None, limit_additional_recursion: bool = False):
    entity_words = entity_name.split()
    if len(entity_words) > 1:
        # For multi-word entities, see if there are modifier words that wrap root words.
        modifiers = await run_in_threadpool(get_entity_modifier_classifier(entity_types).classify,
                                            get_entity_with_sentence_blob(entity_name, sentence),
                                            review=False, review_json=None)
        logger.info(f"modifiers: {modifiers}")
        single_entity_type_classifier = get_single_entity_type_classifier(entity_types)
        for modifier in modifiers["modifiers"]:
            logger.info(f"modifier: {modifier}")
            if modifier["modifies"] == entity_name:

                logger.info(f"modifies('{modifier["modifies"]}') == entity('{entity_name}'), adding multi-word entity")
                entity_record, entity_type_record = await add_entity(
                    modifier["modifies"], entity_type, plurality, node_type, sentence_id)
                await idea_graph.add_modifier(modifier["modifier"])
                await idea_graph.link_entity_to_modifier(
                    entity_record[0]["entity"]["text"], modifier["modifier"], sentence_id)

            elif modifier["modifies"] not in entity_name:
                # If the thing being modified isn't in entity_name words, we're likely looking at some
                # modifies/modifier pair the classifier found that is related to entity_name, but possibly something
                # totally different.

                # Do a type check
                new_entity_type = await run_in_threadpool(
                    single_entity_type_classifier.classify,
                    get_entity_with_sentence_blob(modifier["modifies"], sentence),
                    review=False, review_json=None)
                logger.info(f"entity_type: {entity_type}, new_entity_type: {new_entity_type}")
                new_entity_words = modifier["modifies"].split()
                if len(new_entity_words) > 1:

                    # If the new thing being modified is also multi-word, to avoid possibly recursing forever, we
                    # recurse only once.
                    if limit_additional_recursion:
                        logger.info(f"'{modifier["modifies"]}' is not in {entity_name}, but not doing more")
                        continue
                    else:
                        task = asyncio.create_task(
                            modifier_peeler(modifier["modifies"], new_entity_type["plurality"],
                                            sentence, node_type, sentence_id, entity_types,
                                            entity_type=new_entity_type["entity_type"],
                                            limit_additional_recursion=True))
                else:
                    entity_record, entity_type_record = await add_entity(
                        modifier["modifies"], new_entity_type["entity_type"], new_entity_type["plurality"],
                        node_type, sentence_id)
                    await idea_graph.add_modifier(modifier["modifier"])
                    await idea_graph.link_entity_to_modifier(
                        entity_record[0]["entity"]["text"], modifier["modifier"], sentence_id)

            else:

                logger.info(f"'{modifier["modifies"]}' is in {entity_name}")
                root_words = modifier["modifies"].split()
                if len(root_words) > 1:
                    # If we're still dealing with a multi-word string, dig deeper.
                    logger.info(f"'{modifier["modifies"]}' has more than one word")
                    task = asyncio.create_task(
                        modifier_peeler(modifier["modifies"], plurality,
                                        sentence, node_type, sentence_id, entity_types,
                                        entity_type=entity_type))
                else:
                    entity_record, entity_type_record = await add_entity(
                        modifier["modifies"], entity_type, plurality, node_type, sentence_id)
                    await idea_graph.add_modifier(modifier["modifier"])
                    await idea_graph.link_entity_to_modifier(
                        entity_record[0]["entity"]["text"], modifier["modifier"], sentence_id)
    else:
        await add_entity(entity_name, entity_type, plurality, node_type, sentence_id)


entity_controller = EntityController()


async def main():
    await entity_controller.on_periodic_run()


if __name__ == "__main__":
    setup_logging()
    while True:
        asyncio.run(main())
