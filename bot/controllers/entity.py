import asyncio
import inflect

from bot.graph.idea import SentenceMergeEventHandler, idea_graph
from bot.lang.classifiers import get_entity_type_classifier
from observability.logging import logging, setup_logging

inflect_engine = inflect.engine()
logger = logging.getLogger(__name__)


class EntityController(SentenceMergeEventHandler):
    def __init__(self, interval_seconds: int = 30):
        self.interval_seconds = interval_seconds
        self.merge_events = []

    async def on_periodic_run(self):
        logger.info(f"on_periodic_run: {self.merge_events}")

    async def on_sentence_merge(self, sentence: str, node_type: str, sentence_id: int, sentence_parameters):
        logger.info(f"on_sentence_merge: {node_type}, sentence_id={sentence_id}, {sentence_parameters}, {sentence}")

        # Query entity types from graph
        # TODO: is sorting by connections enough or should we also care about how many are recent?
        entity_types = [t["entity_type"]["text"] for t in await idea_graph.get_entity_type_desc_by_connections()]
        logger.debug(f"entity_types: {entity_types}")

        # Used queried entity types to classify sentence
        openai_entities = get_entity_type_classifier(entity_types).classify(sentence)
        logger.debug(f"openai_entities: {openai_entities}")
        if "entities" not in openai_entities:
            logger.warning(f"expected `entities` field is missing in: {openai_entities}")
        else:
            for entity in openai_entities["entities"]:
                try:
                    words = entity["entity"].split()

                    await asyncio.gather(*[
                            idea_graph.add_entity(entity["entity"]),
                            idea_graph.add_entity_type(entity["entity_type"])])
                    await idea_graph.link_entity_to_entity_type(entity["entity"], entity["entity_type"])
                    await idea_graph.link_entity_to_sentence(entity["entity"], sentence_id, node_type)
                except Exception as e:
                    logger.warning(f"failed to add entity: {entity}", e)
        self.merge_events.append({
            "openai_entities": openai_entities,
            "node_type": node_type,
            "sentence": sentence,
            "sentence_id": sentence_id,
            "sentence_parameters": sentence_parameters,
        })


entity_controller = EntityController()


async def main():
    await entity_controller.on_periodic_run()


if __name__ == "__main__":
    setup_logging()
    while True:
        asyncio.run(main())
