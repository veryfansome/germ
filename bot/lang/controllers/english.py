from bs4 import BeautifulSoup
from copy import deepcopy
from starlette.concurrency import run_in_threadpool
from typing import Any
import asyncio
import inflect

from bot.graph.idea import CodeBlockMergeEventHandler, ParagraphMergeEventHandler, SentenceMergeEventHandler, idea_graph
from bot.lang.parsers import get_html_parser
from bot.lang.classifiers import (
    ADJECTIVE_POS_TAGS, ADVERB_POS_TAGS, NOUN_POS_TAGS, PRONOUN_POS_TAGS, VERB_POS_TAGS,
    get_flair_pos_tags, split_to_sentences)
from observability.logging import logging, setup_logging

inflect_engine = inflect.engine()
logger = logging.getLogger(__name__)


async def strip_html_elements(soup: BeautifulSoup, tag: str = None):
    text_elements = []
    html_elements = {}
    for idx, element in enumerate(soup.find_all() if tag is None else soup.find(tag)):
        if element.name is None:  # This is just text
            text_elements.append(element.string)
        else:
            text_elements.append("[oops]")  # Placeholder
            html_elements[idx] = element
    # Iterate through elements to write over the placeholders
    element_artifacts = list(html_elements.values())
    for idx, element in html_elements.items():
        if element.find():  # Has inner elements
            logger.info(f"found <{element.name}>, with inner elements: {element}")
            inner_string, inner_artifacts = await strip_html_elements(element)
            if inner_artifacts:
                element_artifacts += inner_artifacts

            if element.name == "a":
                logger.info(f"href: {element['href']}")
                # TODO: - Replace inner_string
                #       - Use some alphabetical hash based on the domain, with the first letter upper cased so that the
                #         word would be seen as a proper noun by the POS tagger.
                #       - Use regex to capture [\s]*[\w]+\.[\w]+[^\s]*
                #       - Capture protocol, path, and params if any
                #       - Do a domain resolution against captured domain string to see
                #       - But it could also be a local link so we'll have to be careful.
                pass

            text_elements[idx] = inner_string
        else:  # Doesn't have inner elements
            logger.info(f"stripped <{element.name}>, kept inner string: {element.string}")
            if element.name == "a":
                # TODO: Same as above
                pass
            text_elements[idx] = element.string if element.string else ""
    return ''.join(text_elements), element_artifacts


class EnglishController(CodeBlockMergeEventHandler, ParagraphMergeEventHandler, SentenceMergeEventHandler):
    def __init__(self, interval_seconds: int = 30):
        self.interval_seconds = interval_seconds
        self.sentence_merge_artifacts = []

    async def on_code_block_merge(self, code_block: str, code_block_id: int):
        logger.info(f"on_code_block_merge: code_block_id={code_block_id}, {code_block}")

    async def on_periodic_run(self):
        logger.info(f"on_periodic_run: {self.sentence_merge_artifacts}")

    async def on_paragraph_merge(self, paragraph: str, paragraph_id: int):
        logger.info(f"on_paragraph_merge: paragraph_id={paragraph_id}, {paragraph}")

        paragraph_soup = await run_in_threadpool(get_html_parser, f"<p>{paragraph}</p>")
        paragraph_text, paragraph_elements = await strip_html_elements(paragraph_soup, "p")
        logger.info(f"paragraph_text: {paragraph_text}")
        for element in paragraph_elements:
            logger.info(f"paragraph_element: {element}")
            # TODO: If hyperlink:
            #       - Merge a Domain.
            #       - Hash from above and the actual domain name should be attributes of this node.
            #       - Add the protocol, path, and parameters on the vertexes to the paragraph
            #       - Add inner text as a vertex attribute

        if not paragraph_text:
            return

       # previous_sentence_id = None
       # for sentence in split_to_sentences(paragraph):
       #     _, sentence_id, _ = await idea_graph.add_sentence(sentence)
       #     _ = asyncio.create_task(idea_graph.link_paragraph_to_sentence(paragraph_id, sentence_id))
       #     if previous_sentence_id is not None:
       #         _ = asyncio.create_task(
       #             idea_graph.link_sentence_to_previous_sentence(previous_sentence_id, sentence_id))

    async def on_sentence_merge(self, sentence: str, sentence_id: int, sentence_parameters):
        logger.info(f"on_sentence_merge: sentence_id={sentence_id}, {sentence_parameters}, {sentence}")

        # Get POS in the background
        pos_task = asyncio.create_task(run_in_threadpool(get_flair_pos_tags, sentence))
        logger.info(f"pos_tags: {await pos_task}")

        return  # TODO: Remove me
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
        logger.info(f"pos_tags: {artifacts['pos_tags']}")

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
                # TODO: this is really bad - replace this with proper quote and HTML tag processing
                nouns.append((f"{artifacts['pos_tags'][idx-1][0]}{text}", tag))
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

        code_snippets = [word for word in artifacts["pos_tags"] if word[1] == "CODE"]
        logger.info(f"code_snippets: {code_snippets}")

        # TODO: since we're dealing with simple language tasks, try using a flair or transformers LLM
        #       that we can tune.

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


english_controller = EnglishController()


async def main():
    await english_controller.on_periodic_run()


if __name__ == "__main__":
    setup_logging()
    while True:
        asyncio.run(main())
