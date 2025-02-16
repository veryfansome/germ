from starlette.concurrency import run_in_threadpool

from bot.graph.control_plane import CodeBlockMergeEventHandler, ControlPlane, ParagraphMergeEventHandler, SentenceMergeEventHandler
from bot.lang.parsers import get_html_soup, strip_html_elements
from observability.logging import logging

logger = logging.getLogger(__name__)


class EnglishController(CodeBlockMergeEventHandler, ParagraphMergeEventHandler, SentenceMergeEventHandler):
    def __init__(self, control_plane: ControlPlane, interval_seconds: int = 30):
        self.control_plane = control_plane
        self.interval_seconds = interval_seconds
        self.sentence_merge_artifacts = []

    async def on_code_block_merge(self, code_block: str, code_block_id: int):
        logger.info(f"on_code_block_merge: code_block_id={code_block_id}, {code_block}")

    async def on_periodic_run(self):
        logger.info(f"on_periodic_run: {self.sentence_merge_artifacts}")

    async def on_paragraph_merge(self, paragraph: str, paragraph_id: int):
        logger.info(f"on_paragraph_merge: paragraph_id={paragraph_id}, {paragraph}")

        paragraph_soup = await run_in_threadpool(get_html_soup, f"<p>{paragraph}</p>")
        paragraph_text, paragraph_elements = await strip_html_elements(paragraph_soup, "p")
        for element in paragraph_elements:
            logger.info(f"paragraph_element: {element}")
            # TODO: If hyperlink:
            #       - Merge a domain.
            #       - Link the domain to the domain's proper noun, which will automatically connect it to the sentence.
            #       - domain name should be attributes of this domain node.
            #       - Add the protocol, path, and parameters on the vertexes to the paragraph
            #       - Add inner text as a vertex attribute

        if not paragraph_text:
            return

        logger.info(f"paragraph_text: {paragraph_text}")
        previous_sentence_id = None
        #for sentence in split_to_sentences(paragraph_text):
        #    _, sentence_id, _ = await self.control_plane.add_sentence(sentence)
        #    _ = asyncio.create_task(self.control_plane.link_paragraph_to_sentence(paragraph_id, sentence_id))
        #    if previous_sentence_id is not None:
        #        _ = asyncio.create_task(
        #            self.control_plane.link_sentence_to_previous_sentence(previous_sentence_id, sentence_id))

    async def on_sentence_merge(self, sentence: str, sentence_id: int, sentence_parameters):
        logger.info(f"on_sentence_merge: sentence_id={sentence_id}, {sentence_parameters}, {sentence}")
