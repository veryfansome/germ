from abc import ABC, abstractmethod
from asyncio import Task
from datetime import datetime, timedelta, timezone
from sqlalchemy.future import select as sql_select
from starlette.concurrency import run_in_threadpool
import asyncio
import signal
import uuid

from bot.graph.noun import default_semantic_categories
from bot.lang.classifiers import get_sentence_classifier, split_to_sentences
from db.neo4j import AsyncNeo4jDriver
from db.models import AsyncSessionLocal, CodeBlock, Paragraph, Sentence
from observability.logging import logging
from settings.germ_settings import UUID5_NS

logger = logging.getLogger(__name__)


class CodeBlockMergeEventHandler(ABC):
    @abstractmethod
    async def on_code_block_merge(self, code_block: str, code_block_id: int):
        pass


class ParagraphMergeEventHandler(ABC):
    @abstractmethod
    async def on_paragraph_merge(self, paragraph: str, paragraph_id: int):
        pass


class SentenceMergeEventHandler(ABC):
    @abstractmethod
    async def on_sentence_merge(self, sentence: str, sentence_id: int, sentence_parameters):
        pass


class IdeaGraph:
    def __init__(self, driver: AsyncNeo4jDriver):
        self.driver = driver
        self.code_block_merge_event_handlers: list[CodeBlockMergeEventHandler] = []
        self.paragraph_merge_event_handlers: list[ParagraphMergeEventHandler] = []
        self.sentence_merge_event_handlers: list[SentenceMergeEventHandler] = []

    async def add_chat_request(self, chat_request_received_id: int):
        time_occurred = round_time_now_down_to_nearst_interval()
        results = await self.driver.query("""
        MERGE (chatRequest:ChatRequest {chat_request_received_id: $chat_request_received_id, time_occurred: $time_occurred})
        RETURN chatRequest
        """, {
            "chat_request_received_id": chat_request_received_id, "time_occurred": time_occurred,
        })
        logger.info(f"request: {results}")
        return results, time_occurred

    async def add_chat_response(self, chat_response_sent_id: int):
        time_occurred = round_time_now_down_to_nearst_interval()
        results = await self.driver.query("""
        MERGE (chatResponse:ChatResponse {chat_response_sent_id: $chat_response_sent_id, time_occurred: $time_occurred})
        RETURN chatResponse
        """, {
            "chat_response_sent_id": chat_response_sent_id, "time_occurred": time_occurred,
        })
        logger.info(f"response: {results}")
        return results, time_occurred

    async def add_chat_session(self, chat_session_id: int):
        results = await self.driver.query("""
        MERGE (chatSession:ChatSession {chat_session_id: $chat_session_id, time_started: $time_started})
        RETURN chatSession
        """, {
            "chat_session_id": chat_session_id, "time_started": round_time_now_down_to_nearst_interval()
        })
        logger.info(f"session: {results}")
        return results

    async def add_code_block(self, code_block: str, language: str = None):
        code_block = code_block.strip()
        # Generate signature for lookup in PostgreSQL
        code_block_signature = uuid.uuid5(UUID5_NS, code_block)
        async with (AsyncSessionLocal() as rdb_session):
            async with rdb_session.begin():
                code_block_select_stmt = sql_select(CodeBlock).where(CodeBlock.code_block_signature == code_block_signature)
                code_block_select_result = await rdb_session.execute(code_block_select_stmt)
                rdb_record = code_block_select_result.scalars().first()

                # Create record if not found
                if not rdb_record:
                    rdb_record = CodeBlock(text=code_block, code_block_signature=code_block_signature)
                    rdb_session.add(rdb_record)
                    await rdb_session.commit()

        graph_results = await self.driver.query("""
        MERGE (code_block:CodeBlock {text: $text, language: $language, code_block_id: $code_block_id})
        RETURN code_block
        """, {
            "text": code_block,
            "language": language,
            "code_block_id": rdb_record.code_block_id,
        })
        logger.info(f"code block: {graph_results}")
        async_tasks = []
        for handler in self.code_block_merge_event_handlers:
            async_tasks.append(asyncio.create_task(
                handler.on_code_block_merge(code_block, rdb_record.code_block_id)))
        return graph_results, rdb_record.code_block_id, async_tasks

    def add_code_block_merge_event_handler(self, handler: CodeBlockMergeEventHandler):
        self.code_block_merge_event_handlers.append(handler)

    async def add_default_semantic_categories(self) -> list[Task]:
        async_tasks = []
        for category in default_semantic_categories:
            async_tasks.append(asyncio.create_task(self.add_semantic_category(category)))
        return async_tasks

    async def add_document(self, name: str, body: str):
        """
        An article, a poem, or even a chapter of a book.

        :param name:
        :param body:
        :return:
        """
        document_record = await self.driver.query(
            "MERGE (d:Document {name: $name}) RETURN d", {"name": name})
        logger.info(f"document: {document_record}")

        last_sentence_id = None
        for body_sentence in split_to_sentences(body):
            _, this_sentence_id, async_tasks = await self.add_sentence(body_sentence)

            sentence_do_document_link_record = await self.driver.query(
                "MATCH (d:Document {name: $document}), (s:Sentence {sentence_id: $sentence_id}) "
                "MERGE (d)-[r:CONTAINS]->(s) "
                "RETURN r", {
                    "document": name, "sentence_id": this_sentence_id
                })
            if last_sentence_id is not None:
                sentence_to_sentence_link = await self.driver.query(
                    "MATCH (l:Sentence {sentence_id: $last_id}), (t:Sentence {sentence_id: $this_id}) "
                    "MERGE (l)-[r:PRECEDES]->(t) "
                    "RETURN r", {
                        "last_id": last_sentence_id, "this_id": this_sentence_id,
                    })
                logger.info(f"sentence/sentence link: {sentence_to_sentence_link}")

            logger.info(f"sentence link: {sentence_do_document_link_record}")
            last_sentence_id = this_sentence_id
        return document_record

    async def add_header(self, header: str):
        header = header.strip()
        results = await self.driver.query("""
        MERGE (header:Header {text: $text})
        RETURN header
        """, {
            "text": header
        })
        logger.info(f"header: {results}")
        return results

    async def add_noun_form(self, noun: str, form: str):
        results = await self.driver.query("""
        MERGE (noun:Noun {text: $noun})
        WITH noun
        UNWIND (COALESCE(noun.forms, []) + [$form]) AS form
        WITH noun, COLLECT(DISTINCT form) AS uniqueForms
        SET noun.forms = uniqueForms
        RETURN noun
        """, {
            "noun": noun, "form": form
        })
        logger.info(f"noun: {results}")
        return results

    async def add_noun_plural_form(self, noun: str, plural: str):
        noun_record = await self.driver.query("""
        MATCH (noun:Noun {text: $noun})
        SET noun.plural_forms = coalesce(noun.plural_forms, []) + [$plural]
        RETURN noun
        """, {"noun": noun, "plural": plural})
        logger.info(f"noun: {noun_record}")
        return noun_record

    async def add_paragraph(self, paragraph: str):
        paragraph = paragraph.strip()
        # Generate signature for lookup in PostgreSQL
        paragraph_signature = uuid.uuid5(UUID5_NS, paragraph)
        async with (AsyncSessionLocal() as rdb_session):
            async with rdb_session.begin():
                paragraph_select_stmt = sql_select(Paragraph).where(Paragraph.paragraph_signature == paragraph_signature)
                paragraph_select_result = await rdb_session.execute(paragraph_select_stmt)
                rdb_record = paragraph_select_result.scalars().first()

                # Create record if not found
                if not rdb_record:
                    rdb_record = Paragraph(text=paragraph, paragraph_signature=paragraph_signature)
                    rdb_session.add(rdb_record)
                    await rdb_session.commit()

        graph_results = await self.driver.query("""
        MERGE (paragraph:Paragraph {text: $text, paragraph_id: $paragraph_id})
        RETURN paragraph
        """, {
            "text": paragraph,
            "paragraph_id": rdb_record.paragraph_id,
        })
        logger.info(f"paragraph: {graph_results}")
        async_tasks = []
        for handler in self.paragraph_merge_event_handlers:
            async_tasks.append(asyncio.create_task(
                handler.on_paragraph_merge(paragraph, rdb_record.paragraph_id)))
        return graph_results, rdb_record.paragraph_id, async_tasks

    def add_paragraph_merge_event_handler(self, handler: ParagraphMergeEventHandler):
        self.paragraph_merge_event_handlers.append(handler)

    async def add_part_of_speech(self, tag: str):
        results = await self.driver.query("""
        MERGE (pos:PartOfSpeech {tag: $tag})
        RETURN pos
        """, {
            "tag": tag
        })
        logger.info(f"pos tag: {results}")
        return results

    async def add_proper_noun_form(self, proper_noun: str, form: str):
        results = await self.driver.query("""
        MERGE (properNoun:ProperNoun {text: $properNoun})
        WITH properNoun
        UNWIND (COALESCE(properNoun.forms, []) + [$form]) AS form
        WITH properNoun, COLLECT(DISTINCT form) AS uniqueForms
        SET properNoun.forms = uniqueForms
        RETURN properNoun
        """, {
            "properNoun": proper_noun, "form": form
        })
        logger.info(f"proper noun: {results}")
        return results

    async def add_semantic_category(self, semantic_category: str):
        semantic_category_record = await self.driver.query(
            "MERGE (semanticCategory:SemanticCategory {text: $semanticCategory}) RETURN semanticCategory",
            {"semanticCategory": semantic_category})
        logger.info(f"semantic_category: {semantic_category_record}")
        return semantic_category_record

    async def add_sentence(self, sentence: str):
        sentence = sentence.strip()
        # Generate signature for lookup in PostgreSQL
        sentence_signature = uuid.uuid5(UUID5_NS, sentence)
        async with (AsyncSessionLocal() as rdb_session):
            async with rdb_session.begin():
                sentence_select_stmt = sql_select(Sentence).where(Sentence.sentence_signature == sentence_signature)
                sentence_select_result = await rdb_session.execute(sentence_select_stmt)
                rdb_record = sentence_select_result.scalars().first()

                # Create record if not found
                if not rdb_record:
                    rdb_record = Sentence(text=sentence, sentence_signature=sentence_signature)
                    rdb_session.add(rdb_record)
                    await rdb_session.commit()

            # Get sentence type if not set
            sentence_classifier = get_sentence_classifier()
            if (rdb_record.sentence_openai_parameters is None
                    or is_stale_or_over_fetch_count_threshold(
                        rdb_record.sentence_openai_parameters_time_changed,
                        rdb_record.sentence_openai_parameters_fetch_count,
                        hours=24
                    )):
                openai_parameters = await run_in_threadpool(sentence_classifier.classify, sentence)
                async with rdb_session.begin():
                    await rdb_session.refresh(rdb_record)
                    rdb_record.sentence_openai_parameters = openai_parameters
                    rdb_record.sentence_openai_parameters_time_changed = utc_now()
                    rdb_record.sentence_openai_parameters_fetch_count += 1
                    await rdb_session.commit()
            logger.debug(f"sentence openai_parameters: {rdb_record.sentence_openai_parameters}")

            # Merge sentence node
            parameter_spec = sentence_classifier.tool_properties_spec
            graph_results = await self.driver.query(
                ("MERGE (sentence:Sentence "
                 + "{text: $text, sentence_id: $sentence_id, "
                 + ", ".join([f"{k}: ${k}" for k in parameter_spec.keys()])
                 + "}) RETURN sentence"),
                {
                    "text": sentence,
                    "sentence_id": rdb_record.sentence_id,
                    **{k: rdb_record.sentence_openai_parameters[k] for k in parameter_spec.keys()},
                })
            logger.info(f"sentence node: {graph_results}")
        async_tasks = []
        for handler in self.sentence_merge_event_handlers:
            async_tasks.append(asyncio.create_task(
                handler.on_sentence_merge(
                    sentence, rdb_record.sentence_id, rdb_record.sentence_openai_parameters)))
        return graph_results, rdb_record.sentence_id, async_tasks

    def add_sentence_merge_event_handler(self, handler: SentenceMergeEventHandler):
        self.sentence_merge_event_handlers.append(handler)

    async def add_time(self):
        """
        Time nodes are created in set intervals and changed together to match the directionality of real time.
        The idea, here, is that every node created should be linked to its proper chunk of time. This provides a sense
        of temporality.

        :return:
        """
        current_rounded_time = round_time_now_down_to_nearst_interval()
        current_time_record = await self.driver.query(
            "MERGE (t:Time {text: $time}) RETURN t", {"time": str(current_rounded_time)})
        logger.info(f"time: {current_time_record}")
        last_rounded_time = current_rounded_time - timedelta(minutes=15)
        time_record_links = await self.driver.query(
            "MATCH (n:Time {text: $now}), (l:Time {text: $last}) "
            "MERGE (n)-[rp:FOLLOWS]->(l) "
            "MERGE (l)-[rf:PRECEDES]->(n) "
            "RETURN rp, rf", {"now": str(current_rounded_time), "last": str(last_rounded_time)})
        logger.debug(f"time links: {time_record_links}")
        return current_rounded_time, current_time_record, time_record_links

    async def close(self):
        await self.driver.close()

    async def get_semantic_category_desc_by_connections(self):
        # Get a random topic that is linked to two ideas that are not the same and not connected
        semantic_category_results = await self.driver.query("""
            MATCH (semanticCategory:SemanticCategory)
            OPTIONAL MATCH ()-[r]->(semanticCategory)
            RETURN semanticCategory, count(r) AS connections
            ORDER by connections DESC
            """)
        return semantic_category_results

    async def link_chat_request_to_chat_session(self, chat_request_received_id: int, chat_session_id: int, time_occurred):
        results = await self.driver.query("""
        MATCH (req:ChatRequest {chat_request_received_id: $chat_request_received_id}), (session:ChatSession {chat_session_id: $chat_session_id})
        MERGE (session)-[r:RECEIVED {time_occurred: $time_occurred}]->(req)
        RETURN r
        """, {
            "chat_request_received_id": chat_request_received_id, "chat_session_id": chat_session_id,
            "time_occurred": time_occurred,

        })
        logger.info(f"request/session link: {results}")
        return results

    async def link_chat_response_to_chat_request(self, chat_request_received_id: int, chat_response_sent_id: int,
                                                 chat_session_id: int, time_occurred):
        results = await self.driver.query("""
        MATCH (req:ChatRequest {chat_request_received_id: $chat_request_received_id}), (resp:ChatResponse {chat_response_sent_id: $chat_response_sent_id})
        MERGE (resp)-[r:REACTS_TO {chat_session_id: $chat_session_id, time_occurred: $time_occurred}]->(req)
        RETURN r
        """, {
            "chat_request_received_id": chat_request_received_id, "chat_response_sent_id": chat_response_sent_id,
            "chat_session_id": chat_session_id, "time_occurred": time_occurred
        })
        logger.info(f"request/response link: {results}")
        return results

    async def link_chat_response_to_chat_session(self, chat_response_sent_id: int, chat_session_id: int, time_occurred):
        results = await self.driver.query("""
        MATCH (resp:ChatResponse {chat_response_sent_id: $chat_response_sent_id}), (session:ChatSession {chat_session_id: $chat_session_id})
        MERGE (session)-[r:SENT {time_occurred: $time_occurred}]->(resp)
        RETURN r
        """, {
            "chat_response_sent_id": chat_response_sent_id, "chat_session_id": chat_session_id,
            "time_occurred": time_occurred,

        })
        logger.info(f"response/session link: {results}")
        return results

    async def link_noun_form_to_sentence(self, noun: str, form: str, tag: str, sentence_id: int):
        results = await self.driver.query(f"""
        MATCH (noun:Noun {{text: $noun}}), (sentence:Sentence {{sentence_id: $sentence_id}})
        MERGE (sentence)-[r:{tag} {{form: $form, sentence_id: $sentence_id}}]->(noun)
        RETURN r
        """, {
            "noun": noun, "form": form, "sentence_id": sentence_id
        })
        logger.info(f"noun link: {results}")
        return results

    async def link_noun_to_semantic_category(self, noun: str, semantic_category: str, sentence_id: int):
        semantic_category_link_record = await self.driver.query(
            "MATCH (noun:Noun {text: $noun}), (semanticCategory:SemanticCategory {text: $semanticCategory}) "
            "MERGE (noun)-[r:TYPE_OF {sentence_id: $sentence_id}]->(semanticCategory) "
            "RETURN r", {
                "noun": noun, "semanticCategory": semantic_category, "sentence_id": sentence_id,
            })
        logger.info(f"noun type link: {semantic_category_link_record}")
        return semantic_category_link_record

    async def link_page_element_to_chat_response(self, element_type: str, element_attrs,
                                                 chat_response_sent_id: int):
        element_attrs_expr = ", ".join([f"{k}: ${k}" for k in element_attrs.keys()])
        results = await self.driver.query(f"""
        MATCH (element:{element_type} {{{element_attrs_expr}}}), (resp:ChatResponse {{chat_response_sent_id: $chat_response_sent_id}})
        MERGE (resp)-[r:CONTAINS]->(element)
        RETURN r
        """, {
            "chat_response_sent_id": chat_response_sent_id,
            **element_attrs,
        })
        logger.info(f"element/response link: {results}")
        return results

    async def link_paragraph_to_sentence(self, paragraph_id: int, sentence_id: int):
        results = await self.driver.query("""
        MATCH (paragraph:Paragraph {paragraph_id: $paragraph_id}), (sentence:Sentence {sentence_id: $sentence_id})
        MERGE (paragraph)-[r:CONTAINS]->(sentence)
        RETURN r
        """, {
            "paragraph_id": paragraph_id, "sentence_id": sentence_id
        })
        logger.info(f"sentence/paragraph link: {results}")
        return results

    async def link_pos_tag_to_last_pos_tag(self, tag: str, tag_idx: int, last_tag: str, last_tag_idx: int, sentence_id: int):
        results = await self.driver.query("""
        MATCH (pos:PartOfSpeech {tag: $tag}), (lastPos:PartOfSpeech {tag: $last_tag})
        MERGE (lastPos)-[r:PRECEDES {sentence_id: $sentence_id, predecessor_idx: $last_tag_idx, successor_idx: $tag_idx}]->(pos)
        RETURN r
        """, {
            "tag": tag, "tag_idx": tag_idx, "last_tag": last_tag, "last_tag_idx": last_tag_idx,
            "sentence_id": sentence_id,
        })
        logger.info(f"pos link: {results}")
        return results

    async def link_proper_noun_form_to_sentence(self, proper_noun: str, form: str, tag: str, sentence_id: int):
        results = await self.driver.query(f"""
        MATCH (properNoun:ProperNoun {{text: $properNoun}}), (sentence:Sentence {{sentence_id: $sentence_id}})
        MERGE (sentence)-[r:{tag} {{form: $form, sentence_id: $sentence_id}}]->(properNoun)
        RETURN r
        """, {
            "properNoun": proper_noun, "form": form, "sentence_id": sentence_id
        })
        logger.info(f"proper_noun link: {results}")
        return results

    async def link_reactive_sentence_to_chat_request(self, chat_request_received_id: int, sentence_id: int):
        results = await self.driver.query("""
        MATCH (req:ChatRequest {chat_request_received_id: $chat_request_received_id}), (sentence:Sentence {sentence_id: $sentence_id})
        MERGE (sentence)-[r:REACTS_TO]->(req)
        RETURN r
        """, {
            "chat_request_received_id": chat_request_received_id, "sentence_id": sentence_id
        })
        logger.info(f"sentence/request link: {results}")
        return results

    async def link_sentence_to_previous_sentence(self, previous_sentence_id: int, sentence_id: int):
        results = await self.driver.query("""
        MATCH (prev:Sentence {sentence_id: $previous_sentence_id}), (this:Sentence {sentence_id: $sentence_id})
        MERGE (prev)-[r:PRECEDES]->(this)
        RETURN r
        """, {
            "previous_sentence_id": previous_sentence_id, "sentence_id": sentence_id,

        })
        logger.info(f"sentence/previous link: {results}")
        return results

    async def link_successive_page_elements(self, predecessor_type: str, predecessor_attrs,
                                            successor_type: str, successor_attrs, link_attrs):
        link_attrs_expr = ", ".join([f"{k}: ${k}" for k in link_attrs.keys()])
        predecessor_attrs_expr = ", ".join([f"{k}: $predecessor_{k}" for k in predecessor_attrs.keys()])
        successor_attrs_expr = ", ".join([f"{k}: $successor_{k}" for k in successor_attrs.keys()])
        results = await self.driver.query(f"""
        MATCH (predecessor:{predecessor_type} {{{predecessor_attrs_expr}}}), (successor:{successor_type} {{{successor_attrs_expr}}})
        MERGE (predecessor)-[r:PRECEDES {{{link_attrs_expr}}}]->(successor)
        RETURN r
        """, {
            **link_attrs,
            **{f"predecessor_{k}": v for k, v in predecessor_attrs.items()},
            **{f"successor_{k}": v for k, v in successor_attrs.items()},
        })
        logger.info(f"successive element link: {results}")
        return results

    def signal_handler(self, sig, frame):
        asyncio.run(self.close())


def is_stale_or_over_fetch_count_threshold(dt, fetch_count: int, hours: int, max_fetch_count: int = 25):
    """

    :param dt:  Time changed Datetime object
    :param fetch_count: Number of times fetched from OpenAI
    :param hours: Number of hours to wait before fetching again
    :param max_fetch_count: Maximum number of times fetched from OpenAI since the value diminishes.
    :return:
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (fetch_count >= max_fetch_count  # Value of fetching diminishes
            or (utc_now() - dt) > timedelta(hours=hours))


def round_time_now_down_to_nearst_interval(interval_minutes: int = 5):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    minutes = now.minute
    remainder = minutes % interval_minutes  # How many interval_minutes increments have passed
    rounded_minutes = minutes - remainder  # Subtract to round down
    return now.replace(minute=rounded_minutes, second=0, microsecond=0)


def utc_now():
    return datetime.now(timezone.utc)


idea_graph = IdeaGraph(AsyncNeo4jDriver())
signal.signal(signal.SIGINT, idea_graph.signal_handler)
