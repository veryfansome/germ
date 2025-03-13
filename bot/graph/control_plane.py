from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from sqlalchemy import MetaData, Table, and_
from sqlalchemy import insert as sql_insert
from sqlalchemy.future import select as sql_select
from typing import Optional
import asyncio
import uuid

from bot.db.models import AsyncSessionLocal, Sentence, engine
from bot.db.neo4j import AsyncNeo4jDriver
from observability.logging import logging
from settings.germ_settings import UUID5_NS

logger = logging.getLogger(__name__)


class CodeBlockMergeEventHandler(ABC):
    @abstractmethod
    async def on_code_block_merge(self, code_block: str, text_block_id: int):
        pass


class ParagraphMergeEventHandler(ABC):
    @abstractmethod
    async def on_paragraph_merge(self, paragraph: str, paragraph_id: int, paragraph_context):
        pass


class SentenceMergeEventHandler(ABC):
    @abstractmethod
    async def on_sentence_merge(self, sentence: str, sentence_id: int, sentence_context):
        pass


class ControlPlane:
    def __init__(self, driver: AsyncNeo4jDriver):
        self.driver = driver
        self.struct_type_table = None
        self.text_block_table = None

        self.code_block_merge_event_handlers: list[CodeBlockMergeEventHandler] = []
        self.code_text_block_type_id: Optional[int] = None

        self.paragraph_merge_event_handlers: list[ParagraphMergeEventHandler] = []
        self.paragraph_text_block_type_id: Optional[int] = None

        self.sentence_merge_event_handlers: list[SentenceMergeEventHandler] = []

    async def initialize(self):
        self.struct_type_table = Table('struct_type', MetaData(), autoload_with=engine)
        self.text_block_table = Table('text_block', MetaData(), autoload_with=engine)
        async with (AsyncSessionLocal() as rdb_session):
            async with rdb_session.begin():
                code_text_block_type_stmt = self.struct_type_table.select().where(
                    and_(self.struct_type_table.c.group_name == "text_block_type",
                         self.struct_type_table.c.att_pub_ident == "code"))
                code_text_block_type_record = (await rdb_session.execute(code_text_block_type_stmt)).first()
                self.code_text_block_type_id = code_text_block_type_record.struct_type_id

                paragraph_text_block_type_stmt = self.struct_type_table.select().where(
                    and_(self.struct_type_table.c.group_name == "text_block_type",
                         self.struct_type_table.c.att_pub_ident == "paragraph"))
                paragraph_text_block_type_record = (await rdb_session.execute(paragraph_text_block_type_stmt)).first()
                self.paragraph_text_block_type_id = paragraph_text_block_type_record.struct_type_id

    async def add_adjective(self, adj: str):
        results = await self.driver.query("""
        MATCH (adj:Adjective {text: $adj})
        RETURN adj
        """, {
            "adj": adj,
        })
        if not results:
            results = await self.driver.query("""
            MERGE (adj:Adjective {text: $adj})
            WITH adj
            UNWIND (COALESCE(adj.forms, []) + [$adj]) AS forms
            WITH adj, COLLECT(DISTINCT forms) AS uniqueForms
            SET adj.forms = uniqueForms
            RETURN adj
            """, {
                "adj": adj,
            })
            if results:
                logger.info(f"MERGE (adj:Adjective {{text: {adj}}})")
            else:
                logger.error(f"failed to add adj: '{adj}'")
        return results

    async def add_adjective_form(self, adj: str, form: str):
        results = await self.driver.query("""
        MATCH (adj:Adjective {text: $adj})
        WITH adj
        UNWIND (COALESCE(adj.forms, []) + [$form]) AS form
        WITH adj, COLLECT(DISTINCT form) AS uniqueForms
        SET adj.forms = uniqueForms
        RETURN adj
        """, {
            "adj": adj, "form": form
        })
        if results:
            logger.info(f"SET (adj:Adjective {{{adj}}}).forms = {results[0]['adj']['forms']}}} << {form}")
        else:
            logger.error(f"failed to add adj form: '{adj}' form='{form}'")
        return results

    async def add_adverb(self, adv: str):
        results = await self.driver.query("""
        MATCH (adv:Adverb {text: $adv})
        RETURN adv
        """, {
            "adv": adv,
        })
        if not results:
            results = await self.driver.query("""
            MERGE (adv:Adverb {text: $adv})
            WITH adv
            UNWIND (COALESCE(adv.forms, []) + [$adv]) AS forms
            WITH adv, COLLECT(DISTINCT forms) AS uniqueForms
            SET adv.forms = uniqueForms
            RETURN adv
            """, {
                "adv": adv,
            })
            if results:
                logger.info(f"MERGE (adv:Adverb {{text: {adv}}})")
            else:
                logger.error(f"failed to add adv: '{adv}'")
        return results

    async def add_adverb_form(self, adv: str, form: str):
        results = await self.driver.query("""
        MATCH (adv:Adverb {text: $adv})
        WITH adv
        UNWIND (COALESCE(adv.forms, []) + [$form]) AS form
        WITH adv, COLLECT(DISTINCT form) AS uniqueForms
        SET adv.forms = uniqueForms
        RETURN adv
        """, {
            "adv": adv, "form": form
        })
        if results:
            logger.info(f"SET (adv:Adverb {{{adv}}}).forms = {results[0]['adv']['forms']}}} << {form}")
        else:
            logger.error(f"failed to add adv form: '{adv}' form='{form}'")
        return results

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

    async def add_code_block(self, code_block: str, code_block_context):
        code_block = code_block.strip()
        # Generate signature for lookup in PostgreSQL
        code_block_signature = uuid.uuid5(UUID5_NS, code_block)
        async with (AsyncSessionLocal() as rdb_session):
            async with rdb_session.begin():
                rdb_select_stmt = self.text_block_table.select().where(
                    self.text_block_table.c.signature_uuid == code_block_signature)
                rdb_record = (await rdb_session.execute(rdb_select_stmt)).first()

                # Create record if not found
                if not rdb_record:
                    sql_insert_stmt = sql_insert(self.text_block_table).values({
                        "signature_uuid": code_block_signature,
                        "text_block_type_id": self.code_text_block_type_id,
                    }).returning(self.text_block_table.c.text_block_id)

                    # Execute the insert and fetch the returned values
                    rdb_record = (await rdb_session.execute(sql_insert_stmt)).first()

        controller_context = code_block_context.pop("_", {})  # Exclude controller contexts
        code_block_context["time_occurred"] = round_time_now_down_to_nearst_interval()
        code_block_context_expr = ", ".join([f"{k}: ${k}" for k in code_block_context.keys()])
        graph_results = await self.driver.query(f"""
        MERGE (block:CodeBlock {{text: $text, code_block_id: $code_block_id, {code_block_context_expr}}})
        RETURN block
        """, {
            "text": code_block,
            "code_block_id": rdb_record.text_block_id,
            **code_block_context
        })
        code_block_context["_"] = controller_context

        async_tasks = []
        if graph_results:
            result_sig = ", ".join([f"{k}: {v}" for k, v in graph_results[0]['block'].items() if k.endswith("_id")])
            logger.info(f"MERGE (block:CodeBlock {{{result_sig}}}")
            for handler in self.code_block_merge_event_handlers:
                async_tasks.append(asyncio.create_task(
                    handler.on_code_block_merge(code_block, rdb_record.text_block_id)))
        return graph_results, rdb_record.text_block_id, async_tasks

    def add_code_block_merge_event_handler(self, handler: CodeBlockMergeEventHandler):
        self.code_block_merge_event_handlers.append(handler)

    async def add_hyperlink(self, html_tag: str, url: str):
        results = await self.driver.query("""
        MERGE (hyperlink:Hyperlink {url: $url, html_tag: $html_tag})
        RETURN hyperlink
        """, {
            "html_tag": html_tag, "url": url,
        })
        logger.info(f"hyperlink: {results}")
        return results

    async def add_noun(self, noun: str, sentence_id: int):
        results = await self.driver.query("""
        MATCH (noun:Noun {text: $noun, sentence_id: $sentence_id})
        RETURN noun
        """, {
            "noun": noun, "sentence_id": sentence_id,
        })
        if not results:
            results = await self.driver.query("""
            MERGE (noun:Noun {text: $noun, sentence_id: $sentence_id})
            WITH noun
            UNWIND (COALESCE(noun.forms, []) + [$noun]) AS forms
            WITH noun, COLLECT(DISTINCT forms) AS uniqueForms
            SET noun.forms = uniqueForms
            RETURN noun
            """, {
                "noun": noun, "sentence_id": sentence_id,
            })
            if results:
                logger.info(f"MERGE (noun:Noun {{text: {noun}}})")
            else:
                logger.error(f"failed to add noun: '{noun}' sentence_id={sentence_id}")
        return results

    async def add_noun_class(self, noun: str):
        results = await self.driver.query("""
        MATCH (cls:Noun {text: $cls, sentence_id: 0})
        RETURN cls
        """, {
            "cls": noun
        })
        if not results:
            results = await self.driver.query("""
            MERGE (cls:Noun {text: $cls, sentence_id: 0})
            WITH cls
            UNWIND (COALESCE(cls.forms, []) + [$cls]) AS forms
            WITH cls, COLLECT(DISTINCT forms) AS uniqueForms
            SET cls.forms = uniqueForms
            RETURN cls
            """, {
                "cls": noun
            })
            if results:
                logger.info(f"MERGE (cls:Noun {{text: {noun}}})")
            else:
                logger.error(f"failed to add cls: '{noun}'")
        return results

    async def add_noun_class_form(self, noun: str, form: str):
        results = await self.driver.query("""
        MATCH (cls:Noun {text: $cls, sentence_id: 0})
        WITH cls
        UNWIND (COALESCE(cls.forms, []) + [$form]) AS forms
        WITH cls, COLLECT(DISTINCT forms) AS uniqueForms
        SET cls.forms = uniqueForms
        RETURN cls
        """, {
            "cls": noun, "form": form
        })
        if results:
            logger.info(f"SET (cls:Noun {{{noun}}}).forms = {results[0]['cls']['forms']}}} << {form}")
        else:
            logger.error(f"failed to add noun class form: '{noun}' form='{form}'")
        return results

    async def add_noun_form(self, noun: str, form: str, sentence_id: int):
        results = await self.driver.query("""
        MATCH (noun:Noun {text: $noun, sentence_id: $sentence_id})
        WITH noun
        UNWIND (COALESCE(noun.forms, []) + [$form]) AS forms
        WITH noun, COLLECT(DISTINCT forms) AS uniqueForms
        SET noun.forms = uniqueForms
        RETURN noun
        """, {
            "noun": noun, "form": form, "sentence_id": sentence_id,
        })
        if results:
            logger.info(f"SET (noun:Noun {{{noun}}}).forms = {results[0]['noun']['forms']}}} << {form}")
        else:
            logger.error(f"failed to add noun form: '{noun}' form='{form}' sentence_id={sentence_id}")
        return results

    async def add_paragraph(self, paragraph: str, paragraph_context):
        paragraph = paragraph.strip()
        # Generate signature for lookup in PostgreSQL
        paragraph_signature = uuid.uuid5(UUID5_NS, paragraph)
        async with (AsyncSessionLocal() as rdb_session):
            async with rdb_session.begin():
                rdb_select_stmt = self.text_block_table.select().where(
                    self.text_block_table.c.signature_uuid == paragraph_signature)
                rdb_record = (await rdb_session.execute(rdb_select_stmt)).first()

                # Create record if not found
                if not rdb_record:
                    sql_insert_stmt = sql_insert(self.text_block_table).values({
                        "signature_uuid": paragraph_signature,
                        "text_block_type_id": self.paragraph_text_block_type_id,
                    }).returning(self.text_block_table.c.text_block_id)

                    # Execute the insert and fetch the returned values
                    rdb_record = (await rdb_session.execute(sql_insert_stmt)).first()

        controller_context = paragraph_context.pop("_", {})  # Exclude controller contexts
        paragraph_context["paragraph_id"] = rdb_record.text_block_id
        paragraph_context["time_occurred"] = round_time_now_down_to_nearst_interval()
        paragraph_context_expr = ", ".join([f"{k}: ${k}" for k in paragraph_context.keys()])
        graph_results = await self.driver.query(f"""
        MERGE (paragraph:Paragraph {{text: $text, paragraph_id: $paragraph_id, {paragraph_context_expr}}})
        RETURN paragraph
        """, {
            "text": paragraph,
            **paragraph_context,
        })
        paragraph_context["_"] = controller_context

        async_tasks = []
        if graph_results:
            result_sig = ", ".join([f"{k}: {v}" for k, v in graph_results[0]['paragraph'].items() if k.endswith("_id")])
            logger.info(f"MERGE (paragraph:Paragraph {{{result_sig}}}")
            for handler in self.paragraph_merge_event_handlers:
                async_tasks.append(asyncio.create_task(
                    handler.on_paragraph_merge(paragraph, rdb_record.text_block_id, paragraph_context)))
        return graph_results, rdb_record.text_block_id, async_tasks

    def add_paragraph_merge_event_handler(self, handler: ParagraphMergeEventHandler):
        self.paragraph_merge_event_handlers.append(handler)

    async def add_pronoun(self, pronoun: str, sentence_id: int):
        results = await self.driver.query("""
        MERGE (pronoun:Pronoun {text: $pronoun, sentence_id: $sentence_id})
        RETURN pronoun
        """, {
            "pronoun": pronoun, "sentence_id": sentence_id,
        })
        if results:
            logger.info(f"MERGE (pronoun:Pronoun {{text: {pronoun}, sentence_id: {sentence_id}}})")
        else:
            logger.error(f"failed to add pronoun: '{pronoun}' sentence_id={sentence_id}")
        return results

    async def add_sentence(self, sentence: str, sentence_context):
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

            # Merge sentence node
            graph_results = await self.driver.query(
                "MERGE (sentence:Sentence {text: $text, sentence_id: $sentence_id}) RETURN sentence",
                {
                    "text": sentence,
                    "sentence_id": rdb_record.sentence_id,
                })
            logger.info(f"sentence node: {graph_results}")
        async_tasks = []
        for handler in self.sentence_merge_event_handlers:
            async_tasks.append(asyncio.create_task(
                handler.on_sentence_merge(
                    sentence, rdb_record.sentence_id, sentence_context)))
        return graph_results, rdb_record.sentence_id, async_tasks

    def add_sentence_merge_event_handler(self, handler: SentenceMergeEventHandler):
        self.sentence_merge_event_handlers.append(handler)

    async def close(self):
        await self.driver.shutdown()

    async def get_paragraph(self, paragraph_id: int):
        return await self.driver.query("""
        MATCH (paragraph:Paragraph {paragraph_id: $paragraph_id})
        RETURN paragraph
        """, {
            "paragraph_id": paragraph_id,
        })

    async def link_adj_as_noun_attr(self, adj: str, noun: str, sentence_id: int, negative: bool = False):
        link_type = "NOT_ATTR_OF" if negative else "ATTR_OF"
        results = await self.driver.query(f"""
        MATCH (adj:Adjective {{text: $adj}}), (noun:Noun {{text: $noun, sentence_id: $sentence_id}})
        MERGE (adj)-[r:{link_type}]->(noun)
        RETURN r
        """, {
            "adj": adj, "noun": noun, "sentence_id": sentence_id,
        })
        if results:
            logger.info(f"MERGE (adj:Adjective {{text: {adj}}})-[r:{link_type}]->(noun:Noun {{text: {noun}}})")
        else:
            logger.error(f"failed to link adjective '{adj}' to noun '{noun}'")
        return results

    async def link_adj_as_pronoun_to_attr(self, adj: str, pronoun: str, sentence_id: int, negative: bool = False):
        link_type = "NOT_ATTR_OF" if negative else "ATTR_OF"
        results = await self.driver.query(f"""
        MATCH (adj:Adjective {{text: $adj}}), (pronoun:Pronoun {{text: $pronoun, sentence_id: $sentence_id}})
        MERGE (adj)-[r:{link_type}]->(pronoun)
        RETURN r
        """, {
            "adj": adj, "pronoun": pronoun, "sentence_id": sentence_id,
        })
        if results:
            logger.info(f"MERGE (adj:Adjective {{text: {adj}}})-[r:{link_type}]->(pronoun:Pronoun {{text: {pronoun}}})")
        else:
            logger.error(f"failed to link adjective '{adj}' to pronoun '{pronoun}'")
        return results

    async def link_adv_to_adj(self, adv: str, adj: str):
        results = await self.driver.query("""
        MATCH (adv:Adverb {text: $adv}), (adj:Adjective {text: $adj})
        MERGE (adv)-[r:ATTR_OF]->(adj)
        RETURN r
        """, {
            "adv": adv, "adj": adj,
        })
        if results:
            logger.info(f"MERGE (adv:Adverb {{text: {adv}}})-[r:ATTR_OF]->(adj:Adjective {{text: {adj}}})")
        else:
            logger.error(f"failed to link adverb '{adv}' to adjective '{adj}'")
        return results

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

    async def link_noun_form_to_sentence(self, noun: str, form: str, link_name: str, sentence_id: int):
        link_name = link_name.upper()
        results = await self.driver.query(f"""
        MATCH (noun:Noun {{text: $noun, sentence_id: $sentence_id}}), (sentence:Sentence {{sentence_id: $sentence_id}})
        MERGE (noun)-[r:{link_name} {{form: $form, sentence_id: $sentence_id}}]->(sentence)
        RETURN r
        """, {
            "noun": noun, "form": form, "sentence_id": sentence_id
        })
        if results:
            logger.info(f"MERGE (noun:Noun {{text: {noun}}})-[r:{link_name} {{form: {form}}}]->(sentence:Sentence {{sentence_id: {sentence_id}}})")
        return results

    async def link_noun_to_phrase(self, noun: str, phrase: str, sentence_id: int):
        results = await self.driver.query("""
        MATCH (noun:Noun {text: $noun, sentence_id: $sentence_id}), (phrase:Noun {text: $phrase, sentence_id: $sentence_id})
        MERGE (phrase)-[r:CONTAINS]->(noun)
        RETURN r
        """, {
            "noun": noun, "phrase": phrase, "sentence_id": sentence_id,
        })
        if results:
            logger.info(f"MERGE (phrase:Noun {{text: {phrase}}})-[r:CONTAINS]->(noun:Noun {{text: {noun}}})")
        return results

    async def link_noun_to_possessor(self, possessor: str, noun: str, sentence_id: int):
        results = await self.driver.query("""
        MATCH (pos:Noun {text: $pos, sentence_id: $sentence_id}), (noun:Noun {text: $noun, sentence_id: $sentence_id})
        MERGE (noun)-[r:BELONGING_TO]->(pos)
        RETURN r
        """, {
            "pos": possessor, "noun": noun, "sentence_id": sentence_id,
        })
        if results:
            logger.info(f"MERGE (noun:Noun {{text: {noun}}})-[r:OF]->(pos:Noun {{text: {possessor}}})")
        return results

    async def link_nouns_via_preposition(self, noun1: str, preposition: str, noun2: str, sentence_id: int):
        link_name = preposition.upper()
        results = await self.driver.query(f"""
        MATCH (noun1:Noun {{text: $noun1, sentence_id: $sentence_id}}), (noun2:Noun {{text: $noun2, sentence_id: $sentence_id}})
        MERGE (noun1)-[r:{link_name}]->(noun2)
        RETURN r
        """, {
            "noun1": noun1, "noun2": noun2, "sentence_id": sentence_id,
        })
        if results:
            logger.info(f"MERGE (noun1:Noun {{text: {noun1}}})-[r:{link_name}]->(noun2:Noun {{text: {noun2}}})")
        return results

    async def link_nouns_via_verb(self, nsubj: str, base_verb: str, obj: str, sentence_id: int):
        link_name = base_verb.upper()
        results = await self.driver.query(f"""
        MATCH (nsubj:Noun {{sentence_id: $sentence_id}}), (obj:Noun {{sentence_id: $sentence_id}})
        WHERE $nsubj IN nsubj.forms AND $obj IN obj.forms
        MERGE (nsubj)-[r:{link_name}]->(obj)
        RETURN r
        """, {
            "nsubj": nsubj, "obj": obj, "sentence_id": sentence_id,
        })
        if results:
            logger.info(f"MERGE (nsubj:Noun {{text: {nsubj}}})-[r:{link_name}]->(obj:Noun {{text: {obj}}})")
        return results

    async def link_page_element_to_chat_request(self, element_type: str, element_attrs,
                                                chat_request_received_id: int):
        element_attrs_expr = ", ".join([f"{k}: ${k}" for k in element_attrs.keys()])
        results = await self.driver.query(f"""
        MATCH (element:{element_type} {{{element_attrs_expr}}}), (req:ChatRequest {{chat_request_received_id: $chat_request_received_id}})
        MERGE (req)-[r:CONTAINS]->(element)
        RETURN element,r
        """, {
            "chat_request_received_id": chat_request_received_id,
            **element_attrs,
        })
        if results:
            result_sig = ", ".join([f"{k}: {v}" for k, v in results[0]['element'].items() if k.endswith("_id")])
            logger.info("MERGE "
                        f"(req:ChatRequest {{chat_request_received_id: {chat_request_received_id}}})-[r:CONTAINS]->"
                        f"(element:{element_type} {{{result_sig}}})")
        return results

    async def link_page_element_to_chat_response(self, element_type: str, element_attrs,
                                                 chat_response_sent_id: int):
        element_attrs_expr = ", ".join([f"{k}: ${k}" for k in element_attrs.keys()])
        results = await self.driver.query(f"""
        MATCH (element:{element_type} {{{element_attrs_expr}}}), (resp:ChatResponse {{chat_response_sent_id: $chat_response_sent_id}})
        MERGE (resp)-[r:CONTAINS]->(element)
        RETURN element,r
        """, {
            "chat_response_sent_id": chat_response_sent_id,
            **element_attrs,
        })
        if results:
            result_sig = ", ".join([f"{k}: {v}" for k, v in results[0]['element'].items() if k.endswith("_id")])
            logger.info("MERGE "
                        f"(req:ChatResponse {{chat_response_sent_id: {chat_response_sent_id}}})-[r:CONTAINS]->"
                        f"(element:{element_type} {{{result_sig}}})")
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

    async def link_pronoun_to_sentence(self, pronoun: str, link_name: str, sentence_id: int):
        link_name = link_name.upper()
        results = await self.driver.query(f"""
        MATCH (pronoun:Pronoun {{text: $pronoun, sentence_id: $sentence_id}}), (sentence:Sentence {{sentence_id: $sentence_id}})
        MERGE (pronoun)-[r:{link_name} {{sentence_id: $sentence_id}}]->(sentence)
        RETURN r
        """, {
            "pronoun": pronoun, "sentence_id": sentence_id
        })
        if results:
            logger.info(f"MERGE (pronoun:Pronoun {{text: {pronoun}}})-[r:{link_name}]->(sentence:Sentence {{sentence_id: {sentence_id}}})")
        else:
            logger.error(f"failed to link pronoun '{pronoun}' to sentence_id {sentence_id}")
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

    async def link_successive_nouns(self, previous_noun: str, noun: str):
        results = await self.driver.query("""
        MATCH (prev:Noun {text: $previous_noun}), (this:Noun {text: $noun})
        MERGE (prev)-[r:PRECEDES]->(this)
        RETURN r
        """, {
            "previous_noun": previous_noun, "noun": noun,

        })
        if results:
            logger.info(f"MERGE (prev:Noun {{text: {previous_noun}}})-[r:PRECEDES]->(this:Noun {{text: {noun}}})")
        return results

    async def link_successive_page_elements(self, predecessor_type: str, predecessor_attrs,
                                            successor_type: str, successor_attrs, link_attrs):
        link_attrs_expr = ", ".join([f"{k}: ${k}" for k in link_attrs.keys()])
        predecessor_attrs_expr = ", ".join([f"{k}: $predecessor_{k}" for k in predecessor_attrs.keys()])
        successor_attrs_expr = ", ".join([f"{k}: $successor_{k}" for k in successor_attrs.keys()])
        results = await self.driver.query(f"""
        MATCH (predecessor:{predecessor_type} {{{predecessor_attrs_expr}}}), (successor:{successor_type} {{{successor_attrs_expr}}})
        MERGE (predecessor)-[r:PRECEDES {{{link_attrs_expr}}}]->(successor)
        RETURN predecessor,r,successor
        """, {
            **link_attrs,
            **{f"predecessor_{k}": v for k, v in predecessor_attrs.items()},
            **{f"successor_{k}": v for k, v in successor_attrs.items()},
        })
        if results:
            predecessor_sig = ", ".join([f"{k}: {v}" for k, v in results[0]['predecessor'].items() if k.endswith("_id")])
            successor_sig = ", ".join([f"{k}: {v}" for k, v in results[0]['successor'].items() if k.endswith("_id")])
            logger.info("MERGE "
                        f"(predecessor:{predecessor_type} {{{predecessor_sig}}})"
                        "-[r:PRECEDES]->"
                        f"(successor:{successor_type} {{{successor_sig}}})")
        return results

    async def link_successive_sentence(self, previous_sentence_id: int, sentence_id: int):
        results = await self.driver.query("""
        MATCH (prev:Sentence {sentence_id: $previous_sentence_id}), (this:Sentence {sentence_id: $sentence_id})
        MERGE (prev)-[r:PRECEDES]->(this)
        RETURN r
        """, {
            "previous_sentence_id": previous_sentence_id, "sentence_id": sentence_id,

        })
        logger.info(f"sentence/previous link: {results}")
        return results

    def signal_handler(self, sig, frame):
        asyncio.run(self.close())


def round_time_now_down_to_nearst_interval(interval_minutes: int = 5):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    minutes = now.minute
    remainder = minutes % interval_minutes  # How many interval_minutes increments have passed
    rounded_minutes = minutes - remainder  # Subtract to round down
    return now.replace(minute=rounded_minutes, second=0, microsecond=0)


def utc_now():
    return datetime.now(timezone.utc)
