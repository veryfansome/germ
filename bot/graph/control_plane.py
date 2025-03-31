from abc import ABC, abstractmethod
from datetime import datetime, timezone
from sqlalchemy import MetaData, Table, and_
from sqlalchemy import insert as sql_insert
from typing import Optional
import asyncio
import uuid

from bot.db.models import AsyncSessionLocal, engine
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
    async def on_paragraph_merge(self, paragraph: str, text_block_id: int, paragraph_context):
        pass


class SentenceMergeEventHandler(ABC):
    @abstractmethod
    async def on_sentence_merge(self, sentence: str, text_block_id: int, sentence_context):
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
        self.sentence_text_block_type_id: Optional[int] = None

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

                sentence_text_block_type_stmt = self.struct_type_table.select().where(
                    and_(self.struct_type_table.c.group_name == "text_block_type",
                         self.struct_type_table.c.att_pub_ident == "sentence"))
                sentence_text_block_type_record = (await rdb_session.execute(sentence_text_block_type_stmt)).first()
                self.sentence_text_block_type_id = sentence_text_block_type_record.struct_type_id

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
        rdb_record = await self.add_text_block(code_block, self.code_text_block_type_id)

        controller_context = code_block_context.pop("_", {})  # Exclude controller contexts
        code_block_context["time_occurred"] = round_time_now_down_to_nearst_interval()
        code_block_context_expr = ", ".join([f"{k}: ${k}" for k in code_block_context.keys()])
        graph_results = await self.driver.query(f"""
        MERGE (block:CodeBlock {{text: $text, text_block_id: $text_block_id, {code_block_context_expr}}})
        RETURN block
        """, {
            "text": code_block,
            "text_block_id": rdb_record.text_block_id,
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

    async def add_noun(self, noun: str, text_block_id: int):
        results = await self.driver.query("""
        MATCH (noun:Noun {text: $noun, text_block_id: $text_block_id})
        RETURN noun
        """, {
            "noun": noun, "text_block_id": text_block_id,
        })
        if not results:
            results = await self.driver.query("""
            MERGE (noun:Noun {text: $noun, text_block_id: $text_block_id})
            WITH noun
            UNWIND (COALESCE(noun.forms, []) + [$noun]) AS forms
            WITH noun, COLLECT(DISTINCT forms) AS uniqueForms
            SET noun.forms = uniqueForms
            RETURN noun
            """, {
                "noun": noun, "text_block_id": text_block_id,
            })
            if results:
                logger.info(f"MERGE (noun:Noun {{text: {noun}}})")
            else:
                logger.error(f"failed to add noun: '{noun}' text_block_id={text_block_id}")
        return results

    async def add_noun_form(self, noun: str, form: str, text_block_id: int):
        results = await self.driver.query("""
        MATCH (noun:Noun {text: $noun, text_block_id: $text_block_id})
        WITH noun
        UNWIND (COALESCE(noun.forms, []) + [$form]) AS forms
        WITH noun, COLLECT(DISTINCT forms) AS uniqueForms
        SET noun.forms = uniqueForms
        RETURN noun
        """, {
            "noun": noun, "form": form, "text_block_id": text_block_id,
        })
        if results:
            logger.info(f"SET (noun:Noun {{{noun}}}).forms = {results[0]['noun']['forms']}}} << {form}")
        else:
            logger.error(f"failed to add noun form: '{noun}' form='{form}' text_block_id={text_block_id}")
        return results

    async def add_paragraph(self, paragraph: str, paragraph_context):
        paragraph = paragraph.strip()
        rdb_record = await self.add_text_block(paragraph, self.paragraph_text_block_type_id)

        controller_context = paragraph_context.pop("_", {})  # Exclude controller contexts
        paragraph_context["text_block_id"] = rdb_record.text_block_id
        paragraph_context["time_occurred"] = round_time_now_down_to_nearst_interval()
        paragraph_context_expr = ", ".join([f"{k}: ${k}" for k in paragraph_context.keys()])
        graph_results = await self.driver.query(f"""
        MERGE (paragraph:Paragraph {{text: $text, text_block_id: $text_block_id, {paragraph_context_expr}}})
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

    async def add_pronoun(self, pronoun: str, text_block_id: int):
        results = await self.driver.query("""
        MERGE (pronoun:Pronoun {text: $pronoun, text_block_id: $text_block_id})
        RETURN pronoun
        """, {
            "pronoun": pronoun, "text_block_id": text_block_id,
        })
        if results:
            logger.info(f"MERGE (pronoun:Pronoun {{text: {pronoun}, text_block_id: {text_block_id}}})")
        else:
            logger.error(f"failed to add pronoun: '{pronoun}' text_block_id={text_block_id}")
        return results

    async def add_sentence(self, sentence: str, sentence_context):
        sentence = sentence.strip()
        rdb_record = await self.add_text_block(sentence, self.sentence_text_block_type_id)

        graph_results = await self.driver.query(
            "MERGE (sentence:Sentence {text: $text, text_block_id: $text_block_id}) RETURN sentence",
            {
                "text": sentence,
                "text_block_id": rdb_record.text_block_id,
            })
        async_tasks = []
        if graph_results:
            result_sig = ", ".join([f"{k}: {v}" for k, v in graph_results[0]['sentence'].items() if k.endswith("_id")])
            logger.info(f"MERGE (sentence:Sentence {{{result_sig}}}")
            for handler in self.sentence_merge_event_handlers:
                async_tasks.append(asyncio.create_task(
                    handler.on_sentence_merge(
                        sentence, rdb_record.text_block_id, sentence_context)))
        return graph_results, rdb_record.text_block_id, async_tasks

    def add_sentence_merge_event_handler(self, handler: SentenceMergeEventHandler):
        self.sentence_merge_event_handlers.append(handler)

    async def add_text_block(self, text: str, text_block_type_id: int):
        # Generate signature for lookup in PostgreSQL
        signature_uuid = uuid.uuid5(UUID5_NS, text)
        async with (AsyncSessionLocal() as rdb_session):
            async with rdb_session.begin():
                rdb_select_stmt = self.text_block_table.select().where(
                    self.text_block_table.c.signature_uuid == signature_uuid)
                rdb_record = (await rdb_session.execute(rdb_select_stmt)).first()

                # Create record if not found
                if not rdb_record:
                    sql_insert_stmt = sql_insert(self.text_block_table).values({
                        "signature_uuid": signature_uuid,
                        "text_block_type_id": text_block_type_id,
                    }).returning(self.text_block_table.c.text_block_id)

                    # Execute the insert and fetch the returned values
                    rdb_record = (await rdb_session.execute(sql_insert_stmt)).first()
        return rdb_record

    async def add_verb(self, verb: str, text_block_id: int):
        results = await self.driver.query("""
        MATCH (verb:Verb {text: $verb, text_block_id: $text_block_id})
        RETURN verb
        """, {
            "verb": verb, "text_block_id": text_block_id,
        })
        if not results:
            results = await self.driver.query("""
            MERGE (verb:Verb {text: $verb, text_block_id: $text_block_id})
            WITH verb
            UNWIND (COALESCE(verb.forms, []) + [$verb]) AS forms
            WITH verb, COLLECT(DISTINCT forms) AS uniqueForms
            SET verb.forms = uniqueForms
            RETURN verb
            """, {
                "verb": verb, "text_block_id": text_block_id,
            })
            if results:
                logger.info(f"MERGE (verb:Verb {{text: {verb}}})")
            else:
                logger.error(f"failed to add verb: '{verb}' text_block_id={text_block_id}")
        return results

    async def add_verb_form(self, verb: str, form: str, text_block_id: int):
        results = await self.driver.query("""
        MATCH (verb:Verb {text: $verb, text_block_id: $text_block_id})
        WITH verb
        UNWIND (COALESCE(verb.forms, []) + [$form]) AS forms
        WITH verb, COLLECT(DISTINCT forms) AS uniqueForms
        SET verb.forms = uniqueForms
        RETURN verb
        """, {
            "verb": verb, "form": form, "text_block_id": text_block_id,
        })
        if results:
            logger.info(f"SET (verb:Verb {{{verb}}}).forms = {results[0]['verb']['forms']}}} << {form}")
        else:
            logger.error(f"failed to add verb form: '{verb}' form='{form}' text_block_id={text_block_id}")
        return results

    async def close(self):
        await self.driver.shutdown()

    async def get_edges(self):
        return await self.driver.query("""
        MATCH (start)-[edge]->(end) WHERE NOT start:__Neo4jMigration RETURN edge, id(edge) AS edgeId, id(start) AS startNodeId, id(end) AS endNodeId
        """)

    async def get_nodes(self):
        return await self.driver.query("""
        MATCH (node) WHERE NOT node:__Neo4jMigration RETURN node, id(node) AS nodeId, labels(node) AS nodeLabels
        """)

    async def get_paragraph(self, text_block_id: int):
        return await self.driver.query("""
        MATCH (paragraph:Paragraph {text_block_id: $text_block_id})
        RETURN paragraph
        """, {
            "text_block_id": text_block_id,
        })

    async def link_adj_as_noun_attr(self, adj: str, noun: str, text_block_id: int, negative: bool = False):
        link_type = "NOT_ATTR_OF" if negative else "ATTR_OF"
        results = await self.driver.query(f"""
        MATCH (adj:Adjective {{text: $adj}}), (noun:Noun {{text: $noun, text_block_id: $text_block_id}})
        MERGE (adj)-[r:{link_type}]->(noun)
        RETURN r
        """, {
            "adj": adj, "noun": noun, "text_block_id": text_block_id,
        })
        if results:
            logger.info(f"MERGE (adj:Adjective {{text: {adj}}})-[r:{link_type}]->(noun:Noun {{text: {noun}}})")
        else:
            logger.error(f"failed to link adjective '{adj}' to noun '{noun}'")
        return results

    async def link_adj_as_pronoun_to_attr(self, adj: str, pronoun: str, text_block_id: int, negative: bool = False):
        link_type = "NOT_ATTR_OF" if negative else "ATTR_OF"
        results = await self.driver.query(f"""
        MATCH (adj:Adjective {{text: $adj}}), (pronoun:Pronoun {{text: $pronoun, text_block_id: $text_block_id}})
        MERGE (adj)-[r:{link_type}]->(pronoun)
        RETURN r
        """, {
            "adj": adj, "pronoun": pronoun, "text_block_id": text_block_id,
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

    async def link_noun_form_to_sentence(self, noun: str, form: str, link_name: str, text_block_id: int):
        link_name = link_name.upper()
        results = await self.driver.query(f"""
        MATCH (noun:Noun {{text: $noun, text_block_id: $text_block_id}}), (sentence:Sentence {{text_block_id: $text_block_id}})
        MERGE (noun)-[r:{link_name} {{form: $form, text_block_id: $text_block_id}}]->(sentence)
        RETURN r
        """, {
            "noun": noun, "form": form, "text_block_id": text_block_id
        })
        if results:
            logger.info(f"MERGE (noun:Noun {{text: {noun}}})-[r:{link_name} {{form: {form}}}]->(sentence:Sentence {{text_block_id: {text_block_id}}})")
        return results

    async def link_noun_to_phrase(self, noun: str, phrase: str, text_block_id: int):
        results = await self.driver.query("""
        MATCH (noun:Noun {text: $noun, text_block_id: $text_block_id}), (phrase:Noun {text: $phrase, text_block_id: $text_block_id})
        MERGE (phrase)-[r:CONTAINS]->(noun)
        RETURN r
        """, {
            "noun": noun, "phrase": phrase, "text_block_id": text_block_id,
        })
        if results:
            logger.info(f"MERGE (phrase:Noun {{text: {phrase}}})-[r:CONTAINS]->(noun:Noun {{text: {noun}}})")
        return results

    async def link_noun_to_possessor(self, possessor: str, noun: str, text_block_id: int):
        results = await self.driver.query("""
        MATCH (pos:Noun {text: $pos, text_block_id: $text_block_id}), (noun:Noun {text: $noun, text_block_id: $text_block_id})
        MERGE (noun)-[r:BELONGING_TO]->(pos)
        RETURN r
        """, {
            "pos": possessor, "noun": noun, "text_block_id": text_block_id,
        })
        if results:
            logger.info(f"MERGE (noun:Noun {{text: {noun}}})-[r:OF]->(pos:Noun {{text: {possessor}}})")
        return results

    async def link_nouns_via_preposition(self, noun1: str, preposition: str, noun2: str, text_block_id: int):
        link_name = preposition.upper()
        results = await self.driver.query(f"""
        MATCH (noun1:Noun {{text: $noun1, text_block_id: $text_block_id}}), (noun2:Noun {{text: $noun2, text_block_id: $text_block_id}})
        MERGE (noun1)-[r:{link_name}]->(noun2)
        RETURN r
        """, {
            "noun1": noun1, "noun2": noun2, "text_block_id": text_block_id,
        })
        if results:
            logger.info(f"MERGE (noun1:Noun {{text: {noun1}}})-[r:{link_name}]->(noun2:Noun {{text: {noun2}}})")
        return results

    async def link_nouns_via_verb(self, nsubj: str, base_verb: str, obj: str, text_block_id: int):
        link_name = base_verb.upper()
        results = await self.driver.query(f"""
        MATCH (nsubj:Noun {{text_block_id: $text_block_id}}), (obj:Noun {{text_block_id: $text_block_id}})
        WHERE $nsubj IN nsubj.forms AND $obj IN obj.forms
        MERGE (nsubj)-[r:{link_name}]->(obj)
        RETURN r
        """, {
            "nsubj": nsubj, "obj": obj, "text_block_id": text_block_id,
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
        MATCH (paragraph:Paragraph {text_block_id: $paragraph_id}), (sentence:Sentence {text_block_id: $sentence_id})
        MERGE (paragraph)-[r:CONTAINS]->(sentence)
        RETURN r
        """, {
            "paragraph_id": paragraph_id, "sentence_id": sentence_id
        })
        logger.info(f"sentence/paragraph link: {results}")
        return results

    async def link_pronoun_to_sentence(self, pronoun: str, link_name: str, text_block_id: int):
        link_name = link_name.upper()
        results = await self.driver.query(f"""
        MATCH (pronoun:Pronoun {{text: $pronoun, text_block_id: $text_block_id}}), (sentence:Sentence {{text_block_id: $text_block_id}})
        MERGE (pronoun)-[r:{link_name} {{text_block_id: $text_block_id}}]->(sentence)
        RETURN r
        """, {
            "pronoun": pronoun, "text_block_id": text_block_id
        })
        if results:
            logger.info(f"MERGE (pronoun:Pronoun {{text: {pronoun}}})-[r:{link_name}]->(sentence:Sentence {{text_block_id: {text_block_id}}})")
        else:
            logger.error(f"failed to link pronoun '{pronoun}' to text_block_id {text_block_id}")
        return results

    async def link_reactive_sentence_to_chat_request(self, chat_request_received_id: int, text_block_id: int):
        results = await self.driver.query("""
        MATCH (req:ChatRequest {chat_request_received_id: $chat_request_received_id}), (sentence:Sentence {text_block_id: $text_block_id})
        MERGE (sentence)-[r:REACTS_TO]->(req)
        RETURN r
        """, {
            "chat_request_received_id": chat_request_received_id, "text_block_id": text_block_id
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

    async def link_successive_sentence(self, previous_text_block_id: int, text_block_id: int):
        results = await self.driver.query("""
        MATCH (prev:Sentence {text_block_id: $previous_text_block_id}), (this:Sentence {text_block_id: $text_block_id})
        MERGE (prev)-[r:PRECEDES]->(this)
        RETURN r
        """, {
            "previous_text_block_id": previous_text_block_id, "text_block_id": text_block_id,

        })
        logger.info(f"sentence/previous link: {results}")
        return results

    async def link_verb_form_to_sentence(self, verb: str, form: str, link_name: str, text_block_id: int):
        link_name = link_name.upper()
        results = await self.driver.query(f"""
        MATCH (verb:Verb {{text: $verb, text_block_id: $text_block_id}}), (sentence:Sentence {{text_block_id: $text_block_id}})
        MERGE (verb)-[r:{link_name} {{form: $form, text_block_id: $text_block_id}}]->(sentence)
        RETURN r
        """, {
            "verb": verb, "form": form, "text_block_id": text_block_id
        })
        if results:
            logger.info(f"MERGE (verb:Verb {{text: {verb}}})-[r:{link_name} {{form: {form}}}]->(sentence:Sentence {{text_block_id: {text_block_id}}})")
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
