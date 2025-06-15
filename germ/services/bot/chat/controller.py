import aiohttp
import asyncio
import faiss
import logging
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from germ.api.models import ChatRequest, ChatResponse
from germ.data.vector_anchors import intent_anchors
from germ.database.neo4j import KnowledgeGraph
from germ.observability.annotations import measure_exec_seconds
from germ.services.bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                                         WebSocketSendEventHandler, WebSocketSender, WebSocketSessionMonitor)
from germ.services.models.predict.multi_predict import log_pos_labels
from germ.settings import germ_settings
from germ.utils.parsers import DocElementType, ParsedDoc, parse_markdown_doc

logger = logging.getLogger(__name__)


class MessageMeta(BaseModel):
    doc: ParsedDoc
    pos: list[dict[str, str | list[str]]]
    text_embs: list[list[float]]


class ChatController(WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                     WebSocketSendEventHandler, WebSocketSessionMonitor):
    def __init__(
            self, knowledge_graph: KnowledgeGraph,
            delegate: WebSocketReceiveEventHandler,
    ):
        self.knowledge_graph = knowledge_graph
        self.delegate = delegate

        self.conversations: dict[int, dict] = {}
        self.faiss_intent: faiss.IndexIDMap | None = None
        self.id_to_intent: dict[int, str] = {i: k for i, k in enumerate(intent_anchors)}
        self.sig_to_conversation_id: dict[str, set] = {}
        self.sig_to_message_meta: dict[str, MessageMeta] = {}

    async def on_disconnect(self, conversation_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime, text_sig: str,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                # Indexes user_id and text_sig by dt_created
                "messages": {},
            }
        self.conversations[conversation_id]["messages"][int(dt_created.timestamp())] = {
            "user_id": user_id, "text_sig": text_sig
        }
        self.sig_to_conversation_id.get(text_sig, set()).add(conversation_id)

        if text_sig not in self.sig_to_message_meta:
            # Parse message into a ParsedMarkdownPage, which includes a document scaffold (with idx pointers to text),
            # list of text blobs, and list of code blobs.
            parsed_message = await run_in_threadpool(
                parse_markdown_doc, chat_request.messages[-1].content
            )
            # Get embeddings and POS labels
            text_embs, pos = await asyncio.gather(*[
                get_text_embedding(parsed_message.text),
                get_pos_labels(parsed_message.text)
            ])
            # Cache results
            self.sig_to_message_meta[text_sig] = meta = MessageMeta(
                doc=parsed_message, pos=pos, text_embs=text_embs["embeddings"]
            )
        else:
            meta = self.sig_to_message_meta[text_sig]

        # Walk through each doc element
        all_hydrated_elements = []
        message_len = len(meta.doc.scaffold)
        for element_idx, scaffold_element in enumerate(meta.doc.scaffold):
            hydrated_headings = {}
            for level, heading in scaffold_element.headings.items():
                hydrated_headings[level] = [(meta.text_embs[idx], meta.pos[idx]) for idx in heading.text]

            hydrated_element = []
            if scaffold_element.type == DocElementType.CODE_BLOCK:
                pass  # TODO
            elif scaffold_element.type == DocElementType.LIST:
                for item_idx, item in enumerate(scaffold_element.items):
                    for text_idx in item.text:
                        hydrated_element.append((
                            meta.text_embs[text_idx], meta.pos[text_idx]
                        ))
            elif scaffold_element.type == DocElementType.PARAGRAPH:
                for text_idx in scaffold_element.text:
                    hydrated_element.append((
                        meta.text_embs[text_idx], meta.pos[text_idx]
                    ))

            # Goals:
            # - Look for intent
            # - Look for patterns and decide response
            #   - Updates to knowledge graph
            #   - Learn new patterns to react to
            #   - Learn new reactions to existing patterns
            # - Instinctive vs learned behavior?
            for emb, pos in hydrated_element:
                exclamatory = "!" in pos["tokens"] and pos["pos"][pos["tokens"].index("!")] == "PUNCT"
                imperative = "Imp" in pos["Mood"]
                indicative = "Ind" in pos["Mood"]
                interrogative = "Int" in pos["Mood"] or (
                        "?" in pos["tokens"] and pos["pos"][pos["tokens"].index("?")] == "PUNCT"
                )
                logger.info(f"Intent signals: Exc:{exclamatory} Imp:{imperative}, "
                            f"Ind:{indicative}, Int:{interrogative}")
                await run_in_threadpool(search_faiss_index, self.faiss_intent, emb, self.id_to_intent)
                #logger.info(f"{pos['text']} >> {emb}")
                log_pos_labels(pos)
                #await self.knowledge_graph.match_synset(tokens=meta.pos[sentence_idx]["tokens"])
                #
            all_hydrated_elements.append((hydrated_headings, hydrated_element))
        # Send to LLM
        await self.delegate.on_receive(user_id, conversation_id, dt_created, text_sig, chat_request, ws_sender)

    async def on_send(self, conversation_id: int, dt_created: datetime, text_sig: str,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        pass

    async def on_start(self):
        embedding_info = await get_text_embedding_info()
        self.faiss_intent = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        for idx, emb in enumerate((await get_text_embedding(intent_anchors))["embeddings"]):
            await run_in_threadpool(index_embedding, self.faiss_intent.add_with_ids, emb, idx)

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        logger.info(f"conversation {conversation_id} is still active")


async def get_pos_labels(texts: list[str]):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/classification/ud",
                                json={"texts": texts}) as response:
            return await response.json()


async def get_text_embedding(texts: list[str]):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/embedding",
                                json={"texts": texts}) as response:
            return await response.json()


async def get_text_embedding_info():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/embedding/info") as response:
            return await response.json()


def index_embedding(index_func, embedding, row_id: int):
    vector = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(vector)
    index_func(vector, np.array([row_id], dtype=np.int64))

def search_faiss_index(index: faiss.IndexIDMap, embedding, id2result: dict[int, str],
                       num_results: int = 3, min_sim_score: float = 0.3):
    vector = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(vector)

    sim_scores, neighbors = index.search(vector, num_results)
    for rank, (vector_id, sim_score) in enumerate(zip(neighbors[0], sim_scores[0]), 1):
        if vector_id != -1 and sim_score > min_sim_score:  # -1 means no match
            logger.info(f"{rank:>2}. vector_id={vector_id} sim={sim_score:.4f} result={id2result[vector_id]}")
    #return results
