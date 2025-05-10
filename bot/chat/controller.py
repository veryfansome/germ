from datetime import datetime
from starlette.concurrency import run_in_threadpool
import aiohttp
import asyncio
import faiss
import logging
import numpy as np
import tiktoken

from bot.api.models import ChatRequest, ChatResponse
from bot.chat import async_openai_client
from bot.graph.control_plane import ControlPlane
from bot.lang.parsers import extract_markdown_page_elements, get_html_soup, strip_html_elements
from bot.lang.patterns import naive_sentence_end_pattern
from bot.lang.tf_idf import get_tf_idf_keywords
from bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler, WebSocketSendEventHandler,
                           WebSocketSender, WebSocketSessionMonitor)
from observability.annotations import measure_exec_seconds
from settings import germ_settings

logger = logging.getLogger(__name__)


class VectorMeta:
    def __init__(self, content: str, token_size: int, vector_id: int, vector, emotions, keywords):
        self.content = content
        self.emotions = emotions
        self.keywords = keywords
        self.token_size = token_size
        self.vector = vector
        self.vector_id = vector_id


class ChatController(WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                     WebSocketSendEventHandler, WebSocketSessionMonitor):
    def __init__(
            self, control_plane: ControlPlane,
            delegate: WebSocketReceiveEventHandler,
            # Based on embeddings model
            embedding_dimensions: int = 3072,  # text-embedding-3-large can be shortened to 256
            token_model: str = "text-embedding-3-large",
            truncation_threshold: int = 8191,
    ):
        self.control_plane = control_plane
        self.delegate = delegate
        self.embedding_dimensions: int = embedding_dimensions
        self.token_encoder = tiktoken.encoding_for_model(token_model)
        self.truncation_threshold = truncation_threshold

        # TODO:
        #   - The current indexes work more like long term memory where vectors from all conversations are stored
        #     together.
        #   - Maybe what's called for is per conversation vector spaces that are thrown away once the conversation
        #     ends. For example, while a conversation is happening, and it is within the truncation window, we let it
        #     stay within the context window of the LLM. Once we exceed the truncation window, we overflow messages
        #     into a faiss, which allows us to recall extended relevant contexts from longer conversations. Once the
        #     conversation ends, we roll all messages into the faiss and do additional tagging, clustering, etc. and
        #     decides what should go into long-term memory. Once this is done, we throw the faiss away.
        #   - Store vector anchors that correlate to various classification labels. When processing new vectors from
        #     conversation indexes, we can compare new entries to these persistent anchors. Overtime, we should be able
        #     to learn new anchors. This allows us to perform classification tasks without relying on static models but
        #     on accumulated memory, thereby learning a larger variety of tasks over time.
        #   - Train a small model to predict the next vector when encountering text. If the next vector is close to our
        #     prediction, then it is not unexpected. If the prediction is unexpected, then we may want to spend energy
        #     processing it. This models human implicit memory, which learns patterns and draws our attention to the
        #     unexpected, which tends to be more interesting.
        self.faiss_assistant_message = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dimensions))
        self.faiss_user_message = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dimensions))

        self.conversations: dict[int, dict] = {}
        self.sig_to_conversation_id: dict[str, {}] = {}
        self.sig_to_vector: dict[str, VectorMeta] = {}
        self.vector_id_to_sig: dict[int, str] = {}

    async def faiss_vector_search(self, faiss_idx, query_meta: VectorMeta, current_conversation_id: int,
                                  num_results: int, min_sim_score: float):
        results = []
        sim_scores, neighbors = await run_in_threadpool(faiss_idx.search, query_meta.vector, num_results)
        for rank, (vector_id, sim_score) in enumerate(zip(neighbors[0], sim_scores[0]), 1):
            if vector_id != -1 and sim_score > min_sim_score:  # -1 means no match
                text_sig = self.vector_id_to_sig[vector_id]
                if current_conversation_id not in self.sig_to_conversation_id[text_sig]:
                    result_meta = self.sig_to_vector[text_sig]
                    logger.info(f"{rank:>2}. vector_id={vector_id} sim={sim_score:.4f} "
                                f"keywords={result_meta.keywords} text={result_meta.content}")
                    results.append((sim_score, results))
        return results

    async def get_text_embedding_vector(self, text: str):
        response = await async_openai_client.embeddings.create(
            dimensions=self.embedding_dimensions,
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        vector = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(vector)  # Important for cosine search
        return vector

    async def index_faiss_vector(self, func, text_sig: str):
        await run_in_threadpool(
            func, self.sig_to_vector[text_sig].vector,
            np.array([self.sig_to_vector[text_sig].vector_id], dtype=np.int64)
        )

    async def on_disconnect(self, conversation_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime, text_sig: str,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "token_size": 0,
                "messages": {},
            }
        dt_created_ts = int(dt_created.timestamp())
        self.conversations[conversation_id]["messages"][dt_created_ts] = {
            "user_id": user_id, "text_sig": text_sig
        }

        newest_message_content = chat_request.messages[-1].content
        if text_sig not in self.sig_to_vector:
            code_blocks, text_chunks, sentence_chunks = await process_markdown_elements(
                await run_in_threadpool(extract_markdown_page_elements, newest_message_content)
            )
            embedding_vector_task = asyncio.create_task(self.get_text_embedding_vector(newest_message_content))
            emo_labeling_task = asyncio.create_task(get_emotions_classifications(sentence_chunks))
            tf_idf_task = asyncio.create_task(
                run_in_threadpool(get_tf_idf_keywords, [" ".join(sentence_chunks)], top=5)
            )

            token_size = len(self.token_encoder.encode(newest_message_content))
            self.sig_to_conversation_id[text_sig] = {conversation_id}
            # Set based on first encountered conversation
            vector_id = convert_conversation_id_and_ts_to_vector_id(conversation_id, dt_created_ts)
            self.vector_id_to_sig[vector_id] = text_sig

            embedding_vector, emo_labels, tf_id_keywords = await asyncio.gather(*[
                embedding_vector_task, emo_labeling_task, tf_idf_task
            ])
            self.sig_to_vector[text_sig] = meta = VectorMeta(
                content=newest_message_content, token_size=token_size,
                vector_id=vector_id, vector=embedding_vector,
                emotions=self.score_emotions(emo_labels, token_size), keywords=tf_id_keywords[0]["keywords"]
            )
        else:
            meta = self.sig_to_vector[text_sig]
            self.sig_to_conversation_id[text_sig].add(conversation_id)

        self.conversations[conversation_id]["token_size"] += meta.token_size
        logger.info(f"tokens: {self.conversations[conversation_id]['token_size']}")
        logger.info(f"emotions: {meta.emotions}")
        logger.info(f"keywords: {meta.keywords}")

        assistant_idx_result, user_idx_results = await asyncio.gather(*[
            self.faiss_vector_search(
                self.faiss_assistant_message, meta, conversation_id, 4, 0.35
            ),
            self.faiss_vector_search(
                self.faiss_user_message, meta, conversation_id, 4, 0.7
            ),
        ])

        # Send to LLM
        await self.delegate.on_receive(user_id, conversation_id, dt_created, text_sig, chat_request, ws_sender)

    async def on_send(self, conversation_id: int, dt_created: datetime, text_sig: str,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        received_message_ts = int(received_message_dt_created.timestamp())
        received_message_sig = self.conversations[conversation_id]["messages"][received_message_ts]["text_sig"]
        dt_created_ts = int(dt_created.timestamp())
        self.conversations[conversation_id]["messages"][dt_created_ts] = {
            "user_id": 0, "text_sig": text_sig
        }

        if text_sig not in self.sig_to_vector:
            code_blocks, text_chunks, sentence_chunks = await process_markdown_elements(
                await run_in_threadpool(extract_markdown_page_elements, chat_response.content)
            )
            embedding_vector_task = asyncio.create_task(self.get_text_embedding_vector(chat_response.content))
            tf_idf_task = asyncio.create_task(
                run_in_threadpool(get_tf_idf_keywords, [" ".join(sentence_chunks)], top=5)
            )

            token_size = len(self.token_encoder.encode(chat_response.content))
            self.sig_to_conversation_id[text_sig] = {conversation_id}
            # Set based on first encountered conversation
            vector_id = convert_conversation_id_and_ts_to_vector_id(conversation_id, dt_created_ts)
            self.vector_id_to_sig[vector_id] = text_sig

            embedding_vector, tf_id_keywords = await asyncio.gather(*[
                embedding_vector_task, tf_idf_task
            ])
            self.sig_to_vector[text_sig] = meta = VectorMeta(
                content=chat_response.content, token_size=token_size,
                vector_id=vector_id, vector=embedding_vector,
                emotions=None, keywords=tf_id_keywords
            )
        else:
            meta = self.sig_to_vector[text_sig]
            self.sig_to_conversation_id[text_sig].add(conversation_id)

        self.conversations[conversation_id]["token_size"] += meta.token_size
        logger.info(f"tokens: {self.conversations[conversation_id]['token_size']}")
        logger.info(f"keywords: {meta.keywords}")

        # NOTES:
        # - Similar questions often recall better than responses and questions
        # - "tell me a joke" and "give me a side-slapper" mean the same thing, but they don't recall each other
        # - Even though ^ doesn't recall on the question, the responses are the same, and maybe we can link the
        #   questions based on close similarity of the responses
        # - Messages with a lot of links might be more valuable just like messages that gets recalled often. Maybe we
        #   can generate per sentence embeddings for these high value messages in cases there are higher value chunks
        #   in them.
        # - If we generate per sentence embeddings and link all sentences together in order, recalling any chunk would
        #   allow us to piece together an episode. This seems more like human episodic memory where an episode is not
        #   recalled in whole but in chunks.

        await asyncio.gather(*[
            self.index_faiss_vector(self.faiss_assistant_message.add_with_ids, text_sig),
            self.index_faiss_vector(self.faiss_user_message.add_with_ids, received_message_sig),
        ])

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        logger.info(f"conversation {conversation_id} is still active")

    def score_emotions(self, emo_labels, total_tokens: int):
        scores = {}
        for labels in emo_labels:
            text_length = len(self.token_encoder.encode(labels["text"]))
            for emo in labels["emotions"]:
                if emo != "neutral":
                    scores[emo] = scores.get(emo, 0) + text_length
        return {k: v / total_tokens for k, v in scores.items()}


def convert_conversation_id_and_ts_to_vector_id(int_id: int, unix_ts: int) -> int:
    # 32-bit bitwise shifting - space efficient, invertible, non-commutative
    return ((int_id & 0xFFFFFFFF) << 32) | (unix_ts & 0xFFFFFFFF)


def convert_vector_id_to_conversation_id_and_ts(vector_id: int) -> tuple[int, int]:
    int_id = vector_id >> 32
    unix_ts = vector_id & 0xFFFFFFFF
    return int_id, unix_ts


async def get_emotions_classifications(texts: list[str]):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/classification/emotions",
                                json={"texts": texts}) as response:
            return await response.json()


async def process_markdown_elements(markdown_elements):
    code_blocks = []
    text_chunks = []
    sentence_chunks = []
    for element in markdown_elements:
        if element[0] in {"heading", "list_item", "paragraph"}:
            p_soup = await run_in_threadpool(get_html_soup, f"<p>{element[1]}</p>")
            p_text, p_elements = await strip_html_elements(p_soup, "p")

            text_chunks.append(p_text)
            text_chunk = p_text
            while text_chunk:
                # Not always perfect but good enough
                sentence_end_match = await run_in_threadpool(
                    naive_sentence_end_pattern.search, text_chunk, 0, len(text_chunk))
                if sentence_end_match:
                    sentence_chunks.append(text_chunk[:sentence_end_match.end()])
                    text_chunk = text_chunk[sentence_end_match.end():].strip()
                else:
                    sentence_chunks.append(text_chunk.strip())
                    text_chunk = ""
        elif element[0] == "block_code":
            code_blocks.append({"content": element[2], "language": element[1]})
        else:
            logger.info(f"skipped element type: {element[0]}")
    return code_blocks, text_chunks, sentence_chunks
