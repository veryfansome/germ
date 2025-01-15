from googletrans import Translator
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from starlette.concurrency import run_in_threadpool
from uuid import uuid5
import asyncio
import logging
import random

from bot.lang.pos import pos_tagger, train_pos_tagger
from settings.germ_settings import UUID5_NS

logger = logging.getLogger(__name__)

embeddings_cache = {}
filter_languages = {
    # Arabic: uses a root-based system where words are formed from a set of consonants, which is quite different
    # from English
    "ar",
    # Finnish: is agglutinative, meaning it forms words and expresses grammatical relationships through the addition
    # of prefixes and suffixes, unlike English.
    "fi",
    # Hungarian: is agglutinative, meaning it forms words and expresses grammatical relationships through the addition
    # of prefixes and suffixes, unlike English.
    "hu",
    # Japanese: uses Subject-Object-Verb (SOV) order, which is different from the SVO structure of English.
    "ja",
    # Korean: follows an SOV order but has a system of honorifics that affects verb forms and vocabulary based on the
    # social status of the interlocutor.
    "ko",
    # Russian: has a complex case system with six grammatical cases (nominative, accusative, genitive, dative,
    # instrumental, and prepositional) that affect nouns, pronouns, and adjectives.
    "ru",
    # Chinese: uses Subject-Verb-Object (SVO) order like English, but it often omits subjects and uses topic-prominent
    # structures
    "zh-cn",
}
sentence_transformers_model = SentenceTransformer('all-MiniLM-L6-v2')
translator = Translator()


def cache_embeddings(text: str):
    signature = uuid5(UUID5_NS, text)
    if signature not in embeddings_cache:  # Skips exact dupes
        embedding = sentence_transformers_model.encode(text)
        embeddings_cache[signature] = embedding.reshape(1, -1)


async def translate_to_dest_lang(text: str, dest_lang: str, src_lang: str = "en"):
    dest_trans = await translator.translate(text, dest=dest_lang)
    return (await translator.translate(dest_trans.text, dest=src_lang)).text, dest_trans


async def main():
    with open("/src/data/germ/pos/ipaddr.txt") as fd:
        for line in fd:
            line = line.strip()
            logger.info(line)  # English (Germanic)
            initial_embeddings_task = asyncio.create_task(
                run_in_threadpool(cache_embeddings, line))

            # The following languages have been chosen to get a wide variety of grammar patterns
            # TODO: - Shuffle the order.
            #       - Translate intermediate steps to english as well and keep them if there are differences.
            #       - Try my original rumination idea here to get different flavors of similar sentences.
            #       - Use embeddings and cosine similarity to measure similarity?

            filters = list(filter_languages)
            random.shuffle(filters)

            zh_cn_trans = await translator.translate(line, dest='zh-cn')  # -> Chinese (Sino-Tibetan)
            es_trans = await translator.translate(zh_cn_trans.text, dest='es')  # -> Spanish (Romance)
            ja_trans = await translator.translate(es_trans.text, dest='ja')  # -> Japanese (Japonic)
            tr_trans = await translator.translate(ja_trans.text, dest='tr')  # -> Turkish (Turkic)
            en_trans = await translator.translate(tr_trans.text, dest='en')  # -> English (Germanic)
            logger.info(en_trans.text)


if __name__ == '__main__':
    from observability.logging import setup_logging

    setup_logging()
    asyncio.run(main())
