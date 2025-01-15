from googletrans import Translator
from openai import AsyncOpenAI
from uuid import uuid5
import asyncio
import logging
import random

from bot.lang.pos import pos_tagger, train_pos_tagger
from settings.germ_settings import UUID5_NS
from settings.openai_settings import DEFAULT_MINI_MODEL, HTTPX_TIMEOUT

logger = logging.getLogger(__name__)

# The following languages have been chosen to get a wide variety of grammar patterns
filter_languages = {
    # Arabic: uses a root-based system where words are formed from a set of consonants
    "ar",
    # German
    "de",
    # Spanish
    "es",
    # Finnish: is agglutinative, meaning it forms words and expresses grammatical relationships through the addition
    # of prefixes and suffixes
    "fi",
    # Hindi: uses SOV word order, gendered nouns, and a complex system of postpositions.
    "hi",
    # Hungarian: is agglutinative, meaning it forms words and expresses grammatical relationships through the addition
    # of prefixes and suffixes
    "hu",
    # Japanese: uses Subject-Object-Verb (SOV) order, which is different from the SVO structure of English.
    "ja",
    # Korean: follows an SOV order but has a system of honorifics that affects verb forms and vocabulary based on the
    # social status of the interlocutor.
    "ko",
    # Russian: has a complex case system with six grammatical cases (nominative, accusative, genitive, dative,
    # instrumental, and prepositional) that affect nouns, pronouns, and adjectives.
    "ru",
    # Chinese: uses Subject-Verb-Object (SVO) order, but often omits subjects and uses topic-prominent structures
    "zh-cn",
}
text_cache = {}
translator = Translator()


async def basic_gpt_prompt(prompt: str, text: str, add_translations: bool = False, label: str = None):
    """
    Use OpenAI's GPT to complete some text prompt.
    """
    async with AsyncOpenAI() as client:
        completion = await client.chat.completions.create(
            messages=([{
                "role": "user",
                "content": f"{prompt}: {text}",
            }]),
            model=DEFAULT_MINI_MODEL, n=1,
            timeout=HTTPX_TIMEOUT)
        if cache_text(completion.choices[0].message.content):
            logger.info(f"{completion.choices[0].message.content} [{label}]" if label
                        else completion.choices[0].message.content)
            if add_translations:
                _ = asyncio.create_task(
                    create_text_variations_using_translation(completion.choices[0].message.content))


def cache_text(text: str) -> bool:
    signature = uuid5(UUID5_NS, text)
    if signature not in text_cache:  # Skips exact dupes
        text_cache[signature] = text
        return True
    else:
        return False


async def change_nouns_and_verbs(text: str, add_translations: bool = False):
    """
    Change nouns and verbs without change meaning.
    """
    await basic_gpt_prompt("Change nouns and verbs, without changing the meaning", text,
                           add_translations=add_translations, label="new nouns and verbs")


async def change_sentence_structure(text: str, add_translations: bool = False):
    """
    Completely change sentence structure without change meaning.
    """
    await basic_gpt_prompt("Completely change the sentence structure, without changing the meaning", text,
                           add_translations=add_translations, label="structure change")


async def create_text_variations_using_translation(text: str, cache_intermediates: bool = False):
    """
    Use chained translations to generate variations, without altering meaning.
    """
    filter_order = random.sample(list(filter_languages), random.randint(3, len(filter_languages)))
    random.shuffle(filter_order)

    filtered_text = None
    text_to_filter = text
    for filter_lang in filter_order:
        filtered_text, foreign_trans = await filter_text_using_translation(text_to_filter, filter_lang)
        if cache_intermediates and cache_text(filtered_text):
            logger.info(f"{filtered_text} [{filter_lang}]")
        text_to_filter = foreign_trans.text  # Play rumors with different languages
    if not cache_intermediates:
        if cache_text(filtered_text):
            logger.info(f"{filtered_text} {filter_order}")


async def distill_text(text: str, count: int = 1, cut_ratio: float = 0.2,
                       stop: int = 1, add_translations: bool = False):
    """
    Shorten, without altering meaning. Set a higher `stop` to do multiple distillations, but things can turn to
    nonsense. Set `add_translations` to True to use translation filters to more variations of distilled text.
    """
    len_to_cut = int(len(text.split()) * cut_ratio)
    async with AsyncOpenAI() as client:
        completion = await client.chat.completions.create(
            messages=([{
                "role": "user",
                "content": f"Rephrase using {len_to_cut} fewer words, without changing the meaning: {text}",
            }]),
            model=DEFAULT_MINI_MODEL, n=1,
            timeout=HTTPX_TIMEOUT)
        if cache_text(completion.choices[0].message.content):
            logger.info(f"{completion.choices[0].message.content} [distilled {count}]")
            if add_translations:
                _ = asyncio.create_task(
                    create_text_variations_using_translation(completion.choices[0].message.content))
        if count < stop:
            _ = asyncio.create_task(
                distill_text(completion.choices[0].message.content, count=count+1, cut_ratio=cut_ratio, stop=stop))


async def filter_text_using_translation(text: str, dest_lang: str, src_lang: str = "en"):
    dest_trans = await translator.translate(text, dest=dest_lang)
    return (await translator.translate(dest_trans.text, dest=src_lang)).text, dest_trans


async def main():
    with open("/src/data/germ/pos/ipaddr.txt") as fd:
        for line in fd:
            line = line.strip()
            logger.info(f"{line} [original]")
            cache_text(line)
            await asyncio.gather(*[
                create_text_variations_using_translation(line),
                change_nouns_and_verbs(line),
                change_sentence_structure(line),
                distill_text(line),
            ])


if __name__ == '__main__':
    from observability.logging import setup_logging

    setup_logging()
    asyncio.run(main())
