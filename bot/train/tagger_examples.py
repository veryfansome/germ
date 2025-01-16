from googletrans import Translator
from openai import AsyncOpenAI
from starlette.concurrency import run_in_threadpool
from uuid import uuid5
import asyncio
import difflib
import logging
import random
import re

from bot.lang.pos import get_pos_tags, pos_tagger_info
from settings.germ_settings import UUID5_NS
from settings.openai_settings import DEFAULT_MINI_MODEL, HTTPX_TIMEOUT

logger = logging.getLogger(__name__)

differ = difflib.Differ()
example_cache = {}
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
formatted_tag_dictionary_blob = '\n'.join(pos_tagger_info['tag_dictionary'].get_items())
translator = Translator()


async def basic_gpt_prompt(prompt: str, text: str,
                           add_distillation: bool = False, add_translations: bool = False, label: str = None):
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
        if await cache_example(completion.choices[0].message.content):
            logger.info(f"{completion.choices[0].message.content} [{label}]" if label
                        else completion.choices[0].message.content)
            if add_distillation:
                _ = asyncio.create_task(
                    distill_text(completion.choices[0].message.content))
            if add_translations:
                _ = asyncio.create_task(
                    create_text_variations_using_translation(completion.choices[0].message.content))


async def cache_example(text: str) -> bool:
    signature = uuid5(UUID5_NS, text)
    if signature not in example_cache:  # Skips exact dupes
        example_cache[signature] = True  # Create placeholder

        async def task():
            # Overwrite placeholder
            pos_tags = await run_in_threadpool(get_pos_tags, text)
            example = "\n".join([f"{txt} {tag}" for txt, tag in pos_tags])
            async with AsyncOpenAI() as client:
                completion = await client.chat.completions.create(
                    messages=([
                        {
                            "role": "system",
                            "content": f"""
                            *Instruction*:
                            - Change flair part-of-speech tags in column 2, to the corrected value if incorrect.
                            - Don't change anything from column 1.
                            *Part-of-speech classes*:
                            {formatted_tag_dictionary_blob}"
                            """
                        },
                        {
                            "role": "user",
                            "content": example,
                        }
                    ]),
                    model=DEFAULT_MINI_MODEL, n=1, temperature=0.0,
                    timeout=HTTPX_TIMEOUT)
                returned_example = re.sub(r"[\s\t]+$", "", completion.choices[0].message.content.strip())
                foo = example.splitlines()
                bar = returned_example.splitlines()
                logger.info(f"foo: {foo}")
                logger.info(f"bar: {bar}")
                if example != returned_example:
                    diff_blob = "\n".join([line for line in difflib.context_diff(
                        foo, bar, lineterm='')])
                    logger.info(f"\n>>>diff: {diff_blob}<<<")
                    example_cache[signature] = returned_example
                else:
                    logger.info(f"caching uncontested example")
                    example_cache[signature] = example

        _ = asyncio.create_task(task())
        return True
    else:
        return False


async def change_nouns_and_verbs(text: str, add_distillation: bool = False, add_translations: bool = False):
    """
    Change nouns and verbs without change meaning.
    """
    await basic_gpt_prompt("Change nouns and verbs, without changing the meaning", text,
                           add_distillation=add_distillation, add_translations=add_translations,
                           label="new nouns and verbs")


async def change_sentence_structure(text: str, add_distillation: bool = False, add_translations: bool = False):
    """
    Completely change sentence structure without change meaning.
    """
    await basic_gpt_prompt("Completely change the sentence structure, without changing the meaning", text,
                           add_distillation=add_distillation, add_translations=add_translations,
                           label="structure change")


async def create_text_variations_using_translation(text: str,
                                                   add_distillation: bool = False,
                                                   cache_intermediates: bool = False):
    """
    Use chained translations to generate variations, without altering meaning.
    """
    filter_order = random.sample(list(filter_languages), random.randint(3, len(filter_languages)))
    random.shuffle(filter_order)

    filtered_text = None
    text_to_filter = text
    for filter_lang in filter_order:
        filtered_text, foreign_trans = await filter_text_using_translation(text_to_filter, filter_lang)
        if cache_intermediates and await cache_example(filtered_text):
            logger.info(f"{filtered_text} [{filter_lang}]")
            if add_distillation:
                _ = asyncio.create_task(distill_text(filtered_text))
        text_to_filter = foreign_trans.text  # Play rumors with different languages
    if not cache_intermediates:  # We already cached this if we're caching intermediates
        if await cache_example(filtered_text):
            logger.info(f"{filtered_text} {filter_order}")
            if add_distillation:
                _ = asyncio.create_task(distill_text(filtered_text))


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
                # We cut from the first half because there is a butterfly effect here. Changing the beginning of a
                # sentence increases the likelihood of generating a more varied overall result.
                "content": f"Cut {len_to_cut} words from the beginning, without changing the meaning: {text}",
            }]),
            model=DEFAULT_MINI_MODEL, n=1,
            timeout=HTTPX_TIMEOUT)
        if await cache_example(completion.choices[0].message.content):
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


async def main(txt_file: str, training_split_percentage: float = 0.8):
    # TODO: - Create a controller that waits for the creation of noun nodes or just works periodically off a ranked
    #         list of choices to prioritize
    #       - On creation of nodes or vertexes, generate additional examples of this noun used in sentences
    #       - Generate new examples of this noun in proper noun form if applicable
    #       - Update taggers with new examples
    #       - Persist sentences and tags in PG for later use - even re-tagging possibly as the model adapts so that
    #         older examples can be adjusted
    #       - I may not want to represent sentences and paragraphs as nodes. Instead, maybe sentence and paragraph data
    #         should live in PG and be represented in the graph as vertexes only. When I read something, it leaves an
    #         impression of the idea in my mind, but I generally don't remember the exact phrasing after some time
    #         even though the impression allows me to recall the general ideas later. This is more efficient storage
    #         wise - neo4j should store impressions of ideas, not texts.

    logger.info("augmenting loaded data")
    with open(txt_file) as fd:
        for line in fd:
            line = line.strip()
            logger.info(f"{line} [original]")
            await cache_example(line)
            # Augment example diversification
            await asyncio.gather(*[
                create_text_variations_using_translation(line, add_distillation=True),
                # TODO: convert nouns and verbs to different forms or tenses.
                change_nouns_and_verbs(line, add_distillation=True),
                change_sentence_structure(line, add_distillation=True),
                distill_text(line),  # Maybe distill should happen at the end of each of the above?
            ])
    candidates = list(example_cache.values())
    training_stop_idx = int(len(candidates) * training_split_percentage)
    with open(f"{txt_file}_train.txt", "w") as fd:
        fd.writelines(candidates[:training_stop_idx])
    test_stop_idx = training_stop_idx + int(len(candidates) * ((1 - training_split_percentage) / 2))
    with open(f"{txt_file}_test.txt", "w") as fd:
        fd.writelines(candidates[training_stop_idx:test_stop_idx])
    with open(f"{txt_file}_dev.txt", "w") as fd:
        fd.writelines(candidates[test_stop_idx:])


if __name__ == '__main__':
    from observability.logging import setup_logging

    setup_logging()
    asyncio.run(main("/src/data/germ/pos/ipaddr.txt"))
