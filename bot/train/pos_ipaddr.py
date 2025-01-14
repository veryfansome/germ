from googletrans import Translator
from openai import OpenAI
import asyncio
import logging

from bot.lang.pos import pos_tagger, train_pos_tagger

logger = logging.getLogger(__name__)
translator = Translator()


async def main():
    with open("/src/data/germ/pos/ipaddr.txt") as fd:
        for line in fd:
            line = line.strip()
            logger.info(line)  # English (Germanic)
            # The following languages have been chosen to get a wide variety of grammar patterns
            # TODO: - Shuffle the order.
            #       - Translate intermediate steps to english as well and keep them if there are differences.
            #       - Try my original rumination idea here to get different flavors of similar sentences.
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
