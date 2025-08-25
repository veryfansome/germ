import aiohttp

from germ.settings import germ_settings


async def get_text_embedding(texts: list[str], prompt: str = "query: "):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.GERM_MODEL_SERVICE_ENDPOINT}/text/embedding",
                                json={"texts": texts, "prompt": prompt}) as response:
            return (await response.json())["embeddings"]