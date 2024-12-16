import asyncio
import numpy as np
from openai import OpenAI
from starlette.concurrency import run_in_threadpool
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

mini = SentenceTransformer('all-MiniLM-L6-v2')


def text_similarity_mini(text1: str, text2: str):
    return cosine_similarity(
        text_to_2d_mini_embeddings(text1),
        text_to_2d_mini_embeddings(text2))


async def text_similarity_openai(text1: str, text2: str, model="text-embedding-3-large"):
    embedding1, embedding2 = await asyncio.gather(*[
        run_in_threadpool(text_to_openai_embeddings, text1, model=model),
        run_in_threadpool(text_to_openai_embeddings, text2, model=model),
    ])

    def _cosine_similarity():
        return cosine_similarity(np.array(embedding1).reshape(1, -1), np.array(embedding2).reshape(1, -1))
    return await run_in_threadpool(_cosine_similarity)


def text_to_2d_mini_embeddings(text: str):
    embedding = mini.encode(text)
    return embedding.reshape(1, -1)


def text_to_openai_embeddings(text: str, model: str = "text-embedding-3-large"):
    with OpenAI() as client:
        resp = client.embeddings.create(input=text, model=model)
        return resp.data[0].embedding


if __name__ == '__main__':
    test_set = [
        # Similar intent, similar word usage
        ("Let's go eat a banana.", "I want a banana."),

        # Similar intent, moderate variation in word usage
        ("Eat apples. They're good for you.", "An apple a day, keeps the doctor away!"),

        # Same meaning, seemingly unrelated word usages
        ("Ah! my teeth are swimming!", "I need to go pee!"),
        ("Ugh! My dogs are barking!", "My feet hurt."),

        # Similar words, different intent or scene
        ("I can't believe you would do something like that.", "Aw! I can't believe you did that!"),
        ("The dog chased the cat up the tree.", "The cat chased the tree up the dog."),

        # Completely unrelated
        ("I'm an accountant.", "Dogs bark."),
        ("Anger is toxic.", "The Earth is round?"),
    ]
    for text_pair in test_set:
        result = text_similarity_mini(text_pair[0], text_pair[1])
        # result = asyncio.run(text_similarity_openai(text_pair[0], text_pair[1], model="text-embedding-3-large"))
        print(f"text 1: {text_pair[0]}")
        print(f"text 2: {text_pair[1]}")
        print(f"Cosine Similarity: {result[0][0]}")
