import openai


def get_text_embedding(text, model="text-embedding-3-small"):
    response = openai.Embedding.create(input=text, model=model)
    return response.data[0].embedding
