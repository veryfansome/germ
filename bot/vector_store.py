import faiss
import numpy as np

from bot.text_embeddings import get_text_embedding


class OpenAITextEmbedding3SmallDim1536:
    def __init__(self):
        self.model_name = "text-embedding-3-small"
        self.model_dimension = 1536
        self.index = faiss.IndexFlatL2(self.model_dimension)

    def add(self, message: str):
        embedding = get_text_embedding(message, self.model_name)
        embedding_np = np.array(embedding).astype('float32').reshape(1, -1)
        self.index.add(embedding_np)
