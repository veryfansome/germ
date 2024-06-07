import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from bot.text_embeddings import get_text_embedding


def choose_response_model(text_blob, options=('dall-e-3', 'gpt-4o')):
    embedding = get_text_embedding(text_blob)
    # For simplicity, we'll choose randomly here; this should be based on a more sophisticated model.
    return np.random.choice(options)