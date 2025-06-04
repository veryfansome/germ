from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re

nltk.download('brown')
from nltk.corpus import brown

brown_docs = [brown.raw(file_id) for file_id in brown.fileids()]
tf_idf_vectorizer = TfidfVectorizer(stop_words='english')


def clean_brown_text(text):
    # Remove specific quote patterns: `` and ''
    text = text.replace("``", "").replace("''", "")
    # Remove spaces before punctuation like period, comma, question mark, exclamation mark, etc.
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Optional: Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


cleaned_brown_docs = [clean_brown_text(doc) for doc in brown_docs]


def get_tf_idf_keywords(texts: list[str], top: int = 3):
    keywords = []
    corpus = texts + cleaned_brown_docs

    tfidf_matrix = tf_idf_vectorizer.fit_transform(corpus, y=None)
    feature_names = tf_idf_vectorizer.get_feature_names_out()
    for text_idx, text in enumerate(texts):
        doc_scores = tfidf_matrix[text_idx].toarray().flatten()
        nonzero_indices = [i for i, score in enumerate(doc_scores) if score > 0]
        sorted_indices = sorted(nonzero_indices, key=lambda i: doc_scores[i], reverse=True)
        top_words = [feature_names[i] for i in sorted_indices][:top]
        keywords.append({"text": text, "keywords": top_words})
    return keywords
