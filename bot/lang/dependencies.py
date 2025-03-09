import nltk

# Dependencies
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("words")
nltk.download('wordnet')

from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

wordnet_lemmatizer = WordNetLemmatizer()
