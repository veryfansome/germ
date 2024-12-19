import nltk
from flair.models import SequenceTagger

# Dependencies
nltk.download('punkt')
nltk.download('punkt_tab')

ner_tagger = SequenceTagger.load("ner")
pos_tagger = SequenceTagger.load("pos")
