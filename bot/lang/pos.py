from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import logging
import os

logger = logging.getLogger(__name__)

# Load the pre-trained POS tagger
pos_tagger = (SequenceTagger.load("pos") if not os.path.isdir("/src/models/germ/pos/final-model.pt")
              else SequenceTagger.load("/src/models/germ/pos/final-model.pt"))
pos_tagger_info = {
    "tag_dictionary": pos_tagger.label_dictionary,
    "tag_format": pos_tagger.tag_format
}


def get_pos_tags(text: str):
    sentence = Sentence(text)
    pos_tagger.predict(sentence)
    return [(word.text, word.tag) for word in sentence]


def train_pos_tagger():
    columns = {0: "text", 1: "pos"}

    # Create the corpus
    corpus = ColumnCorpus("/src/data/germ/pos", columns,
                          # TODO: make this configurable
                          train_file="ipaddr.txt_train.txt",
                          test_file="ipaddr.txt_test.txt",
                          dev_file="ipaddr.txt_dev.txt")

    # Initialize the trainer
    trainer = ModelTrainer(pos_tagger, corpus)

    # Start training
    trainer.train("/src/models/germ/pos",
                  # Reduce learning_rate to 0.001 even if there are signs of overfitting due to small num of examples.
                  learning_rate=0.01,
                  # Small batch for more frequent updates, which may help with generalization with fewer examples.
                  mini_batch_size=1,
                  # Fewer epochs to prevent overfitting
                  max_epochs=3,
                  # Use validation set for early stopping when the model starts to overfit
                  train_with_dev=True,
                  # May need toggling
                  embeddings_storage_mode="gpu")


if __name__ == "__main__":
    logger.info(f"{pos_tagger.tag_type}")
    train_pos_tagger()

