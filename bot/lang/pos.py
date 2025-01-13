from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric
import logging
import os

logger = logging.getLogger(__name__)

columns = {0: 'text', 1: 'pos'}
data_folder = '/src/data/germ/pos'

# Create the corpus
corpus = ColumnCorpus(data_folder, columns, train_file='train.txt', test_file='test.txt')

# Load the pre-trained POS tagger
tagger = (SequenceTagger.load("pos") if not os.path.isdir("/src/models/germ/pos")
          else SequenceTagger.load("/src/models/germ/pos"))
logger.info(f"{tagger.tag_type}")

# Initialize the trainer
trainer = ModelTrainer(tagger, corpus)

# Start training
trainer.train('/src/models/germ/pos',
              # Reduce learning_rate to 0.001 even if there are signs of overfitting due to small num of examples.
              learning_rate=0.01,
              # Small batch for more frequent updates, which may help with generalization with fewer examples.
              mini_batch_size=1,
              # Fewer epochs to prevent overfitting
              max_epochs=3,
              # Use validation set for early stopping when the model starts to overfit
              train_with_dev=True,
              monitor_train=True,
              monitor_test=True,
              embeddings_storage_mode='gpu',  # May need toggling
              evaluation_metric=EvaluationMetric.MICRO_F1_SCORE)
