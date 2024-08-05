from starlette.concurrency import run_in_threadpool
from transformers import AutoModel, AutoTokenizer
import os
import torch
import torch.nn as nn
import torch.optim as optim

from observability.annotations import measure_exec_seconds
from observability.logging import logging, setup_logging
from settings import germ_settings

logger = logging.getLogger(__name__)


class BertClassifier(nn.Module):
    """
    A simple neural network for classifying text using a BERT model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.5):
        super(BertClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


class BertClassificationPredictor:
    def __init__(self, classifier_name: str, labels: list[str],
                 bert_model_name: str = germ_settings.DEFAULT_BERT_MODEL,
                 model_dir: str = germ_settings.MODEL_DIR):
        """
        Trains BertClassifier with given model name and labels for making predictions.

        :param classifier_name:
        :param labels:
        :param bert_model_name:
            - `bert-large-cased` is the most memory-intensive with ~1.3 GB.
            - `bert-base-cased` and `roberta-base` are significantly smaller but still require around ~420-480 MB.
            - `distilbert-base-cased` is more compact, requiring ~250-450 MB.
            - `albert-base-v2` is the most memory efficient, requiring ~45-100 MB but is uncased.
        """
        self.bert_classifier = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.labels = labels

        self.model = AutoModel.from_pretrained(bert_model_name)
        self.model_name = bert_model_name

        self.model_embedding_size = self.model.config.hidden_size
        self.name = classifier_name
        self.save_file = f"{model_dir}/{classifier_name}_{bert_model_name}.pth"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.load_from_save_file()

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    def generate_embeddings(self, text):
        text_inputs = self.tokenizer(text,
                                     return_tensors='pt', truncation=True, padding=True,
                                     max_length=self.tokenizer.model_max_length)
        with torch.no_grad():
            text_outputs = self.model(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token
        return text_embeddings

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    def generate_concatenated_embeddings(self, text):
        """
        Tokenize the text and generate embeddings with a sliding window. This approach ensures that we can
        handle long conversations without losing important context due to token truncation. By using a sliding
        window, we process the text in manageable chunks and combine the embeddings to form a comprehensive
        representation. The caveat is that this is computationally heavy. When initially tested, training times
        increased by an order of magnitude.

        :param text:
        :return: combined embeddings
        """
        tokens = self.tokenizer(text, return_tensors='pt', truncation=False, padding=False)['input_ids'][0]
        max_length = self.tokenizer.model_max_length
        stride = 256  # Overlap between windows

        embeddings_list = []
        for idx in range(0, len(tokens), stride):
            window_tokens = tokens[idx:idx + max_length]
            if len(window_tokens) < max_length:
                padding_length = max_length - len(window_tokens)
                window_tokens = torch.cat([
                    window_tokens,
                    torch.zeros(max_length - len(window_tokens), dtype=torch.long)
                ])
                attention_mask = torch.cat([torch.ones(len(window_tokens) - padding_length), torch.zeros(padding_length)])
            else:
                attention_mask = torch.ones(max_length)
            inputs = {'input_ids': window_tokens.unsqueeze(0), 'attention_mask': attention_mask.unsqueeze(0)}
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
            embeddings_list.append(embeddings)
        return torch.cat(embeddings_list, dim=0)

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    def load_from_save_file(self):
        """
        Load the saved model and tokenizer from disk into new classifier and optimizer and then
        replace the existing one. This ensures we can continue to serve predictions while periodically
        refreshing models from disk.

        :return:
        """
        bert_classifier, optimizer = self.new_bert_classifier()
        load_from_save_file(bert_classifier, optimizer, self.save_file)
        self.bert_classifier = bert_classifier
        self.optimizer = optimizer

    async def load_from_save_file_runnable(self):
        await run_in_threadpool(self.load_from_save_file)

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    def new_bert_classifier(self):
        bert_classifier = BertClassifier(
            self.model_embedding_size,  # input_dim
            128,  # hidden_dim
            len(self.labels),  # output_dim
            dropout_prob=0.5
        )
        optimizer = optim.Adam(bert_classifier.parameters(), lr=0.001)
        return bert_classifier, optimizer

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    def predict_label(self, embeddings):
        self.bert_classifier.eval()
        with torch.no_grad():
            outputs = self.bert_classifier(embeddings)
        _, predicted = torch.max(outputs, 1)
        return self.labels[predicted.item()]

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    def save(self, save_file: str = None):
        if save_file is None:
            save_file = self.save_file
        torch.save({
            'bert_classifier_state_dict': self.bert_classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_file)

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    def train_bert_classifier(self, label: str, embeddings):
        self.bert_classifier.train()
        self.optimizer.zero_grad()
        outputs = self.bert_classifier(embeddings)
        loss = self.criterion(outputs, torch.tensor([self.labels.index(label)], dtype=torch.long))
        loss.backward()
        self.optimizer.step()


def load_from_save_file(bert_classifier: BertClassifier, optimizer: optim.Adam, save_file: str):
    if os.path.exists(save_file):
        checkpoint = torch.load(save_file)
        bert_classifier.load_state_dict(checkpoint['bert_classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        bert_classifier.eval()  # Set the model to evaluation mode
    return bert_classifier, optimizer


def new_activation_predictor(classifier_name: str, model_dir: str = None):
    labels = ["off", "on"]
    if model_dir:
        return BertClassificationPredictor(classifier_name, labels, model_dir=model_dir)
    return BertClassificationPredictor(classifier_name, labels)


if __name__ == "__main__":
    import random
    import time

    setup_logging()

    go_no_go_labels = ["go", "don't go"]
    predictor = BertClassificationPredictor('go-no-go', go_no_go_labels)

    tests = [
        {"label": "go",
         "text": "You should go.",
         "embeddings": predictor.generate_embeddings("You should go.")},
        {"label": "don't go",
         "text": "You should not go.",
         "embeddings": predictor.generate_embeddings("You should not go.")},
    ]
    examples = tests * 10  # Ok to have dupes
    for i in range(10):
        logger.info(f"round {i}")
        time_round_started = time.time()
        random.shuffle(examples)
        for exp in examples:
            predictor.train_bert_classifier(exp['label'], exp['embeddings'])
        logger.info(f"finished round {i} after {time.time() - time_round_started}s")
    tests = tests * 10
    random.shuffle(tests)
    for exp in tests:
        assert exp['label'] == predictor.predict_label(exp['embeddings'])
