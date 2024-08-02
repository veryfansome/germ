from transformers import BertModel, BertTokenizer
import os
import torch
import torch.nn as nn
import torch.optim as optim

from observability.logging import logging, setup_logging
logger = logging.getLogger(__name__)


# Define a simple neural network for classification
class BertClassifier(nn.Module):
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
                 bert_model_name: str = "bert-large-cased"):
        model_dir = os.getenv("MODEL_DIR", "/src/data/germ")

        self.criterion = nn.CrossEntropyLoss()
        self.labels = labels
        self.model = BertModel.from_pretrained(bert_model_name)
        self.model_name = bert_model_name
        self.model_embedding_size = self.model.config.hidden_size
        self.name = classifier_name
        self.save_file = f"{model_dir}/{classifier_name}_{bert_model_name}.pth"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        self.bert_classifier = BertClassifier(
            self.model_embedding_size,  # input_dim
            128,  # hidden_dim
            len(labels),  # output_dim
            dropout_prob=0.5
        )
        self.optimizer = optim.Adam(self.bert_classifier.parameters(), lr=0.001)

    def generate_embeddings(self, text):
        text_inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            text_outputs = self.model(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token
        return text_embeddings

    def load(self, save_file: str = None):
        if save_file is None:
            save_file = self.save_file
        if os.path.exists(save_file):
            checkpoint = torch.load(save_file)
            self.bert_classifier.load_state_dict(checkpoint['bert_classifier_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.bert_classifier.eval()  # Set the model to evaluation mode

    def predict_label(self, embeddings):
        self.bert_classifier.eval()
        with torch.no_grad():
            outputs = self.bert_classifier(embeddings)
        _, predicted = torch.max(outputs, 1)
        return self.labels[predicted.item()]

    def save(self, save_file: str = None):
        if save_file is None:
            save_file = self.save_file
        torch.save({
            'bert_classifier_state_dict': self.bert_classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_file)

    def train_bert_classifier(self, label: str, embeddings):
        self.bert_classifier.train()
        self.optimizer.zero_grad()
        outputs = self.bert_classifier(embeddings)
        loss = self.criterion(outputs, torch.tensor([self.labels.index(label)], dtype=torch.long))
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    import random
    import time

    setup_logging()

    labels = ["go", "don't go"]
    tests = [
        {"label": "go", "text": "You should go."},
        {"label": "don't go", "text": "You should not go."},
    ]
    examples = tests * 10  # Ok to have dupes
    predictor = BertClassificationPredictor('go-no-go', labels)
    for i in range(10):
        logger.info(f"round {i}")
        time_round_started = time.time()
        random.shuffle(examples)
        for exp in examples:
            embeddings = predictor.generate_embeddings(exp['text'])
            predictor.train_bert_classifier(exp['label'], embeddings)
        logger.info(f"finished round {i} after {time.time() - time_round_started}s")
    tests = tests * 10
    random.shuffle(tests)
    for exp in tests:
        embeddings = predictor.generate_embeddings(exp['text'])
        assert exp['label'] == predictor.predict_label(embeddings)
