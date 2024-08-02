from ml.bert_classifier import BertClassificationPredictor
import itertools
import random


def test_bert_classifier():
    verbs = ["look", "move", "sound", "smell"]
    labels = ["bird", "cat", "dog"]
    tests = [{"label": label, "text": f"This {verb}s like a {label}."} for verb, label in list(itertools.product(verbs, labels))]
    examples = tests * 3  # Ok to have dupes
    predictor = BertClassificationPredictor('bird-cat-dog', labels)
    for i in range(10):
        random.shuffle(examples)
        for exp in examples:
            embeddings = predictor.generate_embeddings(exp['text'])
            predictor.train_bert_classifier(exp['label'], embeddings)
    tests = tests * 3
    random.shuffle(tests)
    for exp in tests:
        embeddings = predictor.generate_embeddings(exp['text'])
        assert exp['label'] == predictor.predict_label(embeddings)
