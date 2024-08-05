from ml.bert_classifier import BertClassificationPredictor
import itertools
import random


def test_bert_classifier():
    verbs = ["look", "move", "sound", "smell"]
    labels = ["bird", "cat", "dog"]
    predictor = BertClassificationPredictor('bird-cat-dog', labels)

    tests = [{
        "label": label,
        "text": f"This {verb}s like a {label}.",
        "embeddings": predictor.generate_embeddings(f"This {verb}s like a {label}.")
    } for verb, label in list(itertools.product(verbs, labels))]
    examples = tests * 3  # Ok to have dupes
    for i in range(50):  # Had been 10 iterations with bert-large-cased
        random.shuffle(examples)
        for exp in examples:
            predictor.train_bert_classifier(exp['label'], exp['embeddings'])
    tests = tests * 3
    random.shuffle(tests)
    for exp in tests:
        predicted_label = predictor.predict_label(exp['embeddings'])
        assert exp['label'] == predicted_label
