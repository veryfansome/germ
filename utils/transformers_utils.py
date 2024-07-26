from transformers import (AlbertForSequenceClassification, AlbertTokenizer,
                          BertForSequenceClassification, BertTokenizer,
                          DistilBertForSequenceClassification, DistilBertTokenizer)
import os

BERT_MODEL_NAME = os.getenv('BERT_MODEL_NAME', 'bert-base-uncased')
MODEL_DIR = os.getenv("MODEL_DIR", "/src/data/germ")

bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


def route_transformers_cls(model_name: str):
    model_cls = BertForSequenceClassification
    tokenizer_cls = BertTokenizer
    if model_name.startswith("albert-"):
        model_cls = AlbertForSequenceClassification
        tokenizer_cls = AlbertTokenizer
    elif model_name.startswith("distilbert-"):
        model_cls = DistilBertForSequenceClassification
        tokenizer_cls = DistilBertTokenizer
    return model_cls, tokenizer_cls


def new_tokenizer(*args, **kwargs):
    model_name = BERT_MODEL_NAME if not args else args[0]
    _, tokenizer_cls = route_transformers_cls(model_name)
    return tokenizer_cls.from_pretrained(
        *((BERT_MODEL_NAME, ) if not args else args), **kwargs)


def new_classification_model(*args, **kwargs):
    model_name = BERT_MODEL_NAME if not args else args[0]
    model_cls, _ = route_transformers_cls(model_name)
    return model_cls.from_pretrained(
        *((BERT_MODEL_NAME, ) if not args else args), **kwargs)
