# Dead code. Left for reference.

from transformers import BertTokenizer, BertModel
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset

from ml.message_categorizer import INTENT_CLASSIFICATION_EXAMPLES, LINGUISTIC_ELEMENTS, extract_message_features
from observability.logging import logging, setup_logging
from utils.openai_utils import DEFAULT_CHAT_MODEL

logger = logging.getLogger(__name__)

model_dir = os.getenv("MODEL_DIR", "/src/data/germ")
corpus_file = f"{model_dir}/disciple_message_intent_categorizer_corpus.json"
save_file = f"{model_dir}/disciple_message_intent_categorizer.pth"


# Define a simple Seq2Seq model with attention
class IntentSeq2Seq(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim):
        super(IntentSeq2Seq, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg):
        embedded_src = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded_src)
        embedded_trg = self.embedding(trg)
        outputs, _ = self.decoder(embedded_trg, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions


# Custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_tokens = self.vocab(self.tokenizer(text))
        label_tokens = self.vocab(self.tokenizer(label))
        return torch.tensor(text_tokens), torch.tensor(label_tokens)


if __name__ == '__main__':
    from openai import OpenAI
    import json
    import re

    setup_logging()
    logger.info("initializing disciple_message_intent_categorizer")

    corpus = None
    if not os.path.exists(corpus_file):
        corpus = [
            # Initial examples for the model
            ["greeting", "Hey"],
            ["greeting", "Hi"],
            ["greeting", "there?"],
            ["greeting", "yo"],
            ["farewell", "bye"],
            ["farewell", "Later"],
            ["farewell", "I'm going to bed"],
            ["farewell", "I'm calling it a day"],
            ["description", "I feel great"],
            ["description", "I'm just feeling kinda down right now"],
            ["correction", "no, dall-e-2 was the model you should have used to respond"],
            ["question, description", ' '.join((
                "in my version of scikit-learn, I see this warning `Unresolved attribute reference 'labels_' for class",
                "'object'` on `print(clustering.labels_)` from your example of how to do `Dynamic Category Handling`",
            ))],
            ["greeting", "Hi!"],
            ["farewell", "I'm going to bed"],
            ["correction", "You used the wrong model to respond. you should have used gpt-4o"],
            ["request", "Generate an image of a cat"],
            ["question", "Who was the protagonist in 'To Kill a Mockingbird'?"],
            ["question, description", ' '.join((
                "when I ran this code, I got `[-1 -1 -1 -1 -1 -1]`, which means everything was considered noise,",
                "correct? this code that uses `sklearn` seems completely disconnected from the categories used with",
                "openai",
            ))],
            ["request, description", ' '.join((
                "I want to use the categories as input features to train ML models. In this example, there are two",
                "binary categories. This is too simple. Let's say I want to do intent classification. I might start",
                "with a small list of intent categories but over time, I would want to identify new categories so this",
                "list might grow. how I can account for that in an ML system?",
            ))],
        ]
    else:
        with open(corpus_file, "r") as fd:
            corpus = json.load(fd)

    desired_corpus_size = 20
    new_example_pattern = r".*Intent: \"([^\"]+)\", Message: \"(.*?)?\"$"
    while len(corpus) < desired_corpus_size:
        base_content = ' '.join((
            "Generate a single chat message example from the perspective of a chatbot user to add to the",
            "**Examples** below. This message should be from the user to the bot.\n\n",

            "The **Examples** will be used for ML training, where the **Intent** will be used as labels.",
            "I want a balanced set of examples so prioritize **Intent** categories that are not already present",
            "or are under-represented as you decide the \"next\" message.",
            "For reference, here are some suggested \"Intent\" categories:",
            ", ".join(INTENT_CLASSIFICATION_EXAMPLES) + ".\n\n",

            "The same idea applies to linguistic patterns. I want balanced examples so prioritize linguistic",
            "elements and patterns that are not already present or under-represented as you decide the \"next\"",
            "message. For reference, here are some suggested linguistic patterns to use:",
            ", ".join(LINGUISTIC_ELEMENTS) + ".\n\n",

            "Users type messages so your language should get the point across efficiently.",
            "Occasional use of common abbreviations is OK.\n\n",
        ))
        system_content = ' '.join((
            #"For the next example, generate a messages that fall under description, question, and/or request",
            #"For the next example, generate a messages that is some combination of complaint, description, question, and/or greeting.",
            #"For the next example, generate a message that is some **combination** of categories other than complaint, confession, description, feedback, question, request, and/or suggestion.",

            #"For the next example, generate a message that requests a simple image without qualifying descriptions.",
            #"Don't use the following words or phrases: \"I have to confess\", or \"I have to admit\".",
            #"Choose a subject not already in **Examples**.",

            #"Simulate a scenario where the user is writing a resume and wants grammar checking.",

            #"Use topical subject matter related to disk space utilization analysis on Linux.",
            #"Never duplicate existing code",
            #"Generate a long form message that includes markdown code snippets both in-line and as blocks.",

        ))
        examples = []
        for exp in corpus:
            examples.append(f"- Intent: \"{exp[0]}\", Message: \"{exp[1]}\"\n")
        content = ''.join((
            base_content,
            "### Examples\n\n",
            *examples,
            "\n",
        ))
        with OpenAI() as client:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": content},
                ],
                model=DEFAULT_CHAT_MODEL, n=1, temperature=0.0,
            )
            new_example = completion.choices[0].message.content.strip()
            logger.info("new example: %s", new_example)
            new_example_match = re.match(new_example_pattern, new_example, re.DOTALL)
            if new_example_match:
                new_example_message = new_example_match.group(2)
                new_example_message_features = extract_message_features(
                    [{"role": "user", "content": new_example_message}])
                corpus.append((new_example_message_features["intent_classification"], new_example_message))

    for i in range(len(corpus)):
        exp = corpus[i]
        exp_message_features = extract_message_features(
            [{"role": "user", "content": exp[1]}])
        if exp[0] != exp_message_features["intent_classification"]:
            logger.warning(
                f"classification changed: {exp[0]}->{exp_message_features['intent_classification']}, message: {exp[1]}"
            )
            corpus[i][0] = exp_message_features['intent_classification']
    logger.info("corpus:\n%s", "\n".join([f"- Intent: {e[0]}, Message: {e[1]}" for e in corpus]))
    with open(corpus_file, "w") as fd:
        json.dump(corpus, fd, indent=2)
    exit()

    examples = []
    example_labels = [generate_classification_text(text) for text in examples]

    example_tokenizer = get_tokenizer("basic_english")
    example_vocab = build_vocab_from_iterator(map(example_tokenizer, corpus + example_labels),
                                              specials=["<unk>", "<pad>"])
    example_vocab.set_default_index(example_vocab["<unk>"])

    # Dataset and DataLoader
    dataset = TextDataset(corpus, example_labels, example_tokenizer, example_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x)
