from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AutoPosTuner:

    tag2id = {
        "NN": 0,
        "VBD": 1,
        "DT": 2,
        "IN": 3,
        "JJ": 4,
        # etc...
    }
    id2tag = {v: k for k, v in tag2id.items()}

    def __init__(self, model, tokenizer,
                 checkpoint_dir="models/germ",
                 id2tag=None, tag2id=None):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.tokenizer = tokenizer

        self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        self.id2tag = AutoPosTuner.id2tag if id2tag is None else id2tag
        self.tag2id = AutoPosTuner.tag2id if tag2id is None else tag2id

    def new_trainer(self, dataset, checkpoint_name):
        return Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=f"{self.checkpoint_dir}/{checkpoint_name}",
                eval_strategy="epoch",
                learning_rate=5e-5,
                logging_dir="logs",
                logging_steps=50,
                num_train_epochs=3,
                overwrite_output_dir=True,
                per_device_eval_batch_size=8,
                per_device_train_batch_size=8,
                save_strategy="epoch",
                # Add more if needed
            ),
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=self.data_collator,
            #tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

    def tokenize_and_align_labels(self, examples):
        # We'll build lists that accumulate results for the entire batch.
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        # If `examples["tokens"]` and `examples["tags"]` each contain multiple examples,
        # we iterate over them in parallel.
        for tokens, tags in zip(examples["tokens"], examples["tags"]):
            # Tokenize a *single* example which may produce multiple chunks if >512 tokens.
            outputs = self.tokenizer(
                tokens,  # one example's tokens
                is_split_into_words=True,
                max_length=512,
                padding="max_length",
                return_overflowing_tokens=True,  # enables chunking
                stride=128,  # 128-token overlap
                truncation=True,
            )

            # Each overflowed chunk is stored in `outputs["input_ids"][i]`, etc.
            num_chunks = len(outputs["input_ids"])

            for i in range(num_chunks):
                # For chunk i, get the mapping from subwords back to word indices
                word_ids = outputs.word_ids(batch_index=i)

                # Build the label ids for this chunk
                chunk_labels = []
                for word_id in word_ids:
                    if word_id is None:
                        chunk_labels.append(-100)  # special tokens ([CLS], [SEP], padding)
                    else:
                        # Map to the appropriate tag index
                        chunk_labels.append(self.tag2id[tags[word_id]])

                # Accumulate results
                all_input_ids.append(outputs["input_ids"][i])
                all_attention_masks.append(outputs["attention_mask"][i])
                # If your tokenizer returns token_type_ids or other fields, handle them similarly
                all_labels.append(chunk_labels)

        # Return a dict matching what Hugging Face Datasets expects
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
            # If you have token_type_ids or other fields, include them here too
        }


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    validation_predictions, validation_labels = [], []
    for pred_row, lab_row in zip(predictions, labels):
        for p_, l_ in zip(pred_row, lab_row):
            if l_ != -100:
                validation_predictions.append(p_)
                validation_labels.append(l_)

    # Convert to NumPy arrays
    validation_predictions = np.array(validation_predictions)
    validation_labels = np.array(validation_labels)

    # This works correctly in NumPy
    accuracy = (validation_predictions == validation_labels).mean()
    return {"accuracy": accuracy}


if __name__ == "__main__":
    from datasets import Dataset, DatasetDict
    from observability.logging import setup_logging

    setup_logging()

    deberta_checkpoint_name = "deberta-pos"

    deberta_model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/deberta-base",
        num_labels=len(AutoPosTuner.tag2id),
        id2label=AutoPosTuner.id2tag,
        label2id=AutoPosTuner.tag2id
    )
    deberta_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-base",
        add_prefix_space=True,
        use_fast=True,
    )
    pos_tuner = AutoPosTuner(deberta_model, deberta_tokenizer)

    # Suppose your data is in a Python dict format
    train_data = {
        "tokens": [["The", "cat", "sat"], ["A", "big", "dog", "ran"]],
        "tags":   [["DT",  "NN",  "VBD"], ["DT", "JJ",  "NN",  "VBD"]]
    }
    train_dataset = Dataset.from_dict(train_data)

    validation_data = {
        "tokens": [["Another", "sentence"]],
        "tags":   [["DT", "NN"]]
    }
    validation_dataset = Dataset.from_dict(validation_data)

    # Combine into a DatasetDict if you want
    combined_dataset_dict = DatasetDict({"train": train_dataset, "validation": validation_dataset})

    # Map the tokenize_and_align_labels function
    combined_dataset_dict = combined_dataset_dict.map(pos_tuner.tokenize_and_align_labels, batched=True)
    combined_dataset_dict.set_format("torch")
    pos_trainer = pos_tuner.new_trainer(combined_dataset_dict, deberta_checkpoint_name)

    pos_trainer.train()
    pos_trainer.evaluate()
    pos_trainer.save_model(f"{pos_tuner.checkpoint_dir}/{deberta_checkpoint_name}/final")

