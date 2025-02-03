from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AutoPOSTuner:

    checkpoint_dir = "models/germ"
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
                 checkpoint_dir=None,
                 id2tag=None, tag2id=None):
        self.checkpoint_dir = AutoPOSTuner.checkpoint_dir if checkpoint_dir is None else checkpoint_dir
        self.model = model
        self.tokenizer = tokenizer

        self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        self.id2tag = AutoPOSTuner.id2tag if id2tag is None else id2tag
        self.tag2id = AutoPOSTuner.tag2id if tag2id is None else tag2id

    def new_trainer(self, dataset, checkpoint_name):
        return Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=f"{self.checkpoint_dir}/{checkpoint_name}",
                eval_strategy="epoch",
                learning_rate=5e-5,
                logging_dir="data/training_logs",
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
    """
    Compute accuracy, precision, recall, and F1 scores from model predictions.

    This function expects `p` to be a tuple (predictions, labels), where:
      - predictions: a NumPy array of shape (batch_size, sequence_length, num_labels)
      - labels: a NumPy array of shape (batch_size, sequence_length) with -100 for tokens to ignore.

    Returns a dictionary with overall accuracy, macro-averaged and micro-averaged precision,
    recall, and F1 scores.
    """

    # Unpack predictions and labels; convert logits to predicted label indices.
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    # Flatten predictions and labels, ignoring tokens with label == -100.
    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_predictions.append(pred)
                true_labels.append(label)

    true_predictions = np.array(true_predictions)
    true_labels = np.array(true_labels)

    # Calculate overall accuracy.
    accuracy = (true_predictions == true_labels).mean()

    # Calculate precision, recall, and F1 score using macro-averaging.
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average="macro", zero_division=0
    )

    # Calculate precision, recall, and F1 score using micro-averaging.
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average="micro", zero_division=0
    )

    return {
        # We compute accuracy as the mean of correctly predicted tokens over all non-ignored tokens.
        "accuracy": accuracy,
        # Macro-averaging: Computes the metrics for each label independently and then takes the average—this gives
        # equal weight to each class.
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        # Micro-averaging: Aggregates the contributions of all classes to compute the metrics—this is typically useful
        # when you have class imbalance. The parameter zero_division=0 ensures that if a class has no predicted samples
        # (which could otherwise lead to a division by zero), the score is set to 0.
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }


def generate_classification_report(true_labels, true_predictions, id2tag):
    """
    Generate a detailed classification report using scikit-learn's classification_report.

    Parameters:
      - true_labels: a list or numpy array of true label ids.
      - true_predictions: a list or numpy array of predicted label ids.
      - id2tag: dictionary mapping label id to label string.
    """
    from sklearn.metrics import classification_report

    # Convert label ids to their tag names.
    true_tags = [id2tag[label] for label in true_labels]
    pred_tags = [id2tag[pred] for pred in true_predictions]

    report = classification_report(true_tags, pred_tags, zero_division=0)
    logger.info(f"Classification Report:\n{report}")
    return report


def get_model(model_name_or_path: str):
    return AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(AutoPOSTuner.tag2id),
        id2label=AutoPOSTuner.id2tag,
        label2id=AutoPOSTuner.tag2id
    )


def get_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        add_prefix_space=True,
        use_fast=True,
    )


if __name__ == "__main__":
    from datasets import Dataset, DatasetDict
    from observability.logging import setup_logging

    setup_logging()

    deberta_model_name = "microsoft/deberta-base"
    deberta_checkpoint_name = "deberta-pos"

    deberta_model = get_model(deberta_model_name)
    deberta_tokenizer = get_tokenizer(deberta_model_name)
    pos_tuner = AutoPOSTuner(deberta_model, deberta_tokenizer)

    deberta_final_checkpoint_name = f"{pos_tuner.checkpoint_dir}/{deberta_checkpoint_name}/final"

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

    # -----------------------------
    # Additional Analysis: Generate Classification Report
    # -----------------------------

    # Run predictions on the validation dataset.
    predictions_output = pos_trainer.predict(combined_dataset_dict["validation"])
    predictions = predictions_output.predictions
    labels = predictions_output.label_ids

    flat_predictions = []
    flat_labels = []
    # Flatten predictions and labels while ignoring -100.
    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(np.argmax(pred_seq, axis=-1), label_seq):
            if label != -100:
                flat_predictions.append(pred)
                flat_labels.append(label)

    # Generate and show the classification report.
    generate_classification_report(flat_labels, flat_predictions, AutoPOSTuner.id2tag)

    pos_trainer.save_model(deberta_final_checkpoint_name)
    deberta_tokenizer.save_pretrained(deberta_final_checkpoint_name)


