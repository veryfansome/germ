from sklearn.metrics import precision_recall_fscore_support
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AutoPOSTuner:
    checkpoint_dir = "models/germ"
    tag2id = {}  # to be updated from CoNLL training data
    id2tag = {}

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
            ),
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )

    def tokenize_and_align_labels(self, examples):
        # We will accumulate all results over a batch.
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        # Process each example in the batch.
        for tokens, tags in zip(examples["tokens"], examples["pos_tags"]):
            outputs = self.tokenizer(
                tokens,  # one example's tokens
                is_split_into_words=True,
                max_length=512,
                padding="max_length",
                return_overflowing_tokens=True,  # enables chunking
                stride=128,  # 128-token overlap
                truncation=True,
            )

            # Process each overflowed chunk.
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
                        token_tag = tags[word_id]
                        if token_tag is None:
                            # If the token tag is missing, ignore this token.
                            chunk_labels.append(-100)
                        else:
                            # If token_tag is already an int (from a ClassLabel), use it directly.
                            if isinstance(token_tag, int):
                                chunk_labels.append(token_tag)
                            else:
                                chunk_labels.append(self.tag2id[token_tag])
                all_input_ids.append(outputs["input_ids"][i])
                all_attention_masks.append(outputs["attention_mask"][i])
                all_labels.append(chunk_labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
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
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }


def generate_classification_report(true_labels, true_predictions, id2tag):
    """
    Generate a classification report mapping integer labels back to tag names.
    """
    from sklearn.metrics import classification_report

    # Convert label ids to tag names.
    true_tags = [id2tag[label] for label in true_labels]
    pred_tags = [id2tag[pred] for pred in true_predictions]

    report = classification_report(true_tags, pred_tags, zero_division=0)
    logger.info(f"Classification Report:\n{report}")
    return report


def get_model(model_name_or_path: str, **kwargs):
    return AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        **kwargs,
    )


def get_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        add_prefix_space=True,
        use_fast=True,
    )


if __name__ == "__main__":
    import argparse
    from datasets import DatasetDict, DownloadConfig, concatenate_datasets, load_dataset
    from observability.logging import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description='Train a CoNLL-based POS tagger.')
    parser.add_argument("--from-base", help="Load a base model.",
                        action="store_true", default=False)
    parser.add_argument("--train", help="Train model using loaded examples.",
                        action="store_true", default=False)
    args = parser.parse_args()

    # 1. Load CoNLL datasets

    conll_dataset_name = "conll2000"
    #conll_dataset_name = "conll2003"
    #conll_dataset_name = "english_v4"
    #conll_dataset_name = "english_v12"

    ds = load_dataset(conll_dataset_name)
    #ds = load_dataset("ontonotes/conll2012_ontonotesv5", conll_dataset_name,
    #                  download_config=DownloadConfig(max_retries=3, resume_download=True))

    logger.info(f"Train set size: {len(ds['train'])}")
    logger.info(f"Test set size: {len(ds['test'])}")
    logger.info(f"Validation set size: {len(ds['validation']) if 'validation' in ds else 0}")

    # Note: conll2000 only provides train and test splits.
    # For our training procedure, we split the original train split to create our own train/validation splits.
    if conll_dataset_name == "conll2000":
        split_datasets = ds["train"].train_test_split(test_size=0.1, seed=42)
        train_dataset = split_datasets["train"]
        validation_dataset = split_datasets["test"]
    else:
        train_dataset = ds["train"]
        validation_dataset = ds["validation"]

    # Use the provided test split as a final held out test set.
    test_dataset = ds["test"]
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })

    #  Build the POS Tag Mappings
    logger.info(f"Unique tag count: {len(train_dataset.features['pos_tags'].feature.names)}")
    logger.info(f"Unique tags: {train_dataset.features['pos_tags'].feature.names}")

    tag2id = {tag: i for i, tag in enumerate(train_dataset.features['pos_tags'].feature.names)}
    id2tag = {i: tag for tag, i in tag2id.items()}

    # Update class-level variables of AutoPOSTuner.
    AutoPOSTuner.tag2id = tag2id
    AutoPOSTuner.id2tag = id2tag

    # Initialize Model and Tokenizer
    transformer_model = "microsoft/deberta-base"
    checkpoint_name = "deberta-pos-conll"
    final_checkpoint_path = f"models/germ/{checkpoint_name}/final"

    if args.from_base:
        model = get_model(transformer_model,
                          num_labels=len(tag2id),
                          id2label=id2tag,
                          label2id=tag2id)
        tokenizer = get_tokenizer(transformer_model)
    else:
        model = get_model(final_checkpoint_path,
                          num_labels=len(tag2id),
                          id2label=id2tag,
                          label2id=tag2id)
        tokenizer = get_tokenizer(final_checkpoint_path)

    pos_tuner = AutoPOSTuner(model, tokenizer)

    # Tokenize and Align Labels
    # Note: Our tokenize function now uses the “pos_tags” field.
    dataset_dict = dataset_dict.map(pos_tuner.tokenize_and_align_labels, batched=True)
    dataset_dict.set_format("torch")

    # Create Trainer
    trainer = pos_tuner.new_trainer(dataset_dict, checkpoint_name)

    # Train Model
    if args.train:
        trainer.train()
        trainer.evaluate()

        # Save the final model and tokenizer.
        trainer.save_model(final_checkpoint_path)
        tokenizer.save_pretrained(final_checkpoint_path)

    # -----------------------------
    #   Additional Analysis on Test Set
    # -----------------------------
    # Process the test dataset using the tokenize and align function.
    test_dataset = test_dataset.map(pos_tuner.tokenize_and_align_labels, batched=True)
    test_dataset.set_format("torch")

    test_predictions_output = trainer.predict(test_dataset)
    test_predictions = test_predictions_output.predictions
    test_prediction_labels = test_predictions_output.label_ids

    flat_predictions = []
    flat_labels = []
    for test_pred_seq, test_label_seq in zip(test_predictions, test_prediction_labels):
        for test_pred, test_label in zip(np.argmax(test_pred_seq, axis=-1), test_label_seq):
            if test_label != -100:
                flat_predictions.append(test_pred)
                flat_labels.append(test_label)

    generate_classification_report(flat_labels, flat_predictions, id2tag)