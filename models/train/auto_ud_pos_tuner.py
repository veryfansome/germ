from sklearn.metrics import precision_recall_fscore_support
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AutoPOSTuner:
    checkpoint_dir = "models/germ"
    tag2id = {}  # to be updated from UD training data
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
                # Add more if needed
            ),
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )

    def tokenize_and_align_labels(self, examples):
        # We'll build lists that accumulate results for the entire batch.
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        # Process each example in the batch.
        for tokens, tags in zip(examples["tokens"], examples["xpos"]):
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
    Generate a classification report mapping integer labels back to tag names.
    """
    from sklearn.metrics import classification_report

    # Convert label ids to their tag names.
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
    from datasets import concatenate_datasets, load_dataset, DatasetDict
    from observability.logging import setup_logging
    import argparse

    setup_logging()

    arg_parser = argparse.ArgumentParser(description='Build a graph of ideas.')
    arg_parser.add_argument("--from-base", help='Load a base model.',
                            action="store_true", default=False)
    arg_parser.add_argument("--train", help='Train model using loaded examples.',
                            action="store_true", default=False)
    args = arg_parser.parse_args()

    # 1. Load UD datasets: en_ewt and en_gum.
    loaded_datasets = []

    for dataset_name in [
        # 89 total unique classes
        #"en_ewt",     # 50 classes
        #"en_gum",     # 45 classes
        "en_partut",  # 38 classes
    ]:
        downloaded_ds = load_dataset("universal_dependencies", dataset_name)
        loaded_datasets.append(downloaded_ds)
        for split in downloaded_ds.keys():
            logger.info(f"{dataset_name} {split}: {len(downloaded_ds[split])}")

    test_dataset = concatenate_datasets([ds["test"] for ds in loaded_datasets if "test" in ds])
    train_dataset = concatenate_datasets([ds["train"] for ds in loaded_datasets if "train" in ds])
    validation_dataset = concatenate_datasets([ds["validation"] for ds in loaded_datasets if "validation" in ds])
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(validation_dataset)}")

    # 2. Build the POS tag mappings from the combined data (train, validation, test).
    unique_tags = set()
    for ds in [train_dataset, validation_dataset, test_dataset]:
        # Use "xpos" labels
        for tags in ds["xpos"]:
            if tags is None:
                continue
            for tag in tags:
                if tag is not None:
                    unique_tags.add(tag)
    unique_tags = sorted(unique_tags)
    logger.info(f"Unique tag count: {len(unique_tags)}")
    logger.info(f"Unique tags: {unique_tags}")

    tag2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2tag = {i: tag for tag, i in tag2id.items()}

    # Update the AutoPOSTuner class variables so that the model gets the correct number of labels.
    AutoPOSTuner.tag2id = tag2id
    AutoPOSTuner.id2tag = id2tag

    # 3. Initialize model and tokenizer.
    deberta_model_name = "microsoft/deberta-base"
    deberta_checkpoint_name = "deberta-ud-pos"
    # Final checkpoint path
    deberta_final_checkpoint_path = f"models/germ/{deberta_checkpoint_name}/final"

    if args.from_base:
        deberta_model = get_model(deberta_model_name,
                                  num_labels=len(AutoPOSTuner.tag2id),
                                  id2label=AutoPOSTuner.id2tag,
                                  label2id=AutoPOSTuner.tag2id)
        deberta_tokenizer = get_tokenizer(deberta_model_name)
    else:
        deberta_model = get_model(deberta_final_checkpoint_path,
                                  num_labels=len(AutoPOSTuner.tag2id),
                                  id2label=AutoPOSTuner.id2tag,
                                  label2id=AutoPOSTuner.tag2id)
        deberta_tokenizer = get_tokenizer(deberta_final_checkpoint_path)

    pos_tuner = AutoPOSTuner(deberta_model, deberta_tokenizer)

    # 4. Combine the UD training and validation splits into a DatasetDict.
    combined_dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
    })

    # 5. Tokenize and align labels. The function now uses the "xpos" field.
    combined_dataset_dict = combined_dataset_dict.map(pos_tuner.tokenize_and_align_labels, batched=True)
    combined_dataset_dict.set_format("torch")

    # 6. Create and run the Trainer.
    pos_trainer = pos_tuner.new_trainer(combined_dataset_dict, deberta_checkpoint_name)

    if args.train:
        pos_trainer.train()
        pos_trainer.evaluate()

        # Save the final model and tokenizer.
        pos_trainer.save_model(deberta_final_checkpoint_path)
        deberta_tokenizer.save_pretrained(deberta_final_checkpoint_path)

    # -----------------------------
    # Additional Analysis: Generate Classification Report
    # -----------------------------
    # Map the test dataset using the same tokenize_and_align_labels function.
    test_dataset = test_dataset.map(pos_tuner.tokenize_and_align_labels, batched=True)
    test_dataset.set_format("torch")

    test_predictions_output = pos_trainer.predict(test_dataset)
    test_predictions = test_predictions_output.predictions
    test_prediction_labels = test_predictions_output.label_ids

    flat_predictions = []
    flat_labels = []
    for test_pred_seq, test_label_seq in zip(test_predictions, test_prediction_labels):
        for test_pred, test_label in zip(np.argmax(test_pred_seq, axis=-1), test_label_seq):
            if test_label != -100:
                flat_predictions.append(test_pred)
                flat_labels.append(test_label)

    generate_classification_report(flat_labels, flat_predictions, AutoPOSTuner.id2tag)
