import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
import logging

logger = logging.getLogger(__name__)

###########################################################################
# 1. Custom Joint Model: Shared DeBERTa Encoder with Two Classification Heads
###########################################################################


class JointTokenClassificationModel(nn.Module):
    def __init__(self, model_name_or_path, num_pos_labels, num_ner_labels):
        super(JointTokenClassificationModel, self).__init__()
        # Load the base configuration and encoder (without any head)
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        hidden_size = self.config.hidden_size

        # Two classification heads: one for POS and one for NER.
        self.pos_classifier = nn.Linear(hidden_size, num_pos_labels)
        self.ner_classifier = nn.Linear(hidden_size, num_ner_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                pos_labels=None, ner_labels=None):
        # Get hidden states from the shared encoder.
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]  # shape: (batch, seq_len, hidden_size)

        # Compute logits for each task.
        pos_logits = self.pos_classifier(sequence_output)  # (batch, seq_len, num_pos_labels)
        ner_logits = self.ner_classifier(sequence_output)  # (batch, seq_len, num_ner_labels)

        loss = None
        if pos_labels is not None and ner_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            pos_loss = loss_fct(pos_logits.view(-1, pos_logits.size(-1)), pos_labels.view(-1))
            ner_loss = loss_fct(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))
            loss = pos_loss + ner_loss

        # IMPORTANT: Although we return a dict here, Trainer may convert it to a tuple.
        return {"loss": loss, "pos_logits": pos_logits, "ner_logits": ner_logits}


###########################################################################
# 2. Joint Tuner Class for Multi-Task Learning
###########################################################################


class JointPosNerTuner:
    checkpoint_dir = "models/germ"
    # Define a simple POS tag set.
    pos_tag2id = {
        "NN": 0,
        "VBD": 1,
        "DT": 2,
        "IN": 3,
        "JJ": 4,
    }
    pos_id2tag = {v: k for k, v in pos_tag2id.items()}
    # Define a simple NER tag set (using an IOB scheme).
    ner_tag2id = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
    }
    ner_id2tag = {v: k for k, v in ner_tag2id.items()}

    def __init__(self, model, tokenizer, checkpoint_dir=None):
        self.checkpoint_dir = JointPosNerTuner.checkpoint_dir if checkpoint_dir is None else checkpoint_dir
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

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
            compute_metrics=compute_joint_metrics,
        )

    def tokenize_and_align_labels(self, examples):
        """
        For each example (with keys "tokens", "pos_tags", "ner_tags"),
        tokenize the text with sliding-window and produce aligned labels for both tasks.
        """
        all_input_ids = []
        all_attention_masks = []
        all_pos_labels = []
        all_ner_labels = []

        for tokens, pos_tags, ner_tags in zip(examples["tokens"],
                                              examples["pos_tags"],
                                              examples["ner_tags"]):
            outputs = self.tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=512,
                padding="max_length",
                return_overflowing_tokens=True,
                stride=128,
                truncation=True,
            )
            num_chunks = len(outputs["input_ids"])
            for i in range(num_chunks):
                word_ids = outputs.word_ids(batch_index=i)
                pos_chunk_labels = []
                ner_chunk_labels = []
                for word_id in word_ids:
                    if word_id is None:
                        pos_chunk_labels.append(-100)
                        ner_chunk_labels.append(-100)
                    else:
                        pos_chunk_labels.append(JointPosNerTuner.pos_tag2id[pos_tags[word_id]])
                        ner_chunk_labels.append(JointPosNerTuner.ner_tag2id[ner_tags[word_id]])
                all_input_ids.append(outputs["input_ids"][i])
                all_attention_masks.append(outputs["attention_mask"][i])
                all_pos_labels.append(pos_chunk_labels)
                all_ner_labels.append(ner_chunk_labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "pos_labels": all_pos_labels,
            "ner_labels": all_ner_labels,
        }


###########################################################################
# 3. Joint Metrics Function (Corrected)
###########################################################################


def compute_joint_metrics(p):
    """
    Compute separate metrics for POS and NER.
    p is a tuple (predictions, labels). Depending on the Trainer,
    predictions and labels may be returned as a dict or as a tuple.
    This function handles both cases.
    """
    predictions, labels = p

    # --- Handle predictions ---
    if isinstance(predictions, dict):
        pos_logits = predictions.get("pos_logits")
        ner_logits = predictions.get("ner_logits")
    elif isinstance(predictions, (list, tuple)):
        if len(predictions) == 3:
            # Assume structure: (loss, pos_logits, ner_logits)
            pos_logits = predictions[1]
            ner_logits = predictions[2]
        elif len(predictions) == 2:
            # Assume structure: (pos_logits, ner_logits)
            pos_logits = predictions[0]
            ner_logits = predictions[1]
        else:
            raise ValueError("Unexpected predictions tuple length: {}".format(len(predictions)))
    else:
        raise ValueError("Unexpected predictions type: {}".format(type(predictions)))

    # --- Handle labels ---
    if isinstance(labels, dict):
        pos_labels = labels.get("pos_labels")
        ner_labels = labels.get("ner_labels")
    elif isinstance(labels, (list, tuple)):
        if len(labels) >= 2:
            pos_labels, ner_labels = labels[0], labels[1]
        else:
            raise ValueError("Expected labels tuple to have at least 2 elements, got: {}".format(len(labels)))
    else:
        raise ValueError("Unexpected labels type: {}".format(type(labels)))

    # --- Process predictions ---
    pos_predictions = np.argmax(pos_logits, axis=2)
    ner_predictions = np.argmax(ner_logits, axis=2)

    true_pos_predictions, true_pos_labels = [], []
    true_ner_predictions, true_ner_labels = [], []

    # Flatten POS predictions.
    for pred_seq, label_seq in zip(pos_predictions, pos_labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_pos_predictions.append(pred)
                true_pos_labels.append(label)

    # Flatten NER predictions.
    for pred_seq, label_seq in zip(ner_predictions, ner_labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_ner_predictions.append(pred)
                true_ner_labels.append(label)

    true_pos_predictions = np.array(true_pos_predictions)
    true_pos_labels = np.array(true_pos_labels)
    true_ner_predictions = np.array(true_ner_predictions)
    true_ner_labels = np.array(true_ner_labels)

    pos_accuracy = (true_pos_predictions == true_pos_labels).mean()
    ner_accuracy = (true_ner_predictions == true_ner_labels).mean()

    pos_precision, pos_recall, pos_f1, _ = precision_recall_fscore_support(
        true_pos_labels, true_pos_predictions, average="macro", zero_division=0
    )
    ner_precision, ner_recall, ner_f1, _ = precision_recall_fscore_support(
        true_ner_labels, true_ner_predictions, average="macro", zero_division=0
    )

    metrics = {
        "pos_accuracy": pos_accuracy,
        "pos_precision": pos_precision,
        "pos_recall": pos_recall,
        "pos_f1": pos_f1,
        "ner_accuracy": ner_accuracy,
        "ner_precision": ner_precision,
        "ner_recall": ner_recall,
        "ner_f1": ner_f1,
    }
    return metrics


###########################################################################
# 4. Helper Functions for Model & Tokenizer
###########################################################################


# Generate and log the detailed classification reports.
def generate_joint_classification_report(true_pos, pred_pos, true_ner, pred_ner, pos_id2tag, ner_id2tag):
    pos_true_tags = [pos_id2tag[pos_label] for pos_label in true_pos]
    pos_pred_tags = [pos_id2tag[pos_pred] for pos_pred in pred_pos]
    ner_true_tags = [ner_id2tag[ner_label] for ner_label in true_ner]
    ner_pred_tags = [ner_id2tag[ner_pred] for ner_pred in pred_ner]
    pos_report = classification_report(pos_true_tags, pos_pred_tags, zero_division=0)
    ner_report = classification_report(ner_true_tags, ner_pred_tags, zero_division=0)
    logger.info(f"POS Classification Report:\n{pos_report}")
    logger.info(f"NER Classification Report:\n{ner_report}")
    return pos_report, ner_report


def get_joint_model(model_name_or_path: str):
    num_pos_labels = len(JointPosNerTuner.pos_tag2id)
    num_ner_labels = len(JointPosNerTuner.ner_tag2id)
    return JointTokenClassificationModel(model_name_or_path, num_pos_labels, num_ner_labels)


def get_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(model_name_or_path, add_prefix_space=True, use_fast=True)


###########################################################################
# 5. Main Training & Evaluation Script
###########################################################################


if __name__ == "__main__":
    from datasets import Dataset, DatasetDict
    from observability.logging import setup_logging

    setup_logging()

    model_name = "microsoft/deberta-base"
    checkpoint_name = "deberta-joint"

    joint_model = get_joint_model(model_name)
    tokenizer = get_tokenizer(model_name)
    tuner = JointPosNerTuner(joint_model, tokenizer)
    final_checkpoint_name = f"{tuner.checkpoint_dir}/{checkpoint_name}/final"

    # Example joint training data: each sample has tokens, pos_tags, and ner_tags.
    train_data = {
        "tokens": [
            ["John", "lives", "in", "New", "York"],
            ["Apple", "released", "the", "iPhone"]
        ],
        "pos_tags": [
            ["NN", "VBD", "IN", "NN", "NN"],
            ["NN", "VBD", "DT", "NN"]
        ],
        "ner_tags": [
            ["B-PER", "O", "O", "B-LOC", "I-LOC"],
            ["B-ORG", "O", "O", "O"]
        ]
    }
    train_dataset = Dataset.from_dict(train_data)

    validation_data = {
        "tokens": [["Mary", "visited", "Google", "in", "California"]],
        "pos_tags": [["NN", "VBD", "NN", "IN", "NN"]],
        "ner_tags": [["B-PER", "O", "B-ORG", "O", "B-LOC"]]
    }
    validation_dataset = Dataset.from_dict(validation_data)

    combined_dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })

    # Map the tokenization and alignment function.
    combined_dataset = combined_dataset.map(tuner.tokenize_and_align_labels, batched=True)
    combined_dataset.set_format("torch")
    trainer = tuner.new_trainer(combined_dataset, checkpoint_name)

    # Train and evaluate.
    trainer.train()
    trainer.evaluate()

    # -----------------------------
    # Additional Analysis: Generate Joint Classification Reports.
    # -----------------------------

    predictions_output = trainer.predict(combined_dataset["validation"])
    predictions = predictions_output.predictions
    labels = predictions_output.label_ids

    # --- Process predictions ---
    if isinstance(predictions, dict):
        pos_logits = predictions["pos_logits"]
        ner_logits = predictions["ner_logits"]
    elif isinstance(predictions, (list, tuple)):
        if len(predictions) == 3:
            # Assume structure: (loss, pos_logits, ner_logits)
            pos_logits = predictions[1]
            ner_logits = predictions[2]
        elif len(predictions) == 2:
            pos_logits = predictions[0]
            ner_logits = predictions[1]
        else:
            raise ValueError("Unexpected predictions tuple length: {}".format(len(predictions)))
    else:
        raise ValueError("Unexpected predictions type: {}".format(type(predictions)))

    # --- Process labels ---
    if isinstance(labels, dict):
        pos_labels_out = labels["pos_labels"]
        ner_labels_out = labels["ner_labels"]
    elif isinstance(labels, (list, tuple)):
        if len(labels) >= 2:
            pos_labels_out, ner_labels_out = labels[0], labels[1]
        else:
            raise ValueError("Expected labels to have at least 2 elements, got: {}".format(len(labels)))
    else:
        raise ValueError("Unexpected labels type: {}".format(type(labels)))

    # --- Flatten predictions and labels ---
    pos_predictions = np.argmax(pos_logits, axis=2)
    ner_predictions = np.argmax(ner_logits, axis=2)

    flat_pos_predictions = []
    flat_pos_labels = []
    flat_ner_predictions = []
    flat_ner_labels = []

    # Flatten POS predictions.
    for pred_seq, label_seq in zip(pos_predictions, pos_labels_out):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                flat_pos_predictions.append(pred)
                flat_pos_labels.append(label)

    # Flatten NER predictions.
    for pred_seq, label_seq in zip(ner_predictions, ner_labels_out):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                flat_ner_predictions.append(pred)
                flat_ner_labels.append(label)

    generate_joint_classification_report(flat_pos_labels, flat_pos_predictions, flat_ner_labels, flat_ner_predictions,
                                           JointPosNerTuner.pos_id2tag, JointPosNerTuner.ner_id2tag)

    trainer.save_model(final_checkpoint_name)
    tokenizer.save_pretrained(final_checkpoint_name)
