import logging
import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import (classification_report as seq_classification_report, f1_score as seq_f1_score,
                             precision_score as seq_precision_score, recall_score as seq_recall_score)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Weighted Loss Trainer: Subclass Trainer to support weighted loss.
# ------------------------------------------------------------
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Use weighted loss if provided; note ignore_index=-100 to skip subword tokens.
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ------------------------------------------------------------
# AutoNERTuner: Configuration and utility for NER.
# ------------------------------------------------------------
class AutoNERTuner:
    # Updated directory and tag mappings for a common IOB scheme.
    checkpoint_dir = "models/ner"
    tag2id = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8,
    }
    id2tag = {v: k for k, v in tag2id.items()}

    def __init__(self, model, tokenizer, checkpoint_dir=None, id2tag=None, tag2id=None):
        self.checkpoint_dir = AutoNERTuner.checkpoint_dir if checkpoint_dir is None else checkpoint_dir
        self.model = model
        self.tokenizer = tokenizer
        # DataCollatorForTokenClassification supports dynamic padding by default.
        self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        self.id2tag = AutoNERTuner.id2tag if id2tag is None else id2tag
        self.tag2id = AutoNERTuner.tag2id if tag2id is None else tag2id

    def new_trainer(self, dataset, checkpoint_name, class_weights=None):
        # Use WeightedLossTrainer if class_weights is provided.
        trainer_cls = WeightedLossTrainer if class_weights is not None else Trainer
        return trainer_cls(
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
            #tokenizer=self.tokenizer,
            class_weights=class_weights,
        )

    def tokenize_and_align_labels(self, examples):
        """
        Tokenize input tokens while aligning NER labels.
        Instead of padding to a fixed max_length, we allow the collator to do dynamic padding.
        For words split into multiple subwords, only the first token is labeled;
        subsequent subword tokens are set to -100.
        """
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for tokens, tags in zip(examples["tokens"], examples["tags"]):
            outputs = self.tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=512,
                return_overflowing_tokens=True,  # enables chunking if sentence >512 tokens
                stride=128,  # overlap between chunks
                truncation=True,
                # Note: We do not pad here to support dynamic padding.
            )
            num_chunks = len(outputs["input_ids"])
            for i in range(num_chunks):
                word_ids = outputs.word_ids(batch_index=i)
                chunk_labels = []
                previous_word_idx = None
                for word_id in word_ids:
                    if word_id is None:
                        chunk_labels.append(-100)
                    elif word_id != previous_word_idx:
                        # Only label the first subword.
                        chunk_labels.append(self.tag2id[tags[word_id]])
                        previous_word_idx = word_id
                    else:
                        chunk_labels.append(-100)
                all_input_ids.append(outputs["input_ids"][i])
                all_attention_masks.append(outputs["attention_mask"][i])
                all_labels.append(chunk_labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        }


# ------------------------------------------------------------
# Metrics Computation: Token-level and Entity-level (using seqeval)
# ------------------------------------------------------------
def compute_metrics(p):
    """
    Compute token-level metrics (accuracy, macro/micro precision/recall/F1)
    and entity-level metrics using seqeval.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # --- Token-Level Metrics ---
    true_token_preds = []
    true_token_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_token_preds.append(pred)
                true_token_labels.append(label)
    true_token_preds = np.array(true_token_preds)
    true_token_labels = np.array(true_token_labels)
    token_accuracy = (true_token_preds == true_token_labels).mean()
    token_precision_macro, token_recall_macro, token_f1_macro, _ = precision_recall_fscore_support(
        true_token_labels, true_token_preds, average="macro", zero_division=0
    )
    token_precision_micro, token_recall_micro, token_f1_micro, _ = precision_recall_fscore_support(
        true_token_labels, true_token_preds, average="micro", zero_division=0
    )

    # --- Entity-Level Metrics ---
    # Reconstruct predictions and labels per sequence, ignoring -100.
    true_entities = []
    pred_entities = []
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_list = []
        for pred, label in zip(pred_seq, label_seq):
            if label == -100:
                continue
            true_seq.append(AutoNERTuner.id2tag[label])
            pred_seq_list.append(AutoNERTuner.id2tag[pred])
        if true_seq:  # Only include non-empty sequences.
            true_entities.append(true_seq)
            pred_entities.append(pred_seq_list)

    entity_precision = seq_precision_score(true_entities, pred_entities)
    entity_recall = seq_recall_score(true_entities, pred_entities)
    entity_f1 = seq_f1_score(true_entities, pred_entities)
    entity_classification_report = seq_classification_report(true_entities, pred_entities)

    # Combine all metrics into one dictionary.
    return {
        "token_accuracy": token_accuracy,
        "token_precision_macro": token_precision_macro,
        "token_recall_macro": token_recall_macro,
        "token_f1_macro": token_f1_macro,
        "token_precision_micro": token_precision_micro,
        "token_recall_micro": token_recall_micro,
        "token_f1_micro": token_f1_micro,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "entity_classification_report": entity_classification_report,
    }


# ------------------------------------------------------------
# Utility functions to get model and tokenizer.
# ------------------------------------------------------------
def get_model(model_name_or_path: str):
    return AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(AutoNERTuner.tag2id),
        id2label=AutoNERTuner.id2tag,
        label2id=AutoNERTuner.tag2id,
    )


def get_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        add_prefix_space=True,
        use_fast=True,
    )


# ------------------------------------------------------------
# Compute class weights from raw training data.
# ------------------------------------------------------------
def compute_class_weights_from_data(data, tag2id):
    """
    Compute weights inversely proportional to class frequency.
    data: dict with key "tags" which is a list of lists of string labels.
    Returns a torch tensor of weights arranged by label id.
    """
    counts = {tag: 0 for tag in tag2id.keys()}
    total = 0
    for tags in data["tags"]:
        for tag in tags:
            if tag in counts:
                counts[tag] += 1
                total += 1
    num_classes = len(tag2id)
    weights = {}
    for tag, count in counts.items():
        # If count is 0, weight=1 to avoid division by zero.
        weights[tag] = total / (count * num_classes) if count > 0 else 1.0
    # Arrange weights in order of tag id.
    weight_list = [weights[tag] for tag, id_ in sorted(tag2id.items(), key=lambda x: x[1])]
    return torch.tensor(weight_list, dtype=torch.float)


# ------------------------------------------------------------
# Main training and evaluation logic.
# ------------------------------------------------------------
if __name__ == "__main__":
    from datasets import Dataset, DatasetDict
    # Assuming you have a logging setup function.
    from observability.logging import setup_logging

    setup_logging()

    # We are in January 2025.
    model_name = "microsoft/deberta-base"
    checkpoint_name = "deberta-ner"

    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)
    ner_tuner = AutoNERTuner(model, tokenizer)

    final_checkpoint_name = f"{ner_tuner.checkpoint_dir}/{checkpoint_name}/final"

    # Example training data (NER-style)
    train_data = {
        "tokens": [
            ["John", "lives", "in", "New", "York"],
            ["Derek", "and", "Anne", "met", "in", "San", "Francisco"],
            ["Jack", "died", "with", "the", "Titanic"],
            ["Mary", "works", "at", "Google"],
        ],
        "tags": [
            ["B-PER", "O", "O", "B-LOC", "I-LOC"],
            ["B-PER", "O", "B-PER", "O", "O", "B-LOC", "I-LOC"],
            ["B-PER", "O", "O", "O", "B-ORG"],
            ["B-PER", "O", "O", "B-ORG"],
        ]
    }
    train_dataset = Dataset.from_dict(train_data)

    validation_data = {
        "tokens": [
            ["Alice", "visited", "Paris"],
            ["Marco", "Polo", "went", "to", "China"],
            ["Jim", "works", "at", "Facebook"],
            ["Jared", "works", "at", "Yahoo"],
        ],
        "tags": [
            ["B-PER", "O", "B-LOC"],
            ["B-PER", "I-PER", "O", "O", "B-LOC"],
            ["B-PER", "O", "O", "B-ORG"],
            ["B-PER", "O", "O", "B-ORG"],
        ]
    }
    validation_dataset = Dataset.from_dict(validation_data)

    combined_dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})

    # Map tokenization function (without static padding)
    combined_dataset = combined_dataset.map(ner_tuner.tokenize_and_align_labels, batched=True)
    combined_dataset.set_format("torch")

    # Compute class weights from the raw training data.
    class_weights = compute_class_weights_from_data(train_data, AutoNERTuner.tag2id)

    # Initialize trainer with weighted loss.
    trainer = ner_tuner.new_trainer(combined_dataset, checkpoint_name, class_weights=class_weights)

    trainer.train()
    trainer.evaluate()

    # -----------------------------
    # Additional Evaluation: Token-level and Entity-level Classification Report
    # -----------------------------
    predictions_output = trainer.predict(combined_dataset["validation"])
    predictions = predictions_output.predictions
    labels = predictions_output.label_ids

    # For token-level report, flatten predictions and labels.
    flat_preds = []
    flat_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(np.argmax(pred_seq, axis=-1), label_seq):
            if label != -100:
                flat_preds.append(pred)
                flat_labels.append(label)

    token_report = classification_report(
        [AutoNERTuner.id2tag[label] for label in flat_labels],
        [AutoNERTuner.id2tag[pred] for pred in flat_preds],
        zero_division=0
    )
    logger.info(f"Token-level Classification Report:\n{token_report}")

    # Generate entity-level report using seqeval (already included in metrics)
    metrics = compute_metrics((predictions, labels))
    logger.info(f"Entity-level Classification Report:\n{metrics.get('entity_classification_report')}")

    trainer.save_model(final_checkpoint_name)
    tokenizer.save_pretrained(final_checkpoint_name)
