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
                overwrite_output_dir=True,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                learning_rate=5e-5,
                logging_dir="logs",
                logging_steps=50,
                # Add more if needed
            ),
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True
        )

        labels = []
        for i, words in enumerate(examples["tokens"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # map subwords to word indices
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    # This is a special token like [CLS], [SEP], or padding
                    label_ids.append(-100)
                else:
                    # Align label to the first subword
                    label_ids.append(self.tag2id[examples["tags"][i][word_id]])
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


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

    deberta_model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/deberta-base",
        num_labels=len(AutoPosTuner.tag2id),
        id2label=AutoPosTuner.id2tag,
        label2id=AutoPosTuner.tag2id
    ),
    deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
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
    combined_dataset_dict = combined_dataset_dict.map(lambda x: pos_tuner.tokenize_and_align_labels(x), batched=True)
    combined_dataset_dict.set_format("torch")
    pos_trainer = pos_tuner.new_trainer(combined_dataset_dict, "deberta-pos")

