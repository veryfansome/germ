from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)
from datasets import load_dataset

# Load your dataset
# Assume you have a dataset in the CoNLL format
dataset = load_dataset("conll2003")  # Replace with your dataset

# Load a pre-trained model and tokenizer
base4_model_name = "albert-base-v2"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(base4_model_name)
data_collator = DataCollatorForTokenClassification(tokenizer)

unique_labels = set()
for example in dataset['train']:
    unique_labels.update(example['ner_tags'])
model = AutoModelForTokenClassification.from_pretrained(base4_model_name, num_labels=len(unique_labels))


# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',  # or 'longest'
        max_length=128  # Set a maximum sentence length based on your needs
    )
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if id is None else label[id] for id in word_ids]  # Align labels with tokens
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


if __name__ == "__main__":
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./data/germ-ner",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()
