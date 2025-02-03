import logging
import torch

logger = logging.getLogger(__name__)


def predict_long_text_ner(model, tokenizer, words, id2tag, device, max_length=512, stride=128):
    """
    Predict NER tags for a long text (list of words) using a sliding-window approach.

    This function tokenizes the input text with sliding-window chunking (if the input exceeds max_length)
    and performs predictions on each chunk. It then maps the predicted labels back to the original words.
    For words that appear in multiple overlapping chunks, the prediction from the first occurrence is used.

    Parameters:
      - model: The fine-tuned token classification model for NER.
      - tokenizer: The corresponding tokenizer.
      - words: List of tokens/words (e.g., ["John", "lives", "in", "New", "York", ...]).
      - id2tag: Mapping from label IDs to NER tag names (e.g., {0: "O", 1: "B-PER", ...}).
      - device: torch.device to run inference on.
      - max_length: Maximum tokens per chunk (must be ≤ model max length).
      - stride: Number of tokens to overlap between chunks.

    Returns:
      - A list of tuples (word, predicted_tag) for the original text.
    """
    # Tokenize the entire input with sliding-window chunking.
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        max_length=max_length,
        stride=stride,
        truncation=True,
        return_overflowing_tokens=True,
    )

    # Dictionary to accumulate predictions for each original word index.
    final_predictions = {}
    num_chunks = len(encoding["input_ids"])

    for i in range(num_chunks):
        # Extract the i-th chunk and add a batch dimension.
        input_ids = encoding["input_ids"][i].unsqueeze(0).to(device)
        attention_mask = encoding["attention_mask"][i].unsqueeze(0).to(device)

        # Run inference.
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # Shape: (1, sequence_length, num_labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()

        # Get mapping from tokens in this chunk to their original word indices.
        word_ids = encoding.word_ids(batch_index=i)
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # Skip special tokens ([CLS], [SEP], or padding).
            # If this word hasn't been assigned a prediction yet, record it.
            if word_id not in final_predictions:
                final_predictions[word_id] = predictions[token_idx]
            # Optionally: combine overlapping predictions (e.g., majority vote) if desired.

    # Build the final result: list of (word, predicted_tag).
    merged_predictions = []
    for idx, word in enumerate(words):
        tag_id = final_predictions.get(idx, None)
        tag = id2tag[tag_id] if tag_id is not None else "O"
        merged_predictions.append((word, tag))

    return merged_predictions


def merge_ner_predictions(predictions):
    """
    Merge token-level NER predictions into contiguous named entities.

    Parameters:
      - predictions: List of tuples (word, tag).

    Returns:
      - List of tuples (entity_text, entity_tag, start_index, end_index) where
        start_index and end_index indicate the span of words forming the entity.
    """
    entities = []
    current_entity = []
    current_tag = None
    start_idx = None

    for i, (word, tag) in enumerate(predictions):
        if tag.startswith("B-"):
            # End any current entity.
            if current_entity:
                entities.append((" ".join(current_entity), current_tag, start_idx, i - 1))
                current_entity = []
            current_entity = [word]
            current_tag = tag[2:]  # Remove the "B-" prefix
            start_idx = i
        elif tag.startswith("I-") and current_entity and tag[2:] == current_tag:
            current_entity.append(word)
        else:
            # tag is "O" or an unexpected I- tag; finish any ongoing entity.
            if current_entity:
                entities.append((" ".join(current_entity), current_tag, start_idx, i - 1))
                current_entity = []
    # End of sequence: flush any remaining entity.
    if current_entity:
        entities.append((" ".join(current_entity), current_tag, start_idx, len(predictions) - 1))

    return entities


if __name__ == "__main__":
    # Adjust the import path according to your project structure.
    from bot.train.auto_ner_tuner import AutoNERTuner, get_model, get_tokenizer
    from observability.logging import setup_logging

    setup_logging()

    # Load the fine-tuned NER model and tokenizer.
    ner_checkpoint = "deberta-ner"  # Update checkpoint name as needed.
    ner_model = get_model(f"models/ner/{ner_checkpoint}/final")
    ner_tokenizer = get_tokenizer(f"models/ner/{ner_checkpoint}/final")

    # Retrieve the id2tag mapping from the model configuration.
    id2tag = ner_model.config.id2label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model.to(device)
    ner_model.eval()

    # Example long text for NER.
    new_text = (
        "John lives in New York and works at Google. "
        "He visited the Eiffel Tower in Paris last summer. "
        "Mary, his colleague from Microsoft, also went on the trip."
    )
    # A simple split is used here; in practice consider a more robust word tokenizer.
    words = new_text.split()

    # Predict NER tags for the new text.
    token_predictions = predict_long_text_ner(ner_model, ner_tokenizer, words, id2tag, device,
                                              max_length=512, stride=128)

    print("Token-level Predictions:")
    for word, tag in token_predictions:
        print(f"{word} -> {tag}")

    # Optionally, merge token-level predictions into named entity spans.
    entities = merge_ner_predictions(token_predictions)
    print("\nMerged Entities:")
    for entity, tag, start, end in entities:
        print(f"Entity: '{entity}', Tag: {tag}, Span: ({start}, {end})")
