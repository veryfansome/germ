import logging
import torch

logger = logging.getLogger(__name__)


def predict_long_text(model, tokenizer, words, id2tag, device, max_length=512, stride=128):
    """
    Predict POS tags for a long text (list of words) using a sliding window.

    Parameters:
      - model: the fine-tuned token classification model.
      - tokenizer: the corresponding tokenizer.
      - words: list of tokens/words (e.g., ["The", "cat", "sat", ...]).
      - id2tag: mapping from label ids to tag names.
      - device: torch.device.
      - max_length: maximum tokens per chunk (must be ≤ model max length).
      - stride: number of tokens to overlap between chunks.

    Returns:
      - A list of tuples (word, predicted_tag) for the original text.
    """

    # Tokenize the entire input with sliding-window chunking.
    # Setting return_overflowing_tokens=True causes the tokenizer to return multiple chunks
    # if the input exceeds max_length. Each chunk will be padded to max_length.
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        max_length=max_length,
        stride=stride,
        truncation=True,
        return_overflowing_tokens=True,
    )

    # A dictionary to accumulate predictions for each original word index.
    final_predictions = {}

    # The tokenizer will produce one or more chunks.
    num_chunks = len(encoding["input_ids"])
    for i in range(num_chunks):
        # Extract chunk i and add a batch dimension.
        input_ids = encoding["input_ids"][i].unsqueeze(0).to(device)
        attention_mask = encoding["attention_mask"][i].unsqueeze(0).to(device)

        # Run the model to obtain logits.
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # logits shape: (1, sequence_length, num_labels)
            logits = outputs.logits
            # Get predicted label IDs for this chunk.
            predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()

        # Get the mapping from each subword token in this chunk to its original word index.
        # (Returns a list of the same length as the chunk’s token sequence, with None for special tokens.)
        word_ids = encoding.word_ids(batch_index=i)

        # Loop over tokens in this chunk and record the prediction for each original word.
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # Skip special tokens ([CLS], [SEP], or padding).
            # If this word has not been assigned a prediction yet, record it.
            if word_id not in final_predictions:
                final_predictions[word_id] = predictions[token_idx]
            # Otherwise, you could combine predictions (e.g., majority vote) if desired.
            # For simplicity, we'll keep the prediction from its first occurrence.

    # Now, build the final result: a list pairing each original word with its predicted tag.
    merged_predictions = []
    for idx, word in enumerate(words):
        # If for some reason a word wasn’t predicted (shouldn't happen), assign a default tag.
        tag_id = final_predictions.get(idx, None)
        tag = id2tag[tag_id] if tag_id is not None else "O"
        merged_predictions.append((word, tag))

    return merged_predictions


if __name__ == "__main__":
    from bot.train.auto_pos_tuner import AutoPosTuner, get_model, get_tokenizer
    from observability.logging import setup_logging

    setup_logging()

    deberta_checkpoint_name = "deberta-pos"
    deberta_model = get_model(f"{AutoPosTuner.checkpoint_dir}/{deberta_checkpoint_name}/final")
    deberta_tokenizer = get_tokenizer(f"{AutoPosTuner.checkpoint_dir}/{deberta_checkpoint_name}/final")

    id2tag = deberta_model.config.id2label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deberta_model.to(device)
    deberta_model.eval()

    # Example long text (here simply splitting a string into words).
    # In practice, your text might be much longer.
    new_text = (
        "This is a long text example that might exceed the typical 512-token limit. "
        "It is being used to demonstrate the sliding window approach for making predictions. "
        "Each segment will overlap to preserve context, ensuring no part of the text is lost. "
        "The fine-tuned model will output POS tags for each word accordingly."
    )
    words = new_text.split()  # Simple splitting; consider using a robust tokenizer for real texts.

    # Predict POS tags for the new text.
    predicted_tags = predict_long_text(deberta_model, deberta_tokenizer, words, id2tag, device,
                                       max_length=512, stride=128)

    # Print results.
    print("Predictions:")
    for word, tag in predicted_tags:
        print(f"{word} -> {tag}")