import torch
import logging

logger = logging.getLogger(__name__)


def predict_long_text_joint(model, tokenizer, words, pos_id2tag, ner_id2tag, device, max_length=512, stride=128):
    """
    Predict POS and NER tags for a long text (list of words) using a sliding window.

    Parameters:
      - model: the fine-tuned joint token classification model.
      - tokenizer: the corresponding tokenizer.
      - words: list of tokens/words (e.g., ["The", "cat", "sat", ...]).
      - pos_id2tag: mapping from POS label ids to tag names.
      - ner_id2tag: mapping from NER label ids to tag names.
      - device: torch.device.
      - max_length: maximum tokens per chunk (must be ≤ model max length).
      - stride: number of tokens to overlap between chunks.

    Returns:
      - A list of tuples (word, predicted_pos_tag, predicted_ner_tag) for the original text.
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

    # Dictionary to accumulate predictions by original word index.
    final_predictions = {}  # key: original word index; value: (pos_prediction, ner_prediction)

    num_chunks = len(encoding["input_ids"])
    for i in range(num_chunks):
        # Get a single chunk.
        input_ids = encoding["input_ids"][i].unsqueeze(0).to(device)
        attention_mask = encoding["attention_mask"][i].unsqueeze(0).to(device)

        # Run the model in evaluation mode.
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # Extract POS and NER logits. Expected shapes: (1, seq_len, num_labels)
            pos_logits = outputs["pos_logits"]
            ner_logits = outputs["ner_logits"]
            # Convert logits to predictions.
            pos_predictions = torch.argmax(pos_logits, dim=2)[0].cpu().numpy()
            ner_predictions = torch.argmax(ner_logits, dim=2)[0].cpu().numpy()

        # Map each token in this chunk back to its original word index.
        word_ids = encoding.word_ids(batch_index=i)
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # Skip special tokens ([CLS], [SEP], or padding)
            # If this word has not yet been assigned a prediction, record the predictions.
            if word_id not in final_predictions:
                final_predictions[word_id] = (pos_predictions[token_idx], ner_predictions[token_idx])
            # Otherwise, you might combine predictions (e.g. majority vote) in overlapping regions.
            # For simplicity, we keep the first occurrence.

    # Build the final list of predictions for each original word.
    merged_predictions = []
    for idx, word in enumerate(words):
        # If no prediction was recorded for a word, use a default.
        if idx in final_predictions:
            pos_pred_id, ner_pred_id = final_predictions[idx]
            pos_tag = pos_id2tag.get(pos_pred_id, "O")
            ner_tag = ner_id2tag.get(ner_pred_id, "O")
        else:
            pos_tag = "O"
            ner_tag = "O"
        merged_predictions.append((word, pos_tag, ner_tag))

    return merged_predictions


def report_state_dict_mismatch(model, checkpoint_path):
    """
    Loads the state dictionary from checkpoint_path and compares it to model.state_dict().
    Reports on missing keys, unexpected keys, and mismatched shapes.
    """
    # Determine how to load the checkpoint based on file extension.
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("Please install safetensors (`pip install safetensors`) to load safetensors files.")
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load state dict into model (without strict mode to capture mismatches).
    result = model.load_state_dict(state_dict, strict=False)

    logger.info("Missing keys:")
    for key in result.missing_keys:
        logger.info("  ", key)
    logger.info("Unexpected keys:")
    for key in result.unexpected_keys:
        logger.info("  ", key)

    # Check for shape mismatches.
    mismatched_keys = []
    model_dict = model.state_dict()
    for key, param in model_dict.items():
        if key in state_dict:
            if param.shape != state_dict[key].shape:
                mismatched_keys.append((key, param.shape, state_dict[key].shape))
    logger.info("Mismatched keys (shape differences):")
    for key, expected_shape, loaded_shape in mismatched_keys:
        logger.info(f"  {key}: expected {expected_shape}, got {loaded_shape}")

    logger.info(f"Summary: {len(result.missing_keys)} missing keys, {len(result.unexpected_keys)} unexpected keys, "
                f"{len(mismatched_keys)} mismatched keys.")


if __name__ == "__main__":
    from bot.train.auto_ner_pos_joint_tuner import JointPosNerTuner, get_joint_model, get_tokenizer
    from observability.logging import setup_logging

    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to your saved joint model.
    joint_model_path = "models/germ/deberta-joint/final"

    # Load the joint model and tokenizer.
    joint_model = get_joint_model(joint_model_path)
    # report_state_dict_mismatch(joint_model, f"{joint_model_path}/model.safetensors")
    #
    # Missing keys:
    #    encoder.embeddings.position_embeddings.weight
    #
    # Unexpected keys:
    #    encoder.encoder.rel_embeddings.weight
    #    encoder.encoder.layer.0.attention.self.pos_proj.weight
    #    encoder.encoder.layer.0.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.0.attention.self.pos_q_proj.weight
    #    encoder.encoder.layer.1.attention.self.pos_proj.weight
    #    encoder.encoder.layer.1.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.1.attention.self.pos_q_proj.weight
    #    encoder.encoder.layer.2.attention.self.pos_proj.weight
    #    encoder.encoder.layer.2.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.2.attention.self.pos_q_proj.weight
    #    encoder.encoder.layer.3.attention.self.pos_proj.weight
    #    encoder.encoder.layer.3.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.3.attention.self.pos_q_proj.weight
    #    encoder.encoder.layer.4.attention.self.pos_proj.weight
    #    encoder.encoder.layer.4.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.4.attention.self.pos_q_proj.weight
    #    encoder.encoder.layer.5.attention.self.pos_proj.weight
    #    encoder.encoder.layer.5.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.5.attention.self.pos_q_proj.weight
    #    encoder.encoder.layer.6.attention.self.pos_proj.weight
    #    encoder.encoder.layer.6.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.6.attention.self.pos_q_proj.weight
    #    encoder.encoder.layer.7.attention.self.pos_proj.weight
    #    encoder.encoder.layer.7.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.7.attention.self.pos_q_proj.weight
    #    encoder.encoder.layer.8.attention.self.pos_proj.weight
    #    encoder.encoder.layer.8.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.8.attention.self.pos_q_proj.weight
    #    encoder.encoder.layer.9.attention.self.pos_proj.weight
    #    encoder.encoder.layer.9.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.9.attention.self.pos_q_proj.weight
    #    encoder.encoder.layer.10.attention.self.pos_proj.weight
    #    encoder.encoder.layer.10.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.10.attention.self.pos_q_proj.weight
    #    encoder.encoder.layer.11.attention.self.pos_proj.weight
    #    encoder.encoder.layer.11.attention.self.pos_q_proj.bias
    #    encoder.encoder.layer.11.attention.self.pos_q_proj.weight
    #
    # Mismatched keys (shape differences):
    #
    # Summary: 1 missing keys, 37 unexpected keys, 0 mismatched keys.
    #
    # The report shows that one key (the position embeddings) is missing and that 37 keys (related to additional
    # relative position projection parameters) are unexpected. This is likely due to differences between my custom
    # model’s architecture and the saved checkpoint from my pretrained DeBERTa variant. If downstream task performance
    # is acceptable, these warnings can be safely ignored. Use ignore_mismatched_sizes=True to silence the warnings.

    joint_tokenizer = get_tokenizer(joint_model_path)
    joint_model.to(device)
    joint_model.eval()

    # Get label mappings.
    # Assuming your model's config stores the mappings; otherwise use the ones from your tuner.
    # Here we use the static mappings defined in JointPosNerTuner.

    pos_id2tag = JointPosNerTuner.pos_id2tag
    ner_id2tag = JointPosNerTuner.ner_id2tag

    # Example long text.
    new_text = (
        "John lives in New York and works at Apple. "
        "Mary visited Google in California while meeting with executives."
    )
    # Here we use a simple split. In production, consider using a robust word tokenizer.
    words = new_text.split()

    # Predict joint POS and NER tags.
    predictions = predict_long_text_joint(joint_model, joint_tokenizer, words,
                                          pos_id2tag, ner_id2tag, device,
                                          max_length=512, stride=128)

    # Print the predictions.
    logging.info("Joint Predictions (word, POS, NER):")
    for word, pos_tag, ner_tag in predictions:
        logging.info(f"{word} -> POS: {pos_tag}, NER: {ner_tag}")
