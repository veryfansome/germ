from transformers import DebertaV2TokenizerFast
import torch

from models.predict.multi_head_model import MultiHeadModel
from models.utils import get_torch_device, sp_tokenize


class MultiHeadPredictor:
    def __init__(self, model_name_or_path: str, subfolder=None):
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained(
            model_name_or_path, add_prefix_space=True, subfolder=subfolder)
        self.model = MultiHeadModel.from_pretrained(
            model_name_or_path, subfolder=subfolder)
        self.id2label = self.model.config.label_maps

        self.device = get_torch_device()
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        """
        Perform multi-headed token classification on a single piece of text.

        :param text: The raw text string.

        :return: A dict with {head_name: [predicted_label_for_each_token]} for the tokens in `text`.
        """
        raw_tokens = sp_tokenize(text)

        # We'll do a single-example batch to replicate training chunk logic.
        # is_split_into_words=True => we pass a list of tokens, not a single string.
        # This returns possibly multiple overflows if the sequence is long:
        encoded = self.tokenizer(
            raw_tokens,
            is_split_into_words=True,
            max_length=512,
            stride=128,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length"
        )

        # 'overflow_to_sample_mapping' indicates which chunk maps back to this example's index
        # For a single example, they should all map to 0, but let's handle it anyway:
        sample_map = encoded.get("overflow_to_sample_mapping", [0] * len(encoded["input_ids"]))

        # We'll store predictions for each chunk, then reconcile them.
        chunk_preds = []
        chunk_word_ids = []

        # Model forward:
        # We iterate over each chunk, move them to device, and compute logits_dict.
        for i in range(len(encoded["input_ids"])):
            # Build a batch of size 1 for chunk i
            input_ids_tensor = torch.tensor([encoded["input_ids"][i]], dtype=torch.long).to(self.device)
            attention_mask_tensor = torch.tensor([encoded["attention_mask"][i]], dtype=torch.long).to(self.device)

            # The model forward returns logits_dict since we don't provide labels_dict
            with torch.no_grad():
                logits_dict = self.model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor
                )  # shape for each head: (1, seq_len, num_labels)

            # Convert each head's logits to predicted IDs
            # logits_dict is {head_name: Tensor of shape [1, seq_len, num_labels]}
            pred_ids_dict = {}
            for head_name, logits in logits_dict.items():
                # shape (1, seq_len, num_labels)
                preds = torch.argmax(logits, dim=-1)  # => shape (1, seq_len)
                # Move to CPU numpy
                pred_ids_dict[head_name] = preds[0].cpu().numpy().tolist()

            # Keep track of predicted IDs + the corresponding word_ids for alignment
            chunk_preds.append(pred_ids_dict)

            # Also store the chunk's word_ids (so we can map subwords -> actual token index)
            # Note: you MUST call `tokenizer.word_ids(batch_index=i)` with is_split_into_words=True
            # which is only available on a batched encoding. So we re-call it carefully:
            word_ids_chunk = encoded.word_ids(batch_index=i)
            chunk_word_ids.append(word_ids_chunk)

        # Now we combine chunk predictions into a single sequence of token-level labels.
        # Because we used a sliding window, tokens appear in multiple chunks. We can
        # keep the first occurrence, or we might want to carefully handle overlaps.
        # Below is a simplistic approach: We will read each chunk in order, skipping
        # positions with word_id=None or repeated word_id (subword).

        # We'll build final predictions for each head at the *token* level (not subword).
        # For each original token index from 0..len(raw_tokens)-1, we pick the first chunk
        # that includes it, and the subword=first-subword label.

        # We define an array of "final predictions" for each head, size = len(raw_tokens).
        final_pred_labels = {**{
            "text": text,
            "tokens": raw_tokens,
        }, **{
            head: ["O"] * len(raw_tokens)  # or "O" or "" placeholder
            for head in self.id2label.keys()
        }}

        # We'll keep track of which tokens we've already assigned. Each chunk is
        # processed left-to-right, so effectively the earliest chunk covers it.
        assigned_tokens = set()

        for i, pred_dict in enumerate(chunk_preds):
            w_ids = chunk_word_ids[i]
            for pos, w_id in enumerate(w_ids):
                if w_id is None:
                    # This is a special token (CLS, SEP, or padding)
                    continue
                if w_id in assigned_tokens:
                    # Already assigned from a previous chunk
                    continue

                # If it's the first subword of that token, record the predicted label for each head.
                # pred_dict[head_name] is a list of length seq_len
                for head_name, pred_ids in pred_dict.items():
                    label_id = pred_ids[pos]
                    label_str = self.id2label[head_name][label_id]
                    final_pred_labels[head_name][w_id] = label_str

                assigned_tokens.add(w_id)

        return final_pred_labels


if __name__ == "__main__":
    predictor = MultiHeadPredictor("veryfansome/multi-classifier", subfolder="models/o3-mini_20250218")

    test_cases = [
        "How to convince my parents to let me get a Ball python?",
    ]
    for case in test_cases:
        predictions = predictor.predict(case)
        for head_name, labels in predictions.items():
            print(f"{head_name}: {labels}")
