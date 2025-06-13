import logging
import torch
from transformers import DebertaV2TokenizerFast

from germ.services.models.predict.multi_head_model import MultiHeadModel
from germ.utils import get_torch_device
from germ.utils.tokenize import naive_tokenize

logger = logging.getLogger(__name__)


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

    def predict_batch(self, texts: list[str]):
        """
        Perform multi-headed token classification for a batch of texts.
        Returns a list of prediction dicts, one per input text.
        """
        # Split each text into raw tokens
        all_raw_tokens = [naive_tokenize(t) for t in texts]

        # Tokenize in batch mode with is_split_into_words=True. This automatically keeps track of the sample index via
        # 'overflow_to_sample_mapping'.
        encoded = self.tokenizer(
            all_raw_tokens,
            is_split_into_words=True,
            max_length=512,
            stride=128,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length"
        )

        # sample_map[i] tells us which text in the batch chunk i belongs to
        sample_map = encoded.get("overflow_to_sample_mapping", [])
        n_chunks = len(encoded["input_ids"])

        # Do one forward pass for all chunks.
        input_ids_tensor = torch.tensor(encoded["input_ids"], dtype=torch.long).to(self.device)
        attention_mask_tensor = torch.tensor(encoded["attention_mask"], dtype=torch.long).to(self.device)

        with torch.no_grad():
            # model(...) returns logits_dict = {head_name: [batch_size, seq_len, num_labels]}
            logits_dict = self.model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor
            )

        # logits_dict is {head_name: (n_chunks, seq_len, num_labels)}.
        # We'll do argmax across dim=-1 for each head:
        pred_ids_dict = {}
        for head_name, logits in logits_dict.items():
            pred_ids_dict[head_name] = torch.argmax(logits, dim=-1).cpu().numpy()

        # Build final predictions for each text based on the chunk predictions. We'll accumulate predictions per text,
        # then combine them token-by-token.
        final_predictions = []
        for text_idx, raw_tokens in enumerate(all_raw_tokens):
            # For each text, set up a dictionary similar to your single-text predict
            final_pred_labels = {
                "text": texts[text_idx],
                "tokens": raw_tokens,
            }
            # Add a placeholder label list for each head
            for head_name in self.id2label.keys():
                final_pred_labels[head_name] = ["X"] * len(raw_tokens)  # or some default
            final_predictions.append(final_pred_labels)

        # Keep track which tokens were assigned for each text
        assigned_tokens_for_text = [set() for _ in texts]

        # Because we asked for word_ids in “batched” format, we need to call:
        #   tokenizer.word_ids(batch_index=some_index)
        # but that is only valid one chunk at a time. We'll do that in a loop:
        for chunk_idx in range(n_chunks):
            # Determine which text this chunk belongs to
            text_idx = sample_map[chunk_idx]
            word_ids_chunk = encoded.word_ids(batch_index=chunk_idx)

            # For each position in the chunk, map to a token ID in the text. Then set the label.
            for pos, w_id in enumerate(word_ids_chunk):
                if w_id is None:
                    continue  # special token (CLS, SEP, padding, etc.)
                if w_id in assigned_tokens_for_text[text_idx]:
                    continue  # already assigned from an earlier chunk

                # For each head, look up the predicted label:
                for head_name in pred_ids_dict.keys():
                    label_id = pred_ids_dict[head_name][chunk_idx][pos]
                    label_str = self.id2label[head_name][label_id]
                    final_predictions[text_idx][head_name][w_id] = label_str

                assigned_tokens_for_text[text_idx].add(w_id)

        return final_predictions


def log_pos_labels(pos_labels: dict[str, list[str]]):
    token_cnt = len(pos_labels["tokens"])
    token_idx_positions = range(token_cnt)
    longest_token_lengths = [0 for _ in token_idx_positions]
    for idx in token_idx_positions:
        # Get longest token lengths per position
        for head in pos_labels.keys():
            if head == "text":
                continue
            else:
                label_len = len(pos_labels[head][idx])
                if label_len > longest_token_lengths[idx]:
                    longest_token_lengths[idx] = label_len
    log_blobs = []
    for head, labels in pos_labels.items():
        # Legible formatting for examples
        if head == "text":
            log_blobs.append(f"{head}{' ' * (12 - len(head))}{labels}")
            positions_blob = ''.join([
                f"{l},{' ' * (longest_token_lengths[i] - len(str(l)) + 3)}" if i != token_cnt - 1 else str(
                    l)
                for i, l in enumerate(token_idx_positions)])
            log_blobs.append(f"idx{' ' * 9} {positions_blob}")
        else:
            label_blobs = []
            for idx, label in enumerate(labels):
                label_blobs.append(
                    f"\"{label}\",{' ' * (longest_token_lengths[idx] - len(label) + 1)}" if idx != token_cnt - 1 else f"\"{label}\"")
                if head == "tokens":
                    continue
            log_blobs.append(f"{head}{' ' * (12 - len(head))}[{''.join(label_blobs)}]")
    logger.info(f"pos labels:\n" + ("\n".join(log_blobs)))


if __name__ == "__main__":
    from germ.observability.logging import setup_logging
    setup_logging()

    predictor = MultiHeadPredictor(
        "veryfansome/multi-classifier", subfolder="models/ud_ewt_gum_pud_20250611")
    test_cases = [
        "Hello world!",
        "How should I convince my parents to let me get a Ball python?",
    ]
    for prediction in predictor.predict_batch(test_cases):
        log_pos_labels(prediction)

