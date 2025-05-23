import collections
import json
import math
import numpy as np
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.util import ngrams as nltk_ngrams
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from typing import List

"""
* Components -------------------------------------------------------------

  • **TokenEmbeddingModel**
      – wraps **intfloat/e5-base-v2** and adds a linear *token head* that
        produces order‑aware, L2‑normalised embeddings for *single words*.

  • **fine_tune_token_head()** (scriptable entry point)
      – self‑supervised contrastive training on WikiText‑103 to warm‑up the
        token head before the controller is trained.
"""


class TokenEmbeddingModel(nn.Module):
    """Wraps **intfloat/e5-base-v2** and adds a projection head so that
    *individual words* (tokens) get a reusable, order‑aware embedding.

    The base encoder is optionally frozen for cheap fine‑tuning.
    """

    def __init__(self,
                 #base_model_name: str = "intfloat/e5-base-v2",
                 #embed_dim: int = 768,
                 #proj_hidden: int = 512,
                 base_model_name: str = "intfloat/e5-large-v2",
                 #embed_dim: int = 1024,  # e5-large-v2 has 1024 dimensions
                 #proj_hidden: int = 1536,  # embed_dim x 1.5
                 embed_dim: int = 2048,
                 proj_hidden: int = 3072,  # embed_dim x 1.5
                 freeze_base: bool = True,
                 device: torch.device | str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        # base encoder
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name).to(self.device)
        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # 2‑layer MLP head
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, proj_hidden),
            nn.GELU(),
            nn.LayerNorm(proj_hidden),
            nn.Linear(proj_hidden, embed_dim, bias=False),
        ).to(self.device)
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        self._embed_dim = embed_dim

    @property
    def embed_dim(self):
        return self._embed_dim

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Return **L2‑normalised** embeddings for *texts* (list of words)."""
        batch = self.tokenizer(
            [f"passage: {t}" for t in texts],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.encoder(**batch).last_hidden_state[:, 0, :]  # CLS
        vec = F.normalize(self.proj(out), p=2, dim=-1)
        return vec

    # Convenience wrappers ---------------------------------------------------
    def __call__(self, texts: List[str]) -> torch.Tensor:  # noqa: D401
        return self.encode(texts)


class WordDataset(Dataset):
    """A simple list‑of‑tokens dataset built from WikiText‑103."""

    def __init__(self, tokens: List[str]):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


def build_vocab(
        max_vocab: int = 50000,
        min_entity_freq: int = 100,
        min_ngram_freq: int = 300,
        min_vocab_freq: int = 100,
        ngram_max: int = 7,
        ngram_min: int = 2,
) -> (List[str], List[str]):
    """Extract a vocabulary from *WikiText‑103‑raw‑v1* via datasets."""
    from datasets import load_dataset
    print(f"Loading dataset")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    ds_size = len(ds)
    print(f"Selecting training vocabulary from {ds_size} documents")

    single_lowercase_token_pattern = re.compile(r"^[a-z]+$")
    starts_with_uppercase_pattern = re.compile(r'^[A-Z]')

    ngram_counter = collections.Counter()
    consecutive_capitalized_words_counter = collections.Counter()
    single_token_counter = collections.Counter()

    _bracket_tokens = {"(", ")", "[", "]", "{", "}"}
    _contraction_tokens = {"'d", "'ll", "'m", "'re", "'s", "'t", "'ve"}
    _numeric_suffix_tokens = {"%", "°"}
    _special_at_tokens = {"@,@", "@.@", "@-@"}

    ngram_stop_tokens = {
        *_bracket_tokens,
        *_contraction_tokens,
        *_numeric_suffix_tokens,
        *_special_at_tokens,
        *{
            ",", "...", ".", "?", "!", ":", ";",
            "’", "'", '"',
            "=", "±", "+", "-", "–", "—", "×",
            "⁄", "/", "\\", "$", "£", "&",
            "he", "his", "she", "her",
            "a.m.", "p.m.",
        }
    }

    is_bracket_tokens = 1
    is_contraction_token = 2
    is_special_at_token = 3
    is_starts_with_upper = 4
    for row_id, row in enumerate(ds):
        anomalous_tokens = {}
        uniformly_lowercase_tokens = {}

        tokens = row["text"].split()
        token_len = len(tokens)
        last_token_idx = token_len - 1

        #consecutive_capitalized_words = consecutive_capitalized_words_pattern.findall(row["text"])
        #consecutive_capitalized_words_counter.update(consecutive_capitalized_words)

        idx_offset = 0
        skip_to_idx = None
        for token_idx, token in enumerate(tokens):
            if skip_to_idx is not None and token_idx < skip_to_idx:
                continue
            skip_to_idx = None
            token_idx -= idx_offset
            if bool(single_lowercase_token_pattern.search(token)):
                next_token_idx = token_idx + 1
                if next_token_idx < token_len and (tokens[next_token_idx] in _contraction_tokens
                                                   or tokens[next_token_idx] == "@-@"):
                    if tokens[next_token_idx] in _contraction_tokens:
                        if tokens[next_token_idx] != "'s":
                            uniformly_lowercase_tokens[token_idx] = token + tokens[next_token_idx]
                            idx_offset = 2
                            skip_to_idx = next_token_idx + 1
                        else:
                            uniformly_lowercase_tokens[token_idx] = token
                            uniformly_lowercase_tokens[next_token_idx] = tokens[next_token_idx]
                    elif tokens[next_token_idx] == "@-@":
                        joined_token = token
                        cursor_idx = next_token_idx + 1
                        while cursor_idx is not None and tokens[cursor_idx - 1] == "@-@":
                            joined_token += f"-{tokens[cursor_idx]}"
                            next_cursor_idx = cursor_idx + 2
                            if next_cursor_idx < last_token_idx:
                                cursor_idx = next_cursor_idx
                            else:
                                idx_offset = cursor_idx - token_idx
                                skip_to_idx = cursor_idx
                                cursor_idx = None
                        uniformly_lowercase_tokens[token_idx] = joined_token
                else:
                    # If uniformly lowercase and alpha, just append
                    uniformly_lowercase_tokens[token_idx] = token
            elif token in _bracket_tokens:
                anomalous_tokens[token_idx] = is_bracket_tokens
            elif token in _contraction_tokens:
                anomalous_tokens[token_idx] = is_contraction_token
            elif token in _special_at_tokens:
                anomalous_tokens[token_idx] = is_special_at_token
            elif bool(starts_with_uppercase_pattern.search(token)):
                anomalous_tokens[token_idx] = is_starts_with_upper
        single_token_counter.update(uniformly_lowercase_tokens.values())

        if uniformly_lowercase_tokens:
            uniformly_lowercase_ngrams = []
            for token_group in group_by_consecutive_keys(uniformly_lowercase_tokens):
                group_len = len(token_group)
                for n in range(ngram_min, ngram_max + 1):
                    if group_len == n:
                        uniformly_lowercase_ngrams.append(" ".join(token_group))
                    elif ngram_min < group_len < n:
                        for ngram_group in nltk_ngrams(token_group, group_len):
                            uniformly_lowercase_ngrams.append(" ".join(ngram_group))
            ngram_counter.update(uniformly_lowercase_ngrams)

        if row_id % 100000 == 0:
            print(f"1st pass {row_id}/{ds_size} rows")

    entities = [w for w, c in consecutive_capitalized_words_counter.items() if c > min_entity_freq]
    entities = sorted(entities, key=consecutive_capitalized_words_counter.get, reverse=True)[:max_vocab]
    random.shuffle(entities)
    print(f"entities: {len(entities)}")
    vocab = [w for w, c in single_token_counter.items() if c >= min_vocab_freq]
    vocab = sorted(vocab, key=single_token_counter.get, reverse=True)[:max_vocab]
    random.shuffle(vocab)
    print(f"vocab: {len(vocab)}")
    ngrams = [w for w, c in ngram_counter.items() if c > min_ngram_freq]
    ngrams = sorted(ngrams, key=ngram_counter.get, reverse=True)[:max_vocab]
    random.shuffle(ngrams)
    print(f"ngrams: {len(ngrams)}")
    return entities, ngrams, vocab


def fine_tune_token_head(
        device: str | None = None,
        #epochs: int = 5,
        epochs: int = 8,
        #epochs: int = 16,
        #epochs: int = 24,
        #epochs: int = 32,
        #epochs: int = 40,
        #batch_size: int = 256,
        batch_size: int = 512,
        #lr: float = 1e-4,
        #lr: float = 5e-5,
        lr: float = 1e-5,
        out_dir: str = "data/e5_token_embedding_model",
        seed: int = 42,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    if device is None:
        device = (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
    device_t = torch.device(device)
    model = TokenEmbeddingModel(freeze_base=True, device=device_t).to(device_t)

    # Each training loop iteration processes a batch of tokens. The training objectives are:
    # - Match each word with itself:
    #     Each word's embedding is compared to itself, ensuring that the representation of a
    #     word remains the same. This “positive” pair (word with itself) is ideally the most similar.
    # - Differentiate each word from other words:
    #     Find the most "confusing" other word in the batch — i.e., a word whose embedding is most similar (even though
    #     it’s not supposed to be the same). This confusing word becomes the “negative” example. The model is then
    #     penalized if it mixes up these representations.
    entities, ngrams, vocab = build_vocab()
    with (Path(out_dir)/"entities.json").open("w", encoding="utf-8") as json_f:
        json.dump(entities, json_f, ensure_ascii=False, indent=0)
    with (Path(out_dir)/"ngrams.json").open("w", encoding="utf-8") as json_f:
        json.dump(ngrams, json_f, ensure_ascii=False, indent=0)
    with (Path(out_dir)/"vocab.json").open("w", encoding="utf-8") as json_f:
        json.dump(vocab, json_f, ensure_ascii=False, indent=0)
    dataset = WordDataset(entities + ngrams + vocab)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(dataset, batch_size=batch_size, generator=g, shuffle=True, num_workers=0)

    optimiser = torch.optim.AdamW(model.proj.parameters(), lr=lr, weight_decay=1e-2)

    accum_steps = 4  # ← batch_size 256×4 = 1024 token pairs per “virtual batch”

    # A temperature parameter (tau) is used to adjust the sensitivity of similarity comparisons. It starts higher and
    # is annealed to a lower value during training. This helps fine-tune how sharply the model distinguishes between
    # the correct word and its hardest negative counterpart.
    base_tau, final_tau = 0.075, 0.02

    total_iter = epochs * len(dl)
    scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        0.05 * total_iter,  # warmup steps
        total_iter
    )

    model.train()
    optimiser.zero_grad(set_to_none=True)
    for ep in range(epochs):
        running_loss = 0.0
        for step, batch_tokens in enumerate(dl):
            it = ep * len(dl) + step
            tau = final_tau + 0.5 * (base_tau - final_tau) * (1 + math.cos(math.pi * it / total_iter))

            # The model's `encode` method generates embeddings for all tokens in the batch. These embeddings are
            # L2-normalized since we use cosine similarity.
            embeds = model(batch_tokens)  # (B,d)

            # For each anchor embedding, cosine similarities with all embeddings in the batch are computed.
            # The similarity with itself is ignored by filling it with a large negative value (-1.0). The hardest
            # negative embedding is selected for each anchor by identifying the index of the maximum similarity value
            # that is not from its own embedding.
            with torch.no_grad():
                sim = embeds @ embeds.T  # cosine similarities
                sim.fill_diagonal_(-1.0)  # ignore self-sim
                hard_idx = sim.argmax(dim=-1)  # index of hardest negative for each anchor
            hard_neg = embeds[hard_idx]  # (B,d)

            B = embeds.size(0)
            z_anchor = embeds  # (B,d)
            z_pos = embeds  # positive = self
            z_neg = hard_neg  # hard negatives

            # build the (B×(B+1)) similarity matrix:   [ pos | negs ]
            candidates = torch.cat([z_pos.unsqueeze(1), z_neg.unsqueeze(1)], dim=1)  # (B,2,d)
            candidates = candidates.view(-1, embeds.size(-1))  # (2B,d)

            logits = (z_anchor @ candidates.T) / tau  # (B, 2B)
            labels = torch.arange(B, device=device_t)  # each row’s positive is at column i*2
            labels = labels * 2

            # Contrastive loss is computed using a cross-entropy style setup. For each element in the batch, positive
            # examples are the embeddings themselves, while the hardest negatives are pooled. Loss is scaled down by
            # `accum_steps` # to support gradient accumulation.
            loss = F.cross_entropy(logits, labels) / accum_steps

            # To encourage diversity, a regularization term is added to the loss. It penalizes the overlap
            # (non-orthogonality) between the learned projections in the MLP head's final layer, to keep the different
            # dimensions as independent as possible. This means the model is nudged to learn representations where
            # features don’t overlap too much, making them more robust and less redundant.
            #W = F.normalize(model.proj[-1].weight, dim=1)
            #loss += 1e-4 * (W @ W.T).pow(2).triu(1).mean() / accum_steps

            # Gradients are computed for parameter updates. A gradient accumulation strategy is employed every
            # `accum_steps` micro-steps to form a "virtual batch". Gradients are clipped to prevent exploding gradients,
            # the optimizer updates the model's parameters, and the learning rate schedule steps forward.
            loss.backward()
            running_loss += loss.item() * accum_steps  # un-scale for logging

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # optional
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                scheduler.step()  # one LR-update per *optim* step, not per micro-step

            if step % 50 == 0:
                print(f"epoch {ep + 1}/{epochs} step {step}/{len(dl)} "
                      f"loss {running_loss / max(1, step + 1):.4f}")
        print(f"→ epoch {ep+1} mean loss {running_loss/len(dl):.4f}")

    torch.save(model.state_dict(), Path(out_dir)/"token_head.pt")
    print(f"Saved fine‑tuned TokenEmbeddingModel to {out_dir}/token_head.pt")


def group_by_consecutive_keys(data):
    # Sort the dictionary keys
    sorted_keys = sorted(data.keys())

    grouped_values = []
    current_group = [data[sorted_keys[0]]]

    # Track previous key to detect consecutive runs
    previous_key = sorted_keys[0]
    for key in sorted_keys[1:]:
        # If the current key is consecutive to the previous key,
        # append the corresponding value to the current group
        if key == previous_key + 1:
            current_group.append(data[key])
        else:
            # If not consecutive, start a new group
            grouped_values.append(current_group)
            current_group = [data[key]]
        previous_key = key

    # Don't forget to add the last group
    grouped_values.append(current_group)
    return grouped_values


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)  # covers CUDA & MPS
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    fine_tune_token_head()
