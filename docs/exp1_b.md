I'm working on a project to attempt to build an encoder model that can generate dense sentence embeddings that is also able to distinguish words and multi-word phrases in a way that maximize invertibility. That is to say, given a sentence embedding, another model can be trained to generate text that is as close as possible to the original semantically and syntactically.

Here is the code I have so far:
```
import collections
import json
import math
import numpy as np
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                 base_model_name: str = "intfloat/e5-large-v2",
                 embed_dim: int = 768,
                 proj_hidden: int = 512,
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


def build_vocab(min_freq: int = 100, max_vocab: int = 50000, seed: int = 42) -> (List[str], List[str]):
    """Extract a vocabulary from *WikiText‑103‑raw‑v1* via datasets."""
    from datasets import load_dataset
    print(f"Loading dataset")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    print(f"Selecting training vocabulary from {len(ds)} documents")
    consecutive_upper_words_counter = collections.Counter()
    single_word_counter = collections.Counter()

    all_caps_pattern = re.compile(r'\b[A-Z]+\b')
    consecutive_upper_words_pattern = re.compile(r"(?:^|\b)(?!As\b|By\b|In\b|On\b|The\b)[A-Z][a-z]+(?:\s+of)?(?:\s+[A-Z][a-z]+)+\b")
    single_word_pattern = re.compile(r"\b(?:a|I|[a-zA-Z]{2,})\b")
    upper_pattern = re.compile(r'[A-Z]')
    for row in ds:
        consecutive_upper_words = consecutive_upper_words_pattern.findall(row["text"])
        consecutive_upper_words_counter.update(consecutive_upper_words)
        single_words = single_word_pattern.findall(row["text"])
        single_word_counter.update(single_words)
    # Populate word_count dict with only lowercase words
    word_count = {w: c for w, c in single_word_counter.items() if not bool(upper_pattern.search(w))}
    for w, c in single_word_counter.items():
        if bool(all_caps_pattern.search(w)):
            # Give benefit of doubt and assume acronym
            word_count[w] = c
        elif bool(upper_pattern.search(w)):
            # For words with uppercase characters
            lowered_form = w.lower()
            if lowered_form in word_count:
                # If a lowercase form exists, credit the lowercase form. This handles cases where common words comes
                # at the beginning of a sentence or where a common word is used in a name. We don't need to be too
                # strict since we're just trying to get a sense of frequency for train our projection head.
                word_count[lowered_form] += c
            else:
                # but if no lowercase versions were found, keep the uppercase word
                word_count[w] = c
    # Filter out anything that doesn't pass min_freq threshold
    named_entities = [w for w, c in consecutive_upper_words_counter.items() if c > min_freq]
    named_entities = sorted(named_entities, key=consecutive_upper_words_counter.get, reverse=True)[:max_vocab]
    random.shuffle(named_entities)
    vocab = [w for w, c in word_count.items() if c >= min_freq]
    vocab = sorted(vocab, key=word_count.get, reverse=True)[:max_vocab]
    random.shuffle(vocab)
    print(f"Vocab: {len(vocab)}")
    print(f"Named entities: {len(named_entities)}")
    return vocab, named_entities


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
    vocab, named_entities = build_vocab(seed=seed)
    with (Path(out_dir)/"vocab.json").open("w", encoding="utf-8") as json_f:
        json.dump(vocab, json_f, ensure_ascii=False, indent=0)
    with (Path(out_dir)/"named_entities.json").open("w", encoding="utf-8") as json_f:
        json.dump(named_entities, json_f, ensure_ascii=False, indent=0)
    dataset = WordDataset(vocab + named_entities)
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
            W = F.normalize(model.proj[-1].weight, dim=1)
            loss += 1e-4 * (W @ W.T).pow(2).triu(1).mean() / accum_steps

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


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)  # covers CUDA & MPS
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    fine_tune_token_head()
```

Review the code and suggest improvements - keeping in mind that we are after semantics, syntax, and invertibility rather than pure similarity or clustering. Use o1 to think deeply about possible improvements if you have a tool to do so

---

# Response 1

**Review and Suggestions for Improved Invertibility and Semantic/Syntactic Fidelity**

Below are some considerations and potential changes that may help achieve more robust embeddings focused on (a) preserving fine‑grained syntactic/semantic details, and (b) enabling partial or full invertibility (i.e., guiding a downstream decoder or reconstruction step). These suggestions range from architecture tweaks to training objectives and data preparation.

---

### 1. Model Architecture and Training Objective

1. **Incorporate an Auxiliary Reconstruction Head (Autoencoder‑style):**
    - In addition to the contrastive‐style loss for word embedding, you could add a small decoder that attempts to reconstruct the input tokens (or entire fragments) from the projected embeddings.
    - One option is to gather the output embeddings of tokens, average or combine them (with positional information) to form a single hidden state that a small transformer or MLP decoder can turn back into tokens.
    - Minimizing the reconstruction loss encourages your model’s projection to preserve syntactic and semantic details that otherwise might be lost in purely contrastive training.

2. **Phrase‑Level Contrastive Training:**
    - Currently, the code focuses on single words. For improved invertibility/syntax, consider chunking the text into multi‑word segments (not just named entities). Then train a contrastive objective specifically at that chunk/phrase level.
    - This helps the model learn phrase embeddings that can more directly preserve the relationships among words in each phrase.

3. **Fine‑Tune (Unfreeze) Portions of the Base Model:**
    - Freezing the base encoder is efficient but may limit how well you can adapt the embeddings to preserve detailed syntax for invertibility.
    - Even unfreezing just the top few layers of the transformer (while leaving earlier layers frozen) can help the model adapt to this new objective while still conserving memory/training speed.

4. **Use a Margin or Hard‑Negative Mining Strategy Beyond In-Batch:**
    - You are already picking hardest negatives in the batch. But consider searching a larger memory bank of embeddings or using an external strategy for negative sampling. It can help refine the embeddings and better separate subtle differences needed for reconstructing.

---

### 2. Data Preparation

1. **Focus on More Granular Chunks (N-grams, Named Entities, and Phrases):**
    - Beyond named entities and single words, you could systematically generate phrases (2-grams, 3-grams, etc.) from your corpus.
    - This will let the model see more local context within a single embedding, improving the chance of capturing local syntax crucial for invertibility.

2. **Expand or Curate the Negative Examples:**
    - If you want the model to differentiate subtle syntactic differences (e.g., singular vs. plural, tense changes), you could add curated “confusion sets” of near‑synonyms or morphological variants.
    - This strategy ensures the embeddings encapsulate fine distinctions that a downstream decoder can rely on.

---

### 3. Forward Pass and Code Mechanics

1. **Dimensional Consistency and Optional Larger Projection:**
    - The code sets `embed_dim = 768` for E5‑Large, which has a 1024‑dim hidden size by default. Consider matching them (i.e., projecting to 1024) if you want to preserve maximal detail, or ensure the projection dimension is sufficiently large that important syntactic nuances aren’t lost.
    - You could also experiment with a 2× or 4× embed dimension in the projection if you want to store more “information,” subject to GPU memory limits.

2. **Consider a Multi‑Task Loss:**
    - You can keep your current contrastive objective and add an auxiliary loss (e.g., partial reconstruction or deep consistency). The final loss would be a sum of the contrastive component, reconstruction component, and possibly a diversity or orthogonality penalty.

3. **Optionally Remove or Adjust Orthogonality Penalty:**
    - By default, you have a penalty on `(W @ W.T).pow(2).triu(1)` to encourage orthogonal final projection weights. While this can be useful for a purely semantic representation, it might reduce the capacity to store fine syntactic details in overlapping dimensions.
    - If invertibility is a strong priority, you may want to reduce or remove this penalty (or try strictly smaller weight).

4. **Memory and Efficiency:**
    - If you handle phrases or longer contexts, you might need to reduce your batch size to avoid GPU/memory bottlenecks.
    - You might also consider gradient checkpointing in the transformer backbone if you unfreeze it.

---

### 4. Practical Invertibility Approaches

1. **Train a Small “Decoder” that Inverts Your Embeddings:**
    - After producing your word (or phrase) embeddings, feed them into a small generative model to predict text.
    - You can do this in parallel with your contrastive training. Over time, the embeddings will contain enough “clues” to let the decoder reconstruct text.

2. **Token/Position Fusion in the Embedding:**
    - Standard encoders degrade word‐order info if you only pull the [CLS] token or don’t incorporate positional signals. For strict invertibility, you need more explicit positional encoding or you must store offset/positional cues in the final embedding.
    - One approach: Instead of using just `last_hidden_state[:, 0, :]`, you could gather hidden states from each token and feed them into your projection or a separate aggregator that preserves ordering (e.g., a small RNN over the transformer outputs).

3. **Evaluate Reconstructed Quality Regularly:**
    - If invertibility is an explicit goal, measure the success of text reconstruction (e.g., BLEU or ROUGE with respect to original text) every few epochs. This helps you catch if the model is overfitting on a purely contrastive objective and ignoring reconstructability.

---

### 5. Final Thoughts

- Balancing semantic similarity, syntactic fidelity, and invertibility is a tricky multi‐objective problem.
- A purely contrastive approach (word‐to‐word) tends to compress language in ways that might lose reconstructive detail, unless you add a specialized reconstruction or multi‐task objective.
- Iterating on the ideas above—especially adding a small decoder, unfreezing some layers, and encoding multi‐token phrases—will push the model closer to your goals of robust semantic embedding and invertibility.

By integrating these suggestions, you should see improvements to how well the embeddings preserve not only broad similarity but also subtle syntactic markers that allow a separate (or integrated) decoder to reconstruct text more faithfully.