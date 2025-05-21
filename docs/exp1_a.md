I have a project that requires high quality dense embeddings and I'm tuning my own model for this.
- I plan to use the model's embeddings for sentence level semantic summaries as well as for single tokens representations.
- My ultimate goal is to train another model to predict vectors in this model's embedding space.
- Given an input "target" vector from this model's embedding space generated from some text (<40 words), I want the second model to predict token embeddings one at a time to incrementally rebuild something close to the original text
- I need strong sentence level representations with order awareness
- I need sharp token-level neighbors

I'm planning to follow a two-stage controller architecture like the one depicted below:
```
           target vector (sentence-level)
                    │
            ┌───────┴─────────┐
            │ Selector (Bi-enc)│  – predicts embedding of *word type*
            └───────┬─────────┘
                    │ intention vector
             FAISS nearest-k → candidate tokens
                    │
            ┌───────┴─────────┐
            │ Placer (Tiny Tr) │  – predicts position, may reorder
            └───────┬─────────┘
                    │ chosen (token, position)
         token buffer & hierarchical summary update
```

Here is the code for my embedding model and it's training loop:
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


def build_vocab(min_freq: int = 100, max_vocab: int = 50000, seed: int = 42) -> List[str]:
    """Extract a vocabulary from *WikiText‑103‑raw‑v1* via datasets."""
    from datasets import load_dataset
    print(f"Loading dataset")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    counter = collections.Counter()
    pattern = re.compile(r"[A-Za-z]+")
    print(f"Selecting training vocabulary from {len(ds)} documents")
    for row in ds:
        tokens = pattern.findall(row["text"].lower())
        counter.update(tokens)
    vocab = [w for w, c in counter.items() if c >= min_freq]
    vocab = sorted(vocab, key=counter.get, reverse=True)[:max_vocab]
    random.shuffle(vocab)
    print(f"Vocab size = {len(vocab)}")
    return vocab


def fine_tune_token_head(
        device: str | None = None,
        #epochs: int = 16,
        #epochs: int = 24,
        #epochs: int = 32,
        epochs: int = 40,
        batch_size: int = 256,
        #batch_size: int = 512,
        #lr: float = 1e-4,
        lr: float = 5e-5,
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

    vocab = build_vocab(seed=seed)
    with (Path(out_dir)/"vocab.json").open("w", encoding="utf-8") as json_f:
        json.dump(vocab, json_f, ensure_ascii=False, indent=0)
    dataset = WordDataset(vocab)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(dataset, batch_size=batch_size, generator=g, shuffle=True, num_workers=0)

    optimiser = torch.optim.AdamW(model.proj.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        1500,
        epochs * len(dl)
    )

    accum_steps = 4  # ← batch_size 256×4 = 1024 token pairs per “virtual batch”
    base_tau, final_tau = 0.1, 0.02
    total_iter = epochs * len(dl)

    model.train()
    optimiser.zero_grad(set_to_none=True)
    for ep in range(epochs):
        running_loss = 0.0
        for step, batch_tokens in enumerate(dl):
            it = ep * len(dl) + step
            tau = final_tau + 0.5 * (base_tau - final_tau) * (1 + math.cos(math.pi * it / total_iter))

            embeds = model(batch_tokens)  # (B,d)
            with torch.no_grad():
                sim = embeds @ embeds.T  # cosine similarities
                sim.fill_diagonal_(-1.0)  # ignore self-sim
                hard_idx = sim.argmax(dim=-1)  # index of hardest negative for each anchor
            hard_neg = embeds[hard_idx]  # (B,d)

            B = embeds.size(0)
            z_anchor = embeds  # (B,d)
            z_pos = embeds  # positive = self
            z_neg = hard_neg  # hard negatives

            # build the (B  ×  (B+1)) similarity matrix:   [ pos | negs ]
            candidates = torch.cat([z_pos.unsqueeze(1), z_neg.unsqueeze(1)], dim=1)  # (B,2,d)
            candidates = candidates.view(-1, embeds.size(-1))  # (2B,d)

            logits = (z_anchor @ candidates.T) / tau  # (B, 2B)
            labels = torch.arange(B, device=device_t)  # each row’s positive is at column i*2
            labels = labels * 2
            loss = F.cross_entropy(logits, labels) / accum_steps   # scale!

            # + orthogonality reg
            W = F.normalize(model.proj[-1].weight, dim=1)
            loss += 1e-4 * (W @ W.T).pow(2).triu(1).mean() / accum_steps

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

Review my code. My approach so far has limitations - e.g. the token head ignores multi-word expressions and it suffers from homograph conflation. Recommend solutions for these limitations. Use o1 to think deeply about a response if you have a tool to do so

---

# Response 1

**Review & Observations**

Your self-supervised approach for fine-tuning a token-level head on top of `intfloat/e5-*` is a good starting point for capturing order-aware representations for individual words. Here are a few specific observations about the code and overall approach:

1. **Training Objective**
    - You generate a contrastive objective by comparing each token’s embedding against “hard negatives” from within the same batch.
    - Hard-negative sampling is compelling, but keep in mind that you might accidentally push near-synonyms or morphological variants apart if they happen to appear in the same batch. A more nuanced negative selection or additional positives (e.g., from subword-similar tokens) could help.

2. **Architecture & Freezing**
    - You freeze the base backbone and only train the lightweight MLP head. This is efficient, but also limits the ability of your model to learn deeper token-level nuances (especially relevant when addressing homograph conflation). If you can afford it, unfreezing later layers of the backbone can help the model learn context-sensitive distinctions.

3. **Single-Token Inputs**
    - Each batch element in your current dataloader is just a single token string, passed as `"passage: <token>"`. As you’ve noted, this limits your embeddings to single words with no multi-word expressions. Additionally, it can lead to homograph conflation, because you have almost no surrounding context to disambiguate multiple senses of the same token.

---

## Limitations & Recommended Solutions

### 1. Multi-Word Expressions (MWEs)

**Limitation**  
With single-token inputs, your model cannot learn unique embeddings for multi-word constructs (e.g., “New York”, “prime minister”, “red herring”). These sequences often behave as single semantic units.

**Recommendations**

- **N-Gram Sampling**  
  Instead of feeding just single tokens, randomly sample and feed short n-grams of varying lengths from your corpus (e.g., up to 2–4 tokens). This way, your model’s head sees commonly co-occurring phrases. You can store them as simple strings, such as “red herring,” and your tokenizer will still handle them as subwords—but your projection MLP can learn to cluster their representations distinctly.

- **Phrase-Aware Vocab**  
  If your downstream tasks frequently involve certain bigrams, trigrams, or named entities (like “Los Angeles,” “United States of America,” etc.), consider building a phrase-level vocabulary. You might do this by:
    1. Extracting frequent co-occurrences from a large corpus.
    2. Treating these multi-word phrases as single vocab entries for your training.
    3. Feeding them through `model.encode([phrase])` so the final MLP head includes them in the same embedding space as individual words.

- **Sparsity vs. Coverage**  
  Just be mindful that adding multi-word items to your vocab can blow up its size. You might need a cutoff frequency to keep only the most frequent multi-word expressions.

### 2. Homograph Conflation (Word-Sense Issues)

**Limitation**  
A single token can have multiple unrelated meanings (“bank” → financial institution vs. riverbank). With this approach, they may collapse into one embedding, losing crucial sense distinctions.

**Recommendations**

- **Context-Dependent Token Encoding**  
  Instead of generating a token embedding from just `"passage: bank"`, feed the token within a minimal context snippet from the original text. For example, “passage: He deposited money at the bank” versus “passage: The tree fell by the river bank.”  
  This can be done by:
    1. Extending your dataset to store short contexts for each token (up to ±5–10 words around it).
    2. During training, project that context’s [CLS] hidden state through your MLP.
    3. The same surface form “bank” in different contexts will then produce different contextual embeddings, helping disambiguate senses.

- **Multi-Sense Embeddings**  
  If you truly need distinct embeddings for each sense of a word, consider multi-sense approaches, such as training multiple heads or learning a sense inventory. Common strategies include:
    - **Cluster-based sense discovery**  
      Run your model on many occurrences of each token in context, cluster their embeddings, and store each cluster centroid as a different sense.
    - **Parameteric sense disambiguation**  
      Have a small classification module predict which sense is active based on context, and then select or blend the corresponding sense embedding.

- **Fine-Tune the Backbone**  
  Simply freezing the base model forces all contextual disambiguation to happen in the final MLP. For better sense-separation, unfreeze at least the top few encoder layers. This lets them adjust to generating more discriminative hidden states for ambiguous tokens.

---

## Additional Considerations

1. **Vocabulary Building**
    - Your current approach filters tokens by frequency. You should verify that truly ambiguous or domain-specific terms are still included. It might be worth separately tracking rare but important tokens or expressions.

2. **Regularization & Orthogonality Penalty**
    - You already use an orthogonality penalty (`(W @ W.T).pow(2).triu(1).mean()`). That can keep your projection layer’s weight vectors from collapsing. This is a nice touch, but for multi-sense or multi-word expansions, you might want dynamic approaches that handle synonyms vs. homographs differently, possibly with flexible weighting rather than a strict orthogonality penalty.

3. **Downstream Integration**
    - Because you plan to build a second model that incrementally reconstructs text via token embeddings, it’s important that your embedding space remain consistent across words and phrases. Ensuring robust coverage of multi-word units and sense distinctions will reduce ambiguities in reconstruction.

---

## Summary

- **Multi-Word Expressions:** Introduce n-gram sampling and/or dedicated phrase vocab entries so that the projection head learns distinct embeddings for frequent multi-word units.
- **Homograph Conflation:** Incorporate context windows around each token in your training data, or adopt multi-sense embedding strategies. Unfreezing (at least partially) the base encoder layers can also markedly improve sense separation.

Implementing these suggestions will help make your token embeddings more context-aware and reduce confusion around multi-word phrases and homographs.

---

# Response 2

**Short answer:** Yes, focusing on the “hardest negative” for each token can accidentally push closely related words apart if they happen to be chosen as each other’s negatives. One way to mitigate this is to introduce additional training signals—such as grouping known synonyms or hyponyms together as partial positives or softer negatives. More generally, you can add “semantic constraints” to the contrastive objective that ensure tokens which are semantically related remain closer in the embedding space than truly unrelated words.

---

### Why hard negatives might cause issues
By always choosing the “hardest negative” (the token whose embedding is most similar), your loss function aggressively pushes apart words that happen to share contextual semantics—often synonyms or near-synonyms. Over many iterations, this can distort the geometry of your embedding space so that tokens which genuinely belong together end up too far apart.

### Techniques to preserve similarity
1. **Use multiple negatives**  
   Instead of using a single hardest negative, consider sampling several negative candidates (including “medium” negatives) so you are not always pushing away tokens that are naturally close.

2. **Incorporate softer positive signals**  
   If you have lexical resources or a method to identify synonyms, treat them as “partial positives” (less than anchor–itself, but more than random words). Others sometimes call this “label smoothing” for synonyms.

3. **Adopt a margin-based or circle loss**  
   With margin-based methods, you can specify a distance threshold that negative examples must stay beyond. That way, semantically close words still have some leeway to remain near each other.

4. **Manage the temperature hyperparameter**  
   If the temperature (τ) is too low, the model sharply penalizes subtle differences. A slightly higher τ can reduce unwanted repulsion of near synonyms.

5. **Data-driven synonyms or sub-word expansions**  
   Token-level near-duplicates (like singular/plural variants) might be handled by grouping them into a single representation or by training them with partial positives. This retains local structure and prevents over-separation.

By integrating one or more of these strategies, you can preserve the natural clustering of related words while still nudging genuinely dissimilar words further apart.