import collections
import json
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
                 base_model_name: str = "intfloat/e5-base-v2",
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

        # Single linear head
        #self.proj = nn.Linear(self.encoder.config.hidden_size, embed_dim, bias=False).to(self.device)
        #nn.init.normal_(self.proj.weight, std=0.02)

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

        # Best for single linear head
        #epochs: int = 3,
        #batch_size: int = 256,
        #lr: float = 1e-4,

        #epochs: int = 3,
        #epochs: int = 4,
        #epochs: int = 5,
        #epochs: int = 6,
        #epochs: int = 16,
        #epochs: int = 24,
        #epochs: int = 32,
        epochs: int = 40,
        #batch_size: int = 128,
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

        # Best for single linear head
        #200,

        #1000,
        1500,

        epochs * len(dl)
    )

    # Best for single linear hear
    #temperature = 0.07

    #temperature = 0.07
    temperature = 0.1

    accum_steps = 4  # ← batch_size 256×4 = 1024 token pairs per “virtual batch”

    model.train()
    optimiser.zero_grad(set_to_none=True)
    for ep in range(epochs):
        running_loss = 0.0
        for step, batch_tokens in enumerate(dl):
            embeds = model(batch_tokens)  # (B,d)
            logits = embeds @ embeds.T / temperature  # contrastive
            labels = torch.arange(len(batch_tokens), device=device_t)
            loss = F.cross_entropy(logits, labels) / accum_steps   # scale!
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
