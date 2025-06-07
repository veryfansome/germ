import contextlib
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

################################################################################
# Utility functions
################################################################################


def collate_fn(batch, pad_id: int = 0):
    """
    Pad sequences in the batch to the same length.
    """
    def pad(seq_list, pad_value):
        max_len = max(len(s) for s in seq_list)
        return torch.stack([
            torch.cat([s, torch.full((max_len - len(s),), pad_value, dtype=torch.long)])
            for s in seq_list
        ])

    return {
        "enc_input_ids":   pad([b["enc_input_ids"]   for b in batch], pad_id),
        "dec_input_ids":   pad([b["dec_input_ids"]   for b in batch], pad_id),
        "mlm_labels":      pad([b["mlm_labels"]      for b in batch], -100),
        "clm_labels":      pad([b["clm_labels"]      for b in batch], -100),
        "original_tokens": pad([b["original_tokens"] for b in batch], pad_id),
    }


def detokenize(tokens):
    """
    Convert a list of characters back to a string.
    """
    return "".join(tokens)


def generate_causal_mask(seq_len, device):
    """
    Builds a [seq_len x seq_len] mask that sets positions beyond the current token
    to -inf to prevent the model from attending to future tokens.
    """
    # Triangular matrix: 1's above diagonal
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    # Convert 1 to -inf so that attention softmax ignores these positions
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def mask_characters(tokens, mask_token_id, space_id, vocab_size, mask_prob=0.15, space_bias=0.05):
    """
    Randomly mask out characters (MLM).
    Returns:
      masked_tokens (list of ints)
      mlm_labels (list of ints), where -100 means 'no prediction needed'
    """
    masked_tokens = []
    mlm_labels = []
    for token in tokens:
        if token == 0:  # 0 == [PAD], don't mask padding
            masked_tokens.append(token)
            mlm_labels.append(-100)
            continue
        p = space_bias if token == space_id else mask_prob
        if random.random() < p:
            mlm_labels.append(token)  # Add a real label because we will force the model to predict it
            r = random.random()
            if r < 0.8:
                masked_tokens.append(mask_token_id)  # replace by [MASK]
            elif r < 0.9:
                masked_tokens.append(random.randint(3, vocab_size - 1))  # random char
            else:
                masked_tokens.append(token)  # keep original
        else:
            masked_tokens.append(token)
            mlm_labels.append(-100)  # ignore in loss
    return masked_tokens, mlm_labels


def tokenize(text):
    """
    Character-level tokenizer: returns a list of characters.
    """
    return list(text)


################################################################################
# Dataset
################################################################################


class MultiTaskTextDataset(Dataset):
    """
    A toy dataset to demonstrate multi-task pretraining.
    Each item yields:
      - input_ids: token IDs after applying random MLM
      - mlm_labels: labels for MLM (or -100 if not masked)
      - clm_labels: next character
    """
    def __init__(self, texts, char2idx, idx2char, mask_token="[MASK]", mask_prob=0.15, max_len=50):
        super().__init__()
        self.texts = texts
        self.char2idx = char2idx
        self.idx2char = idx2char

        self.mask_prob = mask_prob
        self.mask_token_id = char2idx[mask_token] if mask_token in char2idx else 1
        self.max_len = max_len
        self.pad_id = char2idx["[PAD]"]
        self.space_id = self.char2idx[" "] if " " in self.char2idx else -1
        self.vocab_size = len(char2idx)

        self.samples = []
        self._build_samples()

    def _build_samples(self):
        for text in self.texts:
            tokens = [self.char2idx[ch] if ch in self.char2idx else self.char2idx["[UNK]"]
                      for ch in tokenize(text)]
            if len(tokens) > self.max_len:
                tokens = tokens[: self.max_len]
            self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]

        # Prepare labels for next-character LM (CLM)
        dec_input_ids = tokens
        clm_labels = tokens[1:] + [self.pad_id]

        # Prepare MLM
        masked_tokens, mlm_labels = mask_characters(
            tokens, self.mask_token_id, self.space_id, self.vocab_size, mask_prob=self.mask_prob,
        )

        return {
            "enc_input_ids": torch.tensor(masked_tokens, dtype=torch.long),
            "dec_input_ids": torch.tensor(dec_input_ids, dtype=torch.long),
            "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long),
            "clm_labels": torch.tensor(clm_labels, dtype=torch.long),
            "original_tokens": torch.tensor(tokens, dtype=torch.long),
        }


################################################################################
# Model Definition
################################################################################


class SimpleCausalTransformerBlock(nn.Module):
    """
    A single Transformer block with causal (autoregressive) masking.
    """
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_mem):
        # x is shape (B, T, E), so we need an attn_mask of size (T, T)
        seq_len = x.size(1)
        device = x.device
        causal_mask = generate_causal_mask(seq_len, device)

        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_out)

        cross_out, _ = self.cross_attn(x, enc_mem, enc_mem)
        x = self.norm2(x + cross_out)

        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return x


class SimpleTransformerBlock(nn.Module):
    """
    A single Transformer block (Encoder-only, bidirectional attention).
    """
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class DualStreamMultiTaskModel(nn.Module):
    """
    A dual-stream model with a shared embedding, one non-causal encoder block for MLM, and one causal decoder
    block for CLM.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 256,
                 num_heads: int = 4,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 feedforward_dim: int = 512,
                 max_seq_len: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Shared embedding
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # Non-causal layers (for MLM)
        self.encoder_layers = nn.ModuleList([
            SimpleTransformerBlock(embed_dim, num_heads, feedforward_dim) for _ in range(num_encoder_layers)
        ])

        # Causal layers (for CLM)
        self.decoder_layers = nn.ModuleList([
            SimpleCausalTransformerBlock(embed_dim, num_heads, feedforward_dim) for _ in range(num_decoder_layers)
        ])

        # Output heads
        self.mlm_head = nn.Linear(embed_dim, vocab_size)
        self.mlm_head.weight = self.token_embedding.weight
        self.clm_head = nn.Linear(embed_dim, vocab_size)
        self.clm_head.weight = self.token_embedding.weight

    @staticmethod
    @torch.no_grad()
    def _arange_like(x, offset: int = 0):
        """
        Returns a [B, L] tensor [[offset, offset+1, …]] on x.device.
        """
        B, L = x.shape[:2]
        return (torch.arange(L, device=x.device) + offset).unsqueeze(0).expand(B, -1)

    def forward(self,
                enc_input_ids: torch.Tensor,  # (B, T)
                dec_input_ids: torch.Tensor):  # (B, K, E)  or  None
        """
        Returns a dict of logits.
        """
        # Non-causal stream
        enc_tok  = self.token_embedding(enc_input_ids)  # (B, T, E)
        enc_pos  = self.position_embedding(self._arange_like(enc_input_ids, offset=0))
        enc_inp  = self.embed_dropout(enc_tok + enc_pos)  # (B, T, E)

        for blk in self.encoder_layers:
            enc_inp = blk(enc_inp)  # (B, T, E)

        # Causal stream
        dec_tok  = self.token_embedding(dec_input_ids)
        dec_pos  = self.position_embedding(self._arange_like(dec_input_ids))
        dec_inp  = self.embed_dropout(dec_tok + dec_pos)
        for blk in self.decoder_layers:
            dec_inp = blk(dec_inp, enc_inp) # (B, T, E)

        # Compute logits
        return {
            "mlm_logits": self.mlm_head(enc_inp),
            "clm_logits": self.clm_head(dec_inp),
            # Expose pooled chunk embedding for caller
            "chunk_emb": enc_inp.mean(dim=1)  # (B, E)
        }


################################################################################
# Training Loop
################################################################################


def run_epoch(
        model, dataloader, device,
        optimizer=None, scheduler=None,
        virtual_batch_size: int = 512,
):
    model.train() if optimizer is not None else model.eval()

    # Task-specific losses
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    clm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Accumulators for each loss
    mlm_loss_accum = 0.0
    clm_loss_accum = 0.0
    total_loss_accum = 0.0
    steps = 0

    no_grad_context = contextlib.nullcontext() if optimizer is not None else torch.inference_mode()
    grad_clip = 1.0
    accum_steps = virtual_batch_size // dataloader.batch_size
    with no_grad_context:
        for batch in dataloader:
            input_ids = batch["enc_input_ids"].to(device)
            dec_input_ids = batch["dec_input_ids"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            clm_labels = batch["clm_labels"].to(device)
            original_tokens = batch["original_tokens"].to(device)

            outputs = model(input_ids, dec_input_ids)
            mlm_logits = outputs["mlm_logits"]
            clm_logits = outputs["clm_logits"]

            # Compute individual losses
            mlm_loss = mlm_loss_fn(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1)
            )
            clm_loss = clm_loss_fn(
                clm_logits.view(-1, clm_logits.size(-1)),
                clm_labels.view(-1)
            )

            # Combine losses (you can add weights here if desired)
            alpha, beta = 1.0, 1.0
            total_loss = (alpha * mlm_loss +
                          beta  * clm_loss)
            if optimizer is not None:
                (total_loss / accum_steps).backward()
                if steps % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

            steps += 1
            if steps % 50 == 0:
                print(
                    "  "
                    f"Step {steps}, "
                    f"Loss: {total_loss.item():.4f}, "
                    f"MLM: {mlm_loss.item():.4f}, "
                    f"CLM: {clm_loss.item():.4f}, "
                )

            # Accumulate
            mlm_loss_accum += mlm_loss.item()
            clm_loss_accum += clm_loss.item()
            total_loss_accum += total_loss.item()

    # Return the average of each loss
    return {
        "mlm_loss": mlm_loss_accum / steps,
        "clm_loss": clm_loss_accum / steps,
        "total_loss": total_loss_accum / steps,
    }


if __name__ == "__main__":
    import json
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from torch.utils.data import DataLoader

    from germ.sandbox.char_encoder_data import build_vocab, build_wikitext_word_corpus

    with open("data/e5_token_embedding_model/all_lowercase_token.json") as f:
        word_corpus = json.load(f)

    _char2idx, _idx2char = build_vocab(word_corpus)
    _vocab_size = len(_idx2char)
    print(f"Vocab size: {_vocab_size}, vocab: {_char2idx.keys()}")

    # 2. Create dataset and dataloader
    ds = MultiTaskTextDataset(
        word_corpus, _char2idx, _idx2char, mask_token="[MASK]", mask_prob=0.15, max_len=64
    )
    dl = DataLoader(
        ds, batch_size=64, shuffle=True, collate_fn=collate_fn
    )

    # 3. Initialize model and optimizer
    _device = torch.device("mps" if torch.mps.is_available() else "cpu")
    _model = DualStreamMultiTaskModel(
        vocab_size=_vocab_size,
        embed_dim=256,
        num_heads=4,
        feedforward_dim=512,
        max_seq_len=64,
    ).to(_device)

    _optimizer = optim.Adam(_model.parameters(), lr=6e-4, betas=(0.9, 0.98), weight_decay=1e-2)
    _warmup = LinearLR(_optimizer, start_factor=0.1, total_iters=250)
    _cosine = CosineAnnealingLR(_optimizer, T_max=48000, eta_min=1e-5)
    _scheduler = SequentialLR(_optimizer, schedulers=[_warmup, _cosine], milestones=[2000])

    # 4. Train
    epochs = 5
    print("Short Text Dataset:")
    for epoch in range(epochs):
        losses_dict = run_epoch(_model, dl, _device, optimizer=_optimizer, scheduler=_scheduler)
        print(
            "  "
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {losses_dict['total_loss']:.4f}, "
            f"MLM: {losses_dict['mlm_loss']:.4f}, "
            f"CLM: {losses_dict['clm_loss']:.4f}, "
        )