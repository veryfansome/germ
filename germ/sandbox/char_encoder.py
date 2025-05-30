import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from germ.sandbox.char_encoder_data import punctuation_chars

################################################################################
# Utility functions
################################################################################


def collate_fn(batch):
    """
    Pad sequences in the batch to the same length.
    """
    def pad_sequence(seq_list, pad_value):
        max_len = max(len(s) for s in seq_list)
        padded = []
        for s in seq_list:
            padded.append(
                torch.cat([s, torch.full((max_len - len(s),), pad_value, dtype=torch.long)])
            )
        return torch.stack(padded, dim=0)

    input_ids = pad_sequence([b["input_ids"] for b in batch], pad_value=0)
    mlm_labels = pad_sequence([b["mlm_labels"] for b in batch], pad_value=-100)
    clm_labels = pad_sequence([b["clm_labels"] for b in batch], pad_value=-100)
    original_tokens = pad_sequence([b["original_tokens"] for b in batch], pad_value=0)

    return {
        "input_ids": input_ids,
        "mlm_labels": mlm_labels,
        "clm_labels": clm_labels,
        "original_tokens": original_tokens,
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


def mask_characters(tokens, mask_token_id, vocab_size, mask_prob=0.15):
    """
    Randomly mask out characters (MLM).
    Returns:
      masked_tokens (list of ints)
      mlm_labels (list of ints), where -100 means 'no prediction needed'
    """
    masked_tokens = []
    mlm_labels = []
    for token in tokens:
        if token not in [0]:  # don't mask padding
            if random.random() < mask_prob:
                # 80% of the time replace with mask_token
                # 10% of the time replace with random
                # 10% keep the same
                r = random.random()
                mlm_labels.append(token)
                if r < 0.8:
                    masked_tokens.append(mask_token_id)
                elif r < 0.9:
                    # random token
                    masked_tokens.append(random.randint(3, vocab_size - 1))
                else:
                    masked_tokens.append(token)
            else:
                masked_tokens.append(token)
                mlm_labels.append(-100)  # ignore index
        else:
            masked_tokens.append(token)
            mlm_labels.append(-100)
    return masked_tokens, mlm_labels


def remove_special_chars(tokens, remove_ids, ignore_label=2):
    """
    Removes any token in 'remove_ids' from the sequence. For each token that
    REMAINS in the 'new_tokens', store the ID of the removed token if it
    immediately preceded it, or 'ignore_label' (defaults to 2) if nothing
    was removed.

    Example:
      tokens       = [10, 30, 5, 30, 6]
      remove_ids   = {30}
      new_tokens   = [10, 5, 6]
      insert_label = [2, 30, 2]   # we removed '30' before the 5, so label=30
                                  # otherwise label=2
    """
    new_tokens = []
    insertion_labels = []
    removed_char = None

    for t in tokens:
        if t in remove_ids:
            # Mark that we found a removed char
            removed_char = t
        else:
            # Keep this token and label with the removed char if any
            new_tokens.append(t)
            if removed_char is None:
                insertion_labels.append(ignore_label)
            else:
                insertion_labels.append(removed_char)
                removed_char = None

    return new_tokens, insertion_labels


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
      - original_tokens: original token IDs (for reconstruction loss)
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
        self.punctuation_ids = [
            self.char2idx[ch] for ch in punctuation_chars if ch in self.char2idx
        ]
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
        clm_labels = tokens[1:] + [self.pad_id]

        # Prepare MLM
        masked_tokens, mlm_labels = mask_characters(
            tokens, self.mask_token_id, self.vocab_size, mask_prob=self.mask_prob
        )

        return {
            "input_ids": torch.tensor(masked_tokens, dtype=torch.long),
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
        # x is shape (B, T, E), so we need an attn_mask of size (T, T)
        seq_len = x.size(1)
        device = x.device
        causal_mask = generate_causal_mask(seq_len, device)

        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
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
    A dual-stream model with a shared embedding, one non-causal encoder block for
    MLM/reconstruction, and one causal decoder block for CLM.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 feedforward_dim: int = 256,
                 max_mem_slots: int = 4,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_mem_slots = max_mem_slots
        self.max_seq_len = max_seq_len

        # Shared embedding
        self.position_embedding = nn.Embedding(max_seq_len + max_mem_slots, embed_dim)  # TODO: why add mem slots?
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # Non-causal layers (for MLM, reconstruction)
        self.encoder_layers = nn.ModuleList([
            SimpleTransformerBlock(embed_dim, num_heads, feedforward_dim) for _ in range(num_encoder_layers)
        ])

        # Causal layers (for CLM)
        self.decoder_layers = nn.ModuleList([
            SimpleCausalTransformerBlock(embed_dim, num_heads, feedforward_dim) for _ in range(num_decoder_layers)
        ])

        # Output heads
        self.mlm_head = nn.Linear(embed_dim, vocab_size)
        self.clm_head = nn.Linear(embed_dim, vocab_size)
        self.reconstruction_head = nn.Linear(embed_dim, vocab_size)

        # Cheap projection so a memory vector looks like a "token"
        self.mem_proj = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    @torch.no_grad()
    def _arange_like(x, offset: int = 0):
        """
        Returns a [B, L] tensor [[offset, offset+1, …]] on x.device.
        """
        B, L = x.shape[:2]
        return (torch.arange(L, device=x.device) + offset).unsqueeze(0).expand(B, -1)

    def forward(self,
                input_ids:       torch.Tensor,  # (B, T)
                memory_embs:     torch.Tensor | None):  # (B, K, E)  or  None
        """
        Returns a dict of logits.  If memory_embs is given, K≤max_mem_slots
        vectors are prepended to the encoder stream and *ignored* for loss
        computation (they carry no labels).
        """
        K = 0 if memory_embs is None else memory_embs.size(1)

        # Non-causal stream
        tok_emb  = self.token_embedding(input_ids)  # (B, T, E)
        pos_emb  = self.position_embedding(self._arange_like(input_ids, offset=K))
        inp_emb  = self.embed_dropout(tok_emb + pos_emb)  # (B, T, E)

        if memory_embs is not None:
            mem_tok = self.mem_proj(memory_embs)  # (B, K, E)
            mem_pos   = self.position_embedding(self._arange_like(memory_embs[:, :, 0], offset=0))
            mem_emb = self.embed_dropout(mem_tok + mem_pos)  # (B, K, E)
            enc_inp = torch.cat([mem_emb, inp_emb], dim=1)  # (B, K+T, E)
        else:
            K = 0
            enc_inp = inp_emb

        for blk in self.encoder_layers:
            enc_inp = blk(enc_inp)  # (B, T, E)
        enc_out_curr = enc_inp[:, K:, :]  # drop mem vectors

        # Causal stream
        dec_inp_emb = inp_emb
        for blk in self.decoder_layers:
            dec_inp_emb = blk(dec_inp_emb) # (B, T, E)

        # Compute logits
        return {
            "mlm_logits": self.mlm_head(enc_out_curr),
            "clm_logits": self.clm_head(dec_inp_emb),
            "reconstruction_logits": self.reconstruction_head(enc_out_curr),
            # Expose pooled chunk embedding for caller
            "chunk_emb": enc_out_curr.mean(dim=1)  # (B, E)
        }


################################################################################
# Training Loop
################################################################################


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    # Task-specific losses
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    clm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    recon_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignoring padded positions (0)

    # Accumulators for each loss
    mlm_loss_accum = 0.0
    clm_loss_accum = 0.0
    recon_loss_accum = 0.0
    total_loss_accum = 0.0
    steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        mlm_labels = batch["mlm_labels"].to(device)
        clm_labels = batch["clm_labels"].to(device)
        original_tokens = batch["original_tokens"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, None)
        mlm_logits = outputs["mlm_logits"]
        clm_logits = outputs["clm_logits"]
        reconstruction_logits = outputs["reconstruction_logits"]

        # Compute individual losses
        mlm_loss = mlm_loss_fn(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1)
        )
        clm_loss = clm_loss_fn(
            clm_logits.view(-1, clm_logits.size(-1)),
            clm_labels.view(-1)
        )
        recon_loss = recon_loss_fn(
            reconstruction_logits.view(-1, reconstruction_logits.size(-1)),
            original_tokens.view(-1)
        )

        # Combine losses (you can add weights here if desired)
        total_loss = mlm_loss + clm_loss + recon_loss
        total_loss.backward()
        optimizer.step()

        # Accumulate
        mlm_loss_accum += mlm_loss.item()
        clm_loss_accum += clm_loss.item()
        recon_loss_accum += recon_loss.item()
        total_loss_accum += total_loss.item()
        steps += 1

    # Return the average of each loss
    return {
        "mlm_loss": mlm_loss_accum / steps,
        "clm_loss": clm_loss_accum / steps,
        "recon_loss": recon_loss_accum / steps,
        "total_loss": total_loss_accum / steps,
    }
