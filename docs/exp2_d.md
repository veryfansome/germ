Here I I want to train an encoder using a vocabulary of characters and use MLM (masking characters), CLM (predicting next character), and a reconstruction loss where the model is required to reconstruct the original input exactly from the embeddings.

Here is my code so far:
```
import contextlib
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

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


def mask_characters(tokens, mask_token_id, space_id, vocab_size, mask_prob=0.35, space_bias=0.05):
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
            tokens, self.mask_token_id, self.space_id, self.vocab_size, mask_prob=self.mask_prob,
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
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.2):
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
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.2):
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
                 embed_dim: int = 256,
                 num_heads: int = 4,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 feedforward_dim: int = 512,
                 max_mem_slots: int = 4,
                 max_seq_len: int = 256,
                 dropout: float = 0.2):
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
        self.mlm_head.weight = self.token_embedding.weight
        self.clm_head = nn.Linear(embed_dim, vocab_size)
        self.reconstruction_head = nn.Linear(embed_dim, vocab_size)
        self.reconstruction_head.weight = self.token_embedding.weight

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
        assert not torch.isnan(enc_out_curr).any(), "NaNs in encoder output"

        # Causal stream
        dec_inp_emb = inp_emb
        for blk in self.decoder_layers:
            dec_inp_emb = blk(dec_inp_emb, enc_out_curr) # (B, T, E)
            assert dec_inp_emb.shape == inp_emb.shape

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


def run_epoch(
        model, dataloader, device,
        optimizer=None, scheduler=None,
        virtual_batch_size: int = 256,
):
    model.train() if optimizer is not None else model.eval()

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

    no_grad_context = contextlib.nullcontext() if optimizer is not None else torch.inference_mode()
    grad_clip = 1.0
    accum_steps = virtual_batch_size // dataloader.batch_size
    with no_grad_context:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            clm_labels = batch["clm_labels"].to(device)
            original_tokens = batch["original_tokens"].to(device)

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
            alpha, beta, gamma = 1.0, 1.0, 0.2
            total_loss = (alpha * mlm_loss +
                          beta  * clm_loss +
                          gamma * recon_loss)
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
                    f"Recon: {recon_loss.item():.4f}"
                )

            # Accumulate
            mlm_loss_accum += mlm_loss.item()
            clm_loss_accum += clm_loss.item()
            recon_loss_accum += recon_loss.item()
            total_loss_accum += total_loss.item()

    # Return the average of each loss
    return {
        "mlm_loss": mlm_loss_accum / steps,
        "clm_loss": clm_loss_accum / steps,
        "recon_loss": recon_loss_accum / steps,
        "total_loss": total_loss_accum / steps,
    }


if __name__ == "__main__":
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from torch.utils.data import DataLoader

    from germ.sandbox.char_encoder_data import build_vocab, build_wikitext_corpus

    wikitext_train = build_wikitext_corpus(max_len=256, split="train")
    wikitext_test = build_wikitext_corpus(max_len=256, split="test")

    _char2idx, _idx2char = build_vocab(wikitext_train + wikitext_test)
    _vocab_size = len(_idx2char)
    print(f"Vocab size: {_vocab_size}, vocab: {_char2idx.keys()}")

    # 2. Create dataset and dataloader
    ds = MultiTaskTextDataset(
        wikitext_train, _char2idx, _idx2char, mask_token="[MASK]", mask_prob=0.2, max_len=256
    )
    dl = DataLoader(
        ds, batch_size=64, shuffle=True, collate_fn=collate_fn
    )

    # 3. Initialize model and optimizer
    _device = torch.device("mps" if torch.mps.is_available() else "cpu")
    _model = DualStreamMultiTaskModel(
        vocab_size=_vocab_size,
        embed_dim=256,
        num_heads=8,
        feedforward_dim=1024,
        max_seq_len=256,
        max_mem_slots=4
    ).to(_device)

    _optimizer = optim.Adam(_model.parameters(), lr=6e-4, betas=(0.9, 0.98), weight_decay=1e-2)
    _warmup = LinearLR(_optimizer, start_factor=0.1, total_iters=2000)
    _cosine = CosineAnnealingLR(_optimizer, T_max=48000, eta_min=1e-5)
    _scheduler = SequentialLR(_optimizer, schedulers=[_warmup, _cosine], milestones=[2000])

    # 4. Train
    epochs = 1
    print("Short Text Dataset:")
    for epoch in range(epochs):
        losses_dict = run_epoch(_model, dl, _device, optimizer=_optimizer, scheduler=_scheduler)
        print(
            "  "
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {losses_dict['total_loss']:.4f}, "
            f"MLM: {losses_dict['mlm_loss']:.4f}, "
            f"CLM: {losses_dict['clm_loss']:.4f}, "
            f"Recon: {losses_dict['recon_loss']:.4f}"
        )
```

Here are the functions I'm using for building a corpus to test this code on:
```
def build_vocab(texts, min_percentile=1):
    """
    Build a character vocabulary from a list of strings, excluding characters in the lowest frequency percentile.
    Returns:
      char2idx (dict)
      idx2char (list)
    """
    # Count the frequency of each character
    counter = Counter(ch for t in texts for ch in t)
    # Calculate frequency percentiles
    frequencies = np.array(list(counter.values()))
    thresholds = np.percentile(frequencies, min_percentile)
    # Filter out characters below the given frequency percentile
    vocab = [ch for ch, freq in counter.items() if freq > thresholds]
    # Add special tokens for mask, start-of-sequence, end-of-sequence if needed
    special_tokens = ["[PAD]", "[MASK]", "[UNK]"]
    vocab = sorted(vocab)
    idx2char = special_tokens + vocab
    char2idx = {ch: i for i, ch in enumerate(idx2char)}
    return char2idx, idx2char


def build_wikitext_corpus(max_len: int = 256, split: str = "train"):
    """Extract a small corpus of clean short-form text from WikiText‑103‑raw‑v1 for encoder testing."""
    print(f"Loading wikitext dataset")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    ds_size = len(ds)
    print(f"Selecting {split} corpus from {ds_size} wikitext documents")
    candidates = []
    startswith_uppercase_pattern = re.compile(r"[A-Z]")
    for row_id, row in enumerate(ds):
        row_text = row["text"].strip()
        if (
                # Filters to reduce noise.
                not row_text or not row_text.isascii()  # Exclude non-english characters
                or not bool(startswith_uppercase_pattern.match(row_text))  # Must start with uppercase
                or row_text[-1] not in {".", "?", "!"}  # Must have standard punctuation
                or len(row_text) < 7  # At least 7 words
        ):
            continue
        if len(row_text) <= max_len:
            candidates.append(row_text)
        if row_id > 0 and row_id % 250000 == 0:
            print(f"Processed {row_id} rows")
    print(f"Selected {len(candidates)} {split} candidates from wikitext")
    for c in random.sample(candidates, 10):
        print(f" => {c}")
    return candidates
```

Here is the initial output from the first epoch of training:
```
Vocab size: 97, vocab: dict_keys(['[PAD]', '[MASK]', '[UNK]', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~'])
Short Text Dataset:
  Step 50, Loss: 20.5741, MLM: 15.7121, CLM: 4.3169, Recon: 2.7255
  Step 100, Loss: 11.7989, MLM: 7.6145, CLM: 3.8497, Recon: 1.6737
  Step 150, Loss: 8.3814, MLM: 4.6039, CLM: 3.5316, Recon: 1.2295
  Step 200, Loss: 7.4729, MLM: 3.8798, CLM: 3.3949, Recon: 0.9909
  Step 250, Loss: 7.0884, MLM: 3.5758, CLM: 3.3315, Recon: 0.9058
  Step 300, Loss: 6.7943, MLM: 3.3597, CLM: 3.2787, Recon: 0.7795
  Step 350, Loss: 6.7020, MLM: 3.3269, CLM: 3.2313, Recon: 0.7187
  Step 400, Loss: 6.6429, MLM: 3.3038, CLM: 3.1949, Recon: 0.7211
  Step 450, Loss: 6.4998, MLM: 3.1953, CLM: 3.1604, Recon: 0.7207
  Step 500, Loss: 6.5353, MLM: 3.2366, CLM: 3.1598, Recon: 0.6942
  Step 550, Loss: 6.5735, MLM: 3.2805, CLM: 3.1527, Recon: 0.7017
  Step 600, Loss: 6.4714, MLM: 3.2365, CLM: 3.1005, Recon: 0.6719
  Step 650, Loss: 6.4766, MLM: 3.2320, CLM: 3.1072, Recon: 0.6872
  Step 700, Loss: 6.5593, MLM: 3.2954, CLM: 3.1244, Recon: 0.6972
  Step 750, Loss: 6.6235, MLM: 3.3804, CLM: 3.1042, Recon: 0.6946
```

Seems like improvement has plateaued. What adjustments would you recommend?