I want to train an encoder using a vocabulary of characters and use MLM (masking characters), CLM (predicting next character), and a reconstruction objective where the model is required to reconstruct the original input exactly from the embeddings.

Here is my code so far:
```
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
                 feedforward_dim: int = 256,
                 max_mem_slots: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_mem_slots = max_mem_slots

        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Non-causal block (for MLM, reconstruction)
        self.encoder_block = SimpleTransformerBlock(embed_dim, num_heads, feedforward_dim)

        # Causal block (for CLM)
        self.decoder_block = SimpleCausalTransformerBlock(embed_dim, num_heads, feedforward_dim)

        # Output heads
        self.mlm_head = nn.Linear(embed_dim, vocab_size)
        self.clm_head = nn.Linear(embed_dim, vocab_size)
        self.reconstruction_head = nn.Linear(embed_dim, vocab_size)

        # Cheap projection so a memory vector looks like a "token"
        self.mem_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                input_ids:       torch.Tensor,  # (B, T)
                memory_embs:     torch.Tensor | None):  # (B, K, E)  or  None
        """
        Returns a dict of logits.  If memory_embs is given, K≤max_mem_slots
        vectors are prepended to the encoder stream and *ignored* for loss
        computation (they carry no labels).
        """

        # Non-causal stream
        inp_emb = self.embedding(input_ids)  # (B, T, E)

        if memory_embs is not None:
            K = memory_embs.size(1)
            mem_tok = self.mem_proj(memory_embs)  # (B, K, E)
            enc_inp = torch.cat([mem_tok, inp_emb], dim=1)  # (B, K+T, E)
        else:
            K = 0
            enc_inp = inp_emb

        enc_out = self.encoder_block(enc_inp)  # (B, T, E)
        enc_out_curr = enc_out[:, K:, :]  # drop mem vectors

        # Causal stream
        causal_emb = self.embedding(input_ids)  # (B, T, E)
        causal_hidden = self.decoder_block(causal_emb) # (B, T, E)

        # Compute logits
        mlm_logits = self.mlm_head(enc_out_curr)
        clm_logits = self.clm_head(causal_hidden)
        reconstruction_logits = self.reconstruction_head(enc_out_curr)

        return {
            "mlm_logits": mlm_logits,
            "clm_logits": clm_logits,
            "reconstruction_logits": reconstruction_logits,
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
```

Use o1 to review the current implementation. Think thoroughly about what I'm trying to accomplish and where my code might fall short. Suggest ways to improve. Take as many steps in your chain of thought as you can.

---

# Response 1

**Review of the Current Implementation**

You have a dual‐stream Transformer designed to handle three objectives at once:
1. **MLM (masked language modeling)**: Randomly mask characters and predict which character should be in each masked location.
2. **CLM (causal language modeling)**: Predict the next character in the sequence.
3. **Reconstruction**: Reconstruct the original token sequence exactly from the encoder embeddings.

Below are suggestions that can help you move closer to your overall goal and address potential gaps:

---

### 1. Positional Information
Right now, the code has no positional encodings or embeddings in either the encoder or decoder. Transformers typically rely on positional information to understand ordering. You can fix this by adding something like:
```python
self.pos_embedding = nn.Embedding(max_seq_length, embed_dim)
```
and then, in the forward pass, adding it to `inp_emb` or `causal_emb` (e.g., by summing a positional embedding indexed by token position).

---

### 2. Handling of CLM Input vs. Label Shift
For causal language modeling, you usually feed tokens `[x1, x2, ..., x(T-1)]` and predict `[x2, ..., xT]`. In your code, you set:
```python
clm_labels = tokens[1:] + [self.pad_id]
```
but you also feed the entire `input_ids` into the decoder block. That often means the model is “seeing” token `xT` when it tries to predict `xT`. You might want to shift your input to the decoder so it does not see the next token in its own input:
1. Let the decoder input be `[x1, x2, ..., x(T-1)]`.
2. Let the target (label) be `[x2, x3, ..., xT]`.

This prevents “cheating” on the next‐character prediction.

---

### 3. Balancing the Three Losses
You are summing the MLM, CLM, and reconstruction losses equally. The reconstruction loss can easily dominate if most tokens are unmasked. Consider adding weights to balance them or schedule different tasks at different stages of training. For example:
```python
total_loss = alpha * mlm_loss + beta * clm_loss + gamma * recon_loss
```
where α, β, and γ are chosen based on which objective you want to emphasize more.

---

### 4. Reconstruction Loss Indexing
You currently ignore index 0 (the padding ID) in the reconstruction loss:
```python
recon_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
```
Check that your dataset never uses 0 for valid tokens, otherwise you may unintentionally ignore some real tokens in the loss. If the only time `0` appears is in padding positions, then this is correct. Otherwise, consider a different scheme (e.g. always pad with a separate padding ID and ignore that).

---

### 5. Memory Embeddings
Your model includes a block for prepending memory embeddings, but you’re always passing `memory_embs=None`. If you intend to use external memory or some extra states (like cross‐chunk context), you’ll need to feed them in and ensure they do not contribute to the MLM/CLM/reconstruction losses (as you have started to do by slicing out `enc_out[:, K:, :]`). Just be sure they are actually used in practice if that’s part of your design.

---

### 6. Model Depth
Each of your Transformer blocks (SimpleTransformerBlock and SimpleCausalTransformerBlock) is only a single layer. The model might learn more expressive representations with multiple layers—e.g. stacking several blocks:
```python
self.encoder_layers = nn.ModuleList([
    SimpleTransformerBlock(embed_dim, num_heads, feedforward_dim) for _ in range(num_encoder_layers)
])
self.decoder_layers = nn.ModuleList([
    SimpleCausalTransformerBlock(embed_dim, num_heads, feedforward_dim) for _ in range(num_decoder_layers)
])
```
Then you’d pass the embeddings through each block in turn.

---

### 7. Weight Tying
Often in language models, the same weight matrix is used for both an input embedding and an output projection. This can improve performance and reduce parameters. If that’s desired:
```python
# Tie the embedding weights and the output head for MLM, for example:
self.mlm_head.weight = self.embedding.weight
```
Repeat similarly for CLM or reconstruction if appropriate.

---

### 8. Sampling & Masking Details
Your MLM routine is workable, but you might consider:
- Adjusting mask probabilities.
- More nuanced handling of the different character types.

Ensure that special tokens (like `[MASK]`, `[PAD]`, etc.) are consistently excluded from random replacement.

---

### 9. Practical Considerations
You will likely also want to:
- Add gradient clipping if exploding gradients are an issue.
- Use a learning rate scheduler (like a warmup + decay).
- Track separate metrics (like exact match on reconstruction vs. perplexity on MLM/CLM) to see how each objective is doing.
- Validate on a held‐out set to make sure you aren’t overfitting.

---

**Overall, the design of a dual‐stream approach for MLM + CLM + reconstruction is promising.** With the above refinements—positional embeddings, rechecking the CLM input‐label shift, weighting the losses, ensuring the reconstruction loss only focuses on valid tokens, and potentially deepening the Transformer or tying weights—your code will be better aligned with standard Transformer best practices and more likely to succeed at the three objectives you have in mind.
