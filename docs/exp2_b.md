I want to train an encoder using a vocabulary of characters and use MLM (masking characters), CLM (predicting next character), and denoising (removing all white spaces and having the model add them back). I also want to add a reconstruction loss where the model is required to reconstruct the original input exactly from the embeddings.

Here is my code so far:
```
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

################################################################################
# Utility functions
################################################################################

def build_vocab(texts):
    """
    Build a character vocabulary from a list of strings.
    Returns:
      char2idx (dict)
      idx2char (list)
    """
    vocab = set()
    for t in texts:
        for ch in t:
            vocab.add(ch)
    # Add special tokens for mask, start-of-sequence, end-of-sequence if needed
    # We reserve index 0 for padding
    special_tokens = ["[PAD]", "[MASK]", "[UNK]"]
    vocab = sorted(list(vocab))
    idx2char = special_tokens + vocab
    char2idx = {ch: i for i, ch in enumerate(idx2char)}
    return char2idx, idx2char

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
    denoise_input = pad_sequence([b["denoise_input"] for b in batch], pad_value=0)
    denoise_labels = pad_sequence([b["denoise_labels"] for b in batch], pad_value=2)
    original_tokens = pad_sequence([b["original_tokens"] for b in batch], pad_value=0)

    return {
        "input_ids": input_ids,
        "mlm_labels": mlm_labels,
        "clm_labels": clm_labels,
        "denoise_input": denoise_input,
        "denoise_labels": denoise_labels,
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

def remove_whitespaces(tokens, space_id):
    """
    Removes all spaces from the token sequence for the denoising task.
    Returns:
      no_space_tokens (list of ints): tokens with spaces removed
      space_positions (list of 0/1): 1 if a space should be inserted before that character
    """
    no_space_tokens = []
    space_positions = []
    was_space = False

    for t in tokens:
        if t == space_id:
            was_space = True
        else:
            no_space_tokens.append(t)
            # Append 1 if the current character was preceded by a space, else 0
            space_positions.append(1 if was_space else 0)
            was_space = False

    return no_space_tokens, space_positions

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
      - denoise_input: input IDs with spaces removed for denoising
      - denoise_labels: 0/1 for where space should be inserted
      - original_tokens: original token IDs (for reconstruction loss)
    """
    def __init__(self, texts, char2idx, idx2char, mask_token="[MASK]", mask_prob=0.15, max_len=50):
        super().__init__()
        self.texts = texts
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.vocab_size = len(char2idx)
        self.mask_prob = mask_prob
        self.mask_token_id = char2idx[mask_token] if mask_token in char2idx else 1
        self.pad_id = char2idx["[PAD]"]
        self.space_id = self.char2idx[" "] if " " in self.char2idx else -1
        self.max_len = max_len
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

        # Prepare denoising input (whitespace removal)
        if self.space_id != -1:
            denoise_input, space_positions = remove_whitespaces(tokens, self.space_id)
        else:
            # If there's no space in vocab, skip
            denoise_input = tokens
            space_positions = [0] * len(tokens)

        return {
            "input_ids": torch.tensor(masked_tokens, dtype=torch.long),
            "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long),
            "clm_labels": torch.tensor(clm_labels, dtype=torch.long),
            "denoise_input": torch.tensor(denoise_input, dtype=torch.long),
            "denoise_labels": torch.tensor(space_positions, dtype=torch.long),
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
    MLM/denoising/reconstruction, and one causal decoder block for CLM.
    """
    #def __init__(self, vocab_size, embed_dim=128, num_heads=4, feedforward_dim=256):
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

        # Non-causal block (for MLM, denoising, reconstruction)
        self.encoder_block = SimpleTransformerBlock(embed_dim, num_heads, feedforward_dim)

        # Causal block (for CLM)
        self.decoder_block = SimpleCausalTransformerBlock(embed_dim, num_heads, feedforward_dim)

        # Output heads
        self.mlm_head = nn.Linear(embed_dim, vocab_size)
        self.clm_head = nn.Linear(embed_dim, vocab_size)
        self.denoise_head = nn.Linear(embed_dim, 2)
        self.reconstruction_head = nn.Linear(embed_dim, vocab_size)

        # Cheap projection so a memory vector looks like a "token"
        self.mem_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                input_ids:       torch.Tensor,  # (B, T)
                denoise_input:   torch.Tensor,  # (B, T_denoise)
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

        # Denoising stream (use non-causal block)
        denoise_emb = self.embedding(denoise_input)            # (B, T_denoise, E)
        denoise_hidden = self.encoder_block(denoise_emb)       # (B, T_denoise, E)

        # Causal stream
        causal_emb = self.embedding(input_ids)  # (B, T, E)
        causal_hidden = self.decoder_block(causal_emb) # (B, T, E)

        # Compute logits
        mlm_logits = self.mlm_head(enc_out_curr)
        clm_logits = self.clm_head(causal_hidden)
        denoise_logits = self.denoise_head(denoise_hidden)
        reconstruction_logits = self.reconstruction_head(enc_out_curr)

        return {
            "mlm_logits": mlm_logits,
            "clm_logits": clm_logits,
            "denoise_logits": denoise_logits,
            "reconstruction_logits": reconstruction_logits,
            # Expose pooled chunk embedding for caller
            "chunk_emb": enc_out_curr.mean(dim=1)  # (B, E)
        }

class MemorySelector(nn.Module):
    """
    Given a current query embedding and a set of candidate memory embeddings,
    outputs log probabilities of selecting each candidate. We'll sample from
    this distribution or pick top-k.
    """
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # We'll produce a score per candidate

    def forward(self, query_embedding, memory_embeddings):
        """
        query_embedding: (embed_dim,)
        memory_embeddings: (M, embed_dim)  # M = number of memory chunks
        Returns:
          logits: (M,) raw scores for how relevant each memory chunk is
        """
        # Expand query to match memory embeddings
        M = memory_embeddings.size(0)
        query_expanded = query_embedding.unsqueeze(0).expand(M, -1)  # (M, embed_dim)

        # Concatenate query and candidate memory embeddings
        combined = torch.cat([query_expanded, memory_embeddings], dim=1)  # (M, embed_dim*2)

        x = torch.relu(self.fc1(combined))
        logits = self.fc2(x).squeeze(-1)  # shape (M,)
        return logits

################################################################################
# Training Loop
################################################################################

def choose_memory(memory_selector: MemorySelector,
                  query_embedding: torch.Tensor,        # (E,)
                  bank: list[torch.Tensor],             # list[(E,)]
                  k: int,
                  device: torch.device):
    """
    Sample *k* distinct indices without replacement from the memory bank
    according to the selector's logits.  Returns:
        chosen_embs   – (1, k, E)
        log_probs_sum – scalar   (sum of log probs of the k picks)
    """
    if len(bank) == 0 or k == 0:
        dummy = torch.zeros(1, 0, query_embedding.size(0), device=device)
        return dummy, torch.tensor(0., device=device)

    cand_embs = torch.stack(bank, dim=0).to(device)              # (M, E)
    logits = memory_selector(query_embedding.to(device), cand_embs)   # (M,)
    probs  = torch.softmax(logits, dim=0)

    # Multinomial w/out replacement – sample iteratively
    chosen_idx = []
    log_prob_terms = []
    tmp_probs = probs.clone()
    for _ in range(min(k, len(bank))):
        idx = torch.multinomial(tmp_probs, 1).item()
        log_prob_terms.append(torch.log(tmp_probs[idx] + 1e-10))
        chosen_idx.append(idx)
        tmp_probs[idx] = 0.0                              # remove so no repeat
        tmp_probs = tmp_probs / tmp_probs.sum()           # renormalise

    chosen_embs = cand_embs[chosen_idx].unsqueeze(0)      # (1, k, E)
    return chosen_embs, torch.stack(log_prob_terms).sum()

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    # Task-specific losses
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    clm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    denoise_loss_fn = nn.CrossEntropyLoss(ignore_index=2)
    recon_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignoring padded positions (0)

    # Accumulators for each loss
    mlm_loss_accum = 0.0
    clm_loss_accum = 0.0
    denoise_loss_accum = 0.0
    recon_loss_accum = 0.0
    total_loss_accum = 0.0
    steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        mlm_labels = batch["mlm_labels"].to(device)
        clm_labels = batch["clm_labels"].to(device)
        denoise_input = batch["denoise_input"].to(device)
        denoise_labels = batch["denoise_labels"].to(device)
        original_tokens = batch["original_tokens"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, denoise_input, None)
        mlm_logits = outputs["mlm_logits"]
        clm_logits = outputs["clm_logits"]
        denoise_logits = outputs["denoise_logits"]
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
        denoise_loss = denoise_loss_fn(
            denoise_logits.view(-1, 2),
            denoise_labels.view(-1)
        )
        recon_loss = recon_loss_fn(
            reconstruction_logits.view(-1, reconstruction_logits.size(-1)),
            original_tokens.view(-1)
        )

        # Combine losses (you can add weights here if desired)
        total_loss = mlm_loss + clm_loss + denoise_loss + recon_loss
        total_loss.backward()
        optimizer.step()

        # Accumulate
        mlm_loss_accum += mlm_loss.item()
        clm_loss_accum += clm_loss.item()
        denoise_loss_accum += denoise_loss.item()
        recon_loss_accum += recon_loss.item()
        total_loss_accum += total_loss.item()
        steps += 1

    # Return the average of each loss
    return {
        "mlm_loss": mlm_loss_accum / steps,
        "clm_loss": clm_loss_accum / steps,
        "denoise_loss": denoise_loss_accum / steps,
        "recon_loss": recon_loss_accum / steps,
        "total_loss": total_loss_accum / steps,
    }

def train_stream(main_model,
                 selector_model,
                 main_optimizer,
                 selector_optimizer,
                 text: str,
                 char2idx: dict,
                 idx2char: list,
                 max_len: int = 40,
                 k_mem: int = 4,
                 device: torch.device = torch.device("cpu")):
    """
    Splits `text` into 40-char chunks, feeds them sequentially.
    After each chunk we:
        1. encode + compute LM / MLM / etc. loss  (supervised)
        2. treat -loss as reward to update the selector (REINFORCE)
        3. push the chunk embedding into the memory bank (FIFO)
    """
    memory_bank: list[torch.Tensor] = []              # list[(E,)]
    max_bank = 256                                    # truncate to this size

    # Loss fns ------------------------------------------------------------
    mlm_loss_fn  = nn.CrossEntropyLoss(ignore_index=-100)
    clm_loss_fn  = nn.CrossEntropyLoss(ignore_index=-100)
    den_loss_fn  = nn.CrossEntropyLoss(ignore_index=2)
    rec_loss_fn  = nn.CrossEntropyLoss(ignore_index=0)

    # --------------------------------------------------------------------
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    for step, chunk in enumerate(chunks, 1):

        # ===== prepare batch (B=1) ======================================
        sample = MultiTaskTextDataset([chunk], char2idx, idx2char,
                                      mask_token="[MASK]", mask_prob=0.2, max_len=max_len)[0]
        batch = collate_fn([sample])
        batch = {k: v.to(device) for k, v in batch.items()}

        # ===== choose K memory slots =====================================
        with torch.no_grad():
            _dummy_out = main_model(batch["input_ids"], batch["denoise_input"], None)
            query_emb  = _dummy_out["chunk_emb"].squeeze(0)     # (E,)

        mem_embs, log_probs_sum = choose_memory(selector_model,
                                                query_emb,
                                                memory_bank,
                                                k_mem,
                                                device)

        # ===== main model forward + backward =============================
        main_model.train()
        main_optimizer.zero_grad()

        out = main_model(batch["input_ids"],
                    batch["denoise_input"],
                    mem_embs)                         # (1, T, …)

        mlm_loss = mlm_loss_fn(out["mlm_logits"].view(-1, main_model.vocab_size),
                               batch["mlm_labels"].view(-1))
        clm_loss = clm_loss_fn(out["clm_logits"].view(-1, main_model.vocab_size),
                               batch["clm_labels"].view(-1))
        den_loss = den_loss_fn(out["denoise_logits"].view(-1, 2),
                               batch["denoise_labels"].view(-1))
        rec_loss = rec_loss_fn(out["reconstruction_logits"].view(-1, main_model.vocab_size),
                               batch["original_tokens"].view(-1))

        loss = mlm_loss + clm_loss + den_loss + 2.0 * rec_loss
        loss.backward()
        main_optimizer.step()

        # ===== RL update for selector ====================================
        # reward = −supervised loss   (smaller loss == better)
        reward = -loss.detach()
        selector_loss = -log_probs_sum * reward

        selector_optimizer.zero_grad()
        selector_loss.backward()
        selector_optimizer.step()

        # ===== book-keeping =============================================
        memory_bank.append(out["chunk_emb"].detach().squeeze(0).cpu())
        # FIFO truncation
        if len(memory_bank) > max_bank:
            memory_bank.pop(0)

        if step % 10 == 0 or step == len(chunks):
            print(f"step {step:03d}/{len(chunks)}  "
                  f"sup-loss {loss.item():6.4f}  sel-loss {selector_loss.item():6.4f}  "
                  f"reward {reward.item():6.4f}")

    return memory_bank

################################################################################
# Putting it all together
################################################################################

if __name__ == "__main__":
    # Example texts
    _short_texts = [
        "Hi",
        "Hello world!",
        "This is a sample text.",
        "Another example text for multi-task learning.",
        # More!,
    ]
    _long_texts = [
        ("Sherlock Holmes took his bottle from the corner of the mantel-piece and his hypodermic syringe from its "
         "neat morocco case. With his long, white, nervous fingers he adjusted the delicate needle and rolled back "
         "his left shirt-cuff. For some little time his eyes rested thoughtfully upon the sinewy forearm and wrist "
         "all dotted and scarred with innumerable puncture-marks."),
        # More!,
    ]

    # 1. Build vocab
    _char2idx, _idx2char = build_vocab(_short_texts + _long_texts)
    _vocab_size = len(_idx2char)

    # 2. Create dataset and dataloader
    _short_text_dataset = MultiTaskTextDataset(_short_texts, _char2idx, _idx2char,
                                    mask_token="[MASK]", mask_prob=0.2, max_len=40)
    _short_text_dataloader = DataLoader(_short_text_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 3. Initialize model and optimizer
    _device = torch.device("mps" if torch.mps.is_available() else "cpu")
    _multi_task_model = DualStreamMultiTaskModel(
        vocab_size=_vocab_size,
        embed_dim=64,
        num_heads=2,
        feedforward_dim=128,
        max_mem_slots=4
    ).to(_device)
    _selector_model = MemorySelector(embed_dim=64, hidden_dim=128).to(_device)

    _multi_task_optimizer = optim.Adam(_multi_task_model.parameters(), lr=2e-4)
    _selector_optimizer = optim.Adam(_selector_model.parameters(), lr=1e-4)


    # 4. Train
    epochs = 5
    print("Short Text Dataset:")
    for epoch in range(epochs):
        losses_dict = train_one_epoch(_multi_task_model, _short_text_dataloader, _multi_task_optimizer, _device)
        print(
            "  "
            f"Epoch {epoch+1}/{epochs}, "
            f"Total: {losses_dict['total_loss']:.4f}, "
            f"MLM: {losses_dict['mlm_loss']:.4f}, "
            f"CLM: {losses_dict['clm_loss']:.4f}, "
            f"Denoise: {losses_dict['denoise_loss']:.4f}, "
            f"Recon: {losses_dict['recon_loss']:.4f}"
        )
    for _text in _long_texts:
        memory_bank = train_stream(
            _multi_task_model, _selector_model,
            _multi_task_optimizer, _selector_optimizer,
            _text, _char2idx, _idx2char,
            max_len=40, k_mem=4, device=_device
        )

    print("Training complete!")
```

I have a relatively small max_len at 40 characters so inevitably, I'm going to need a way to deal with longer texts. I want to have some in-memory store for all previous 40 character chunk embeddings from a document or conversation. I want to modify my DualStreamMultiTaskModel  to use 4 of these chunks at any time when dealing with a new 40 character text chunk. I train a second model to predict which 4-7 chunks from all previously stored chunks to populate these 4-7 slots maybe through RL where the reward is how much the character model benefited from the choices.

I have things mostly wired up.
```
$ date; python -m germ.sandbox.char_encoder; date
Thu May 29 00:27:00 PDT 2025
Short Text Dataset:
  Epoch 1/5, Loss: 11.7554, MLM: 3.7486, CLM: 3.7847, Denoise: 0.7325, Recon: 3.4896
  Epoch 2/5, Loss: 11.7138, MLM: 3.5797, CLM: 3.8708, Denoise: 0.7433, Recon: 3.5199
  Epoch 3/5, Loss: 11.2101, MLM: 3.2616, CLM: 3.7766, Denoise: 0.7173, Recon: 3.4547
  Epoch 4/5, Loss: 11.4370, MLM: 3.7078, CLM: 3.6864, Denoise: 0.6742, Recon: 3.3687
  Epoch 5/5, Loss: 11.7801, MLM: 4.0144, CLM: 3.7376, Denoise: 0.6572, Recon: 3.3710
Long Texts:
  Step 010/10, SL -118.8077, Reward -14.8173, Loss 14.8173 MLM: 3.8783, CLM: 4.0191, Denoise: 0.4416, Recon: 3.2391
Training complete!
Thu May 29 00:27:03 PDT 2025
```