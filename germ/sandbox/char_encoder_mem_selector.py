import torch
import torch.nn as nn

from germ.sandbox.char_encoder import MultiTaskTextDataset, collate_fn

################################################################################
# Model Definition
################################################################################


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
        # No candidate memories → model receives None, selector gets no update
        return None, torch.tensor(0.0, device=device, requires_grad=True)

    cand_embs = torch.stack(bank, dim=0).to(device)              # (M, E)
    logits = memory_selector(query_embedding.to(device), cand_embs)   # (M,)
    probs = torch.softmax(logits, dim=0)

    # Multinomial w/out replacement – sample iteratively
    chosen_idx, log_prob_terms = [], []
    tmp_probs = probs.clone()
    for _ in range(min(k, len(bank))):
        idx = torch.multinomial(tmp_probs, 1).item()
        log_prob_terms.append(torch.log(tmp_probs[idx] + 1e-10))
        chosen_idx.append(idx)
        tmp_probs[idx] = 0.0                              # remove so no repeat
        tmp_probs = tmp_probs / tmp_probs.sum()           # renormalise

    chosen_embs = cand_embs[chosen_idx].unsqueeze(0)      # (1, k, E)
    return chosen_embs, torch.stack(log_prob_terms).sum()


def train_stream(main_model,
                 selector_model,
                 main_optimizer,
                 selector_optimizer,
                 text: str,
                 char2idx: dict,
                 idx2char: list,
                 max_len: int = 40,
                 k_mem: int = 4,
                 ema_beta: float = 0.95,
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
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    clm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    rec_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # --------------------------------------------------------------------
    baseline = None  # first few steps fall back to raw reward
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    for step, chunk in enumerate(chunks, 1):

        # ===== prepare batch (B=1) ======================================
        sample = MultiTaskTextDataset([chunk], char2idx, idx2char,
                                      mask_token="[MASK]", mask_prob=0.2, max_len=max_len)[0]
        batch = collate_fn([sample])
        batch = {k: v.to(device) for k, v in batch.items()}

        # ===== choose K memory slots =====================================
        with torch.no_grad():
            _dummy_out = main_model(batch["input_ids"], None)
            query_emb  = _dummy_out["chunk_emb"].squeeze(0)     # (E,)

        mem_embs, log_probs_sum = choose_memory(selector_model,
                                                query_emb,
                                                memory_bank,
                                                k_mem,
                                                device)

        # ===== main model forward + backward =============================
        main_model.train()
        main_optimizer.zero_grad()

        out = main_model(batch["input_ids"], mem_embs)  # (1, T, …)

        mlm_loss = mlm_loss_fn(out["mlm_logits"].view(-1, main_model.vocab_size),
                               batch["mlm_labels"].view(-1))
        clm_loss = clm_loss_fn(out["clm_logits"].view(-1, main_model.vocab_size),
                               batch["clm_labels"].view(-1))
        rec_loss = rec_loss_fn(out["reconstruction_logits"].view(-1, main_model.vocab_size),
                               batch["original_tokens"].view(-1))

        loss = mlm_loss + clm_loss + 2.0 * rec_loss
        loss.backward()
        main_optimizer.step()

        # ===== RL update for selector ====================================
        if mem_embs is not None:  # only update when something was chosen
            reward = -loss.detach()  # smaller loss == better
            if baseline is None:
                baseline = reward  # first hit initialises EMA
            else:
                baseline = ema_beta * baseline + (1 - ema_beta) * reward
            advantage = reward - baseline  # centre the reward
            selector_loss = -log_probs_sum * advantage

            selector_optimizer.zero_grad()
            selector_loss.backward()
            selector_optimizer.step()

            if step % 10 == 0 or step == len(chunks):
                print(
                    "  "
                    f"Step {step: 3d}/{len(chunks)}, "
                    f"SLoss {selector_loss.item():6.4f}, "
                    f"Reward {reward.item():6.4f}, "
                    f"Baseline {baseline.item():6.4f}, "
                    f"Advantage {advantage.item():6.4f}"
                )

        # ===== book-keeping =============================================
        memory_bank.append(out["chunk_emb"].detach().squeeze(0).cpu())
        # FIFO truncation
        if len(memory_bank) > max_bank:
            memory_bank.pop(0)

        if step % 10 == 0 or step == len(chunks):
            print(
                "  "
                f"Step {step: 3d}/{len(chunks)}, "
                f"MLoss {loss.item():6.4f} "
                f"MLM: {mlm_loss.item():.4f}, "
                f"CLM: {clm_loss.item():.4f}, "
                f"Recon: {rec_loss.item():.4f}"
            )

    return memory_bank
