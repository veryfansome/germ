import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from germ.sandbox.char_encoder import DualStreamMultiTaskModel, MultiTaskTextDataset, collate_fn, train_one_epoch
from germ.sandbox.char_encoder_mem_selector import MemorySelector, train_stream
from germ.sandbox.char_encoder_data import char2idx, idx2char, long_text_examples, short_text_examples


# Example texts

# 1. Build vocab
vocab_size = len(idx2char)
print(f"Vocab size: {vocab_size}, vocab: {char2idx.keys()}")

# 2. Create dataset and dataloader
short_text_dataset = MultiTaskTextDataset(
    short_text_examples, char2idx, idx2char, mask_token="[MASK]", mask_prob=0.2, max_len=40
)
short_text_dataloader = DataLoader(
    short_text_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)

# 3. Initialize model and optimizer
device = torch.device("mps" if torch.mps.is_available() else "cpu")
multi_task_model = DualStreamMultiTaskModel(
    vocab_size=vocab_size,
    embed_dim=256,
    num_heads=8,
    feedforward_dim=1024,
    max_seq_len=256,
    max_mem_slots=4
).to(device)
selector_model = MemorySelector(
    embed_dim=256,
    hidden_dim=1024
).to(device)

multi_task_optimizer = optim.Adam(multi_task_model.parameters(), lr=2e-4)
selector_optimizer = optim.Adam(selector_model.parameters(), lr=1e-4)

# 4. Train
epochs = 5
print("Short Text Dataset:")
for epoch in range(epochs):
    losses_dict = train_one_epoch(multi_task_model, short_text_dataloader, multi_task_optimizer, device)
    print(
        "  "
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {losses_dict['total_loss']:.4f}, "
        f"MLM: {losses_dict['mlm_loss']:.4f}, "
        f"CLM: {losses_dict['clm_loss']:.4f}, "
        f"Recon: {losses_dict['recon_loss']:.4f}"
    )
print("Long Texts:")
for _text in long_text_examples:
    memory_bank = train_stream(
        multi_task_model, selector_model,
        multi_task_optimizer, selector_optimizer,
        _text, char2idx, idx2char,
        max_len=40, k_mem=4, device=device
    )

print("Training complete!")

# TODO:
#   - Baseline before RL – measure loss with the selector disabled (K=0) to confirm the policy is learning
#     something non-trivial.
#   - Critic / baseline – add a tiny value-network to predict expected reward, subtract it from the raw reward to
#     cut variance.
#   - Better memory routing – feed the selected embeddings through a small FiLM or cross-attention block instead of
#     blunt concatenation.
#   - Curriculum – start training without memory, then enable selector after N epochs; avoids policy thrashing
#     while the encoder is still random.
#   - Recon head – I ignore [PAD] (0) so any [UNK] characters are counted as errors.
#   - Byte-level vocab? Character-level MLM/CLM gives you ~128 tokens out-of-the-box if you switch to raw bytes
#     (or UTF-8 code-units) and saves you the whole custom-vocab hassle. The flip side is that recon loss becomes
#     much easier (byte-by-byte copying is trivial), so you may need to scale the reconstruction coefficient back
#     down.
#   - Memory selector variance – Four slots out of 256 candidates is a huge action space. You’ll probably converge
#     faster if you:
#       - use top-k arg-sampling during warm-up,
#       - add an entropy bonus to keep exploration alive,
#       - pre-train the selector with a cheap heuristic (e.g. dot-product similarity).
#   - Gradient flow through mem_proj – Right now the memory encoder never gets a direct reconstruction loss. If you
#     want end-to-end differentiability you can try:
#       - summing the recon loss over both input and memory tokens
#       - or adding an auxiliary contrastive loss between mem_proj(memory_emb) and the fresh‐chunk embedding.
#   - Gradient flow through mem_proj – Right now the memory encoder never gets a direct reconstruction loss. If you
#   - Replay buffer – At >10 k chunks you’ll spend more time on selector forward than on the main model. Consider
#     K-means or product quantisation to keep the buffer compact.
