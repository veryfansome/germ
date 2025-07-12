import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache


@lru_cache(maxsize=None)
def build_staggered_local_causal_mask(
        seq_len: int,
        local_window_size: int,
        n_heads: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Returns a tensor of shape [n_heads, seq_len, seq_len] with
    0.0 on visible positions and -∞ on masked positions.

    Visibility for head h (0-based):
        token i can attend to j  ⇔  0 ≤ i-j < (h+1)*local_window_size
    """
    device = torch.device(device)  # hash-stable key
    idx     = torch.arange(seq_len, device=device)  # [T]
    rel     = idx[None] - idx[:, None]  # [T, T] (i−j)

    window  = torch.arange(1, n_heads + 1, device=device) * local_window_size
    window  = window[:, None, None]  # broadcast ready

    valid   = (rel >= 0).unsqueeze(0) & (rel.unsqueeze(0) < window)   # [H,T,T]

    # torch.finfo gives the proper minimal finite value for fp16/fp32/bf16
    minus_inf = torch.finfo(dtype).min
    mask = torch.where(valid,
                       torch.zeros((), dtype=dtype, device=device),
                       torch.full((), minus_inf, dtype=dtype, device=device))
    return mask


def get_sinusoidal_positional_encoding(seq_len, d_model):
    """
    Create sinusoidal positional encodings, compatible with the transformer architecture.
    Returned shape: [seq_len, d_model]
    """
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # Shape [seq_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc


class DecoderLayer(nn.Module):
    """
    A single layer of a decoder-only Transformer:
      - Self-attention (with custom causal mask)
      - MLP feedforward
    """
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)  # 2× for gate

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)

    def _feed_forward(self, x):
        # SwiGLU
        a, b = self.linear1(x).chunk(2, dim=-1)
        return self.linear2(self.dropout(a * F.silu(b)))

    def forward(self, x, attn_mask=None):
        # Self-attention (decoder-only, causal)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward network
        x = self.norm2(x + self._feed_forward(x))
        return x


class DecoderModel(nn.Module):
    """
    A decoder-only Transformer model that uses:
      - An embedding layer
      - Positional embeddings
      - Multiple DecoderLayers
      - Staggered causal attention for each head
    """
    def __init__(
            self,
            vocab_size: int,
            # d_model should be divisible by n_heads for multihead attention
            d_model: int = 512,
            n_heads: int = 8,
            local_window_size: int = 64,
            max_seq_len: int = 1024,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            num_layers: int = 6,
    ):
        super().__init__()
        self.local_window_size = local_window_size
        self.d_model = d_model
        self.n_heads = n_heads

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer('positional_encoding', get_sinusoidal_positional_encoding(max_seq_len, d_model))

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie lm_head and token_embedding to simplify the model and reduce the number of parameters.
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, pad_token_id=0):
        """
        input_ids shape: [batch_size, seq_len]
        """
        bsz, seq_len = input_ids.size()

        runtime_dtype = self.token_embedding.weight.dtype  # safe & always set

        # Build staggered mask:
        base_mask = build_staggered_local_causal_mask(
            seq_len,
            self.local_window_size,
            self.n_heads,
            dtype=runtime_dtype,
            device=input_ids.device,
        )  # [n_heads, seq_len, seq_len]

        # Broadcast to [bsz*n_heads, seq_len, seq_len] so each head in each batch has a separate view
        attn_mask = (
            base_mask
            .unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
            .expand(bsz, -1, -1, -1)  # [bsz, n_heads, seq_len, seq_len]
            .reshape(bsz * self.n_heads, seq_len, seq_len)
        )  # [bsz*n_heads, seq_len, seq_len]

        # Create a mask for padded positions so we don't want to attend to them
        pad_positions = (input_ids == pad_token_id)  # shape: [bsz, seq_len]
        pad_mask = (
            pad_positions
            .unsqueeze(1)  # [bsz, 1, seq_len]
            .expand(-1, seq_len, -1)  # [bsz, seq_len, seq_len]
            .unsqueeze(1)  # [bsz, 1, seq_len, seq_len]
            .expand(-1, self.n_heads, -1, -1)  # [bsz, n_heads, seq_len, seq_len]
            .reshape(bsz * self.n_heads, seq_len, seq_len)
        )
        attn_mask = attn_mask.masked_fill(pad_mask, torch.finfo(runtime_dtype).min)

        # Token + positional embeddings
        token_embeddings = self.token_embedding(input_ids)  # [bsz, seq_len, d_model]
        pos_embeddings = self.positional_encoding[:seq_len].to(token_embeddings)
        x = token_embeddings + pos_embeddings

        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # Final layer norm and output head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [bsz, seq_len, vocab_size]
        return logits


# Example usage:
if __name__ == "__main__":
    # Define some hyperparameters
    _vocab_size = 10000
    _batch_size = 2
    _seq_length = 128

    # Instantiate the model
    model = DecoderModel(
        vocab_size=_vocab_size,
        d_model=512,
        n_heads=8,
        dim_feedforward=2048,
        num_layers=6,
        dropout=0.1,
        max_seq_len=1024,
        local_window_size=64
    )

    # Create some dummy input data
    dummy_input = torch.randint(0, _vocab_size, (_batch_size, _seq_length))

    # Forward pass
    output_logits = model(dummy_input)
    print("Output shape:", output_logits.shape)  # Should be [2, 128, 10000] in this example
