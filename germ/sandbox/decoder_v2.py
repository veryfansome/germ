import torch
import torch.nn as nn
import torch.nn.functional as F

#  Decoder‑only Transformer (≤ 4k context) with Rotary Positional Embeddings
#  * Pure causal attention (no local masks)
#  * Rotary position encoding (RoPE)
#  * Uses PyTorch 2.7's fused scaled‑dot‑product‑attention (SDPA)
#  * Weight‑tied output head
#  * Tuned for Apple M‑series (MPS) but runs on CPU/GPU as well


class RotaryEmbedding(nn.Module):
    """Pre‑computes RoPE cosine/sine tables and applies them to a (B,H,L,D) tensor."""

    def __init__(self, dim: int, *, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        if dim % 2:
            raise ValueError(f"Rotary dimension must be even, got {dim}.")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [L, D/2]

        self.register_buffer("_cos", torch.cos(freqs))
        self.register_buffer("_sin", torch.sin(freqs))
        self._cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _lookup(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (x.device, x.dtype)
        if key not in self._cache:
            self._cache[key] = (
                self._cos.to(device=x.device, dtype=x.dtype),
                self._sin.to(device=x.device, dtype=x.dtype),
            )
        return self._cache[key]

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        cos, sin = self._lookup(x)
        cos = cos[:seq_len][None, None]  # (1,1,L,D/2)
        sin = sin[:seq_len][None, None]

        x_even, x_odd = x[..., ::2], x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd  = x_even * sin + x_odd * cos

        # interleave even/odd dimensions → original ordering
        rot = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
        return rot


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and SDPA."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, *, max_seq_len: int):
        super().__init__()

        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        if self.head_dim % 2:
            raise ValueError(f"head_dim must be even, got {self.head_dim}.")

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, L, E]
        B, L, E = x.shape
        qkv = self.qkv_proj(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, L, H, D]
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))  # [B, H, L, D]

        # Apply rotary position encoding (in‑place safe on MPS)
        q = self.rope(q, L)
        k = self.rope(k, L)

        # Decide masking strategy
        if pad_mask is None:  # Fast path when no padding
            attn_mask = None
            is_causal = True
        else:  # Slow-path: need explicit mask
            key_mask = pad_mask[:, None, None, :]  # (B,1,1,L)
            # Causal part, shared by every batch/head
            causal = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=x.device), 1
            ).unsqueeze(0).unsqueeze(0)  # (1,1,L,L)

            attn_mask = causal | key_mask  # (B,1,L,L)  True = MASK
            is_causal = False  # causal already in mask

        # Fused attention
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,  # may be None
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )  # [B, H, L, D]

        attn = attn.transpose(1, 2).reshape(B, L, E)
        return self.out_proj(attn)


class FeedForward(nn.Module):
    """SwiGLU feed‑forward block."""

    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w12 = nn.Linear(d_model, dim_ff * 2, bias=True)
        self.w3 = nn.Linear(dim_ff, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)  # gate
        x = self.w3(F.silu(a) * b)  # projection
        return self.drop(x)  # dropout


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float, *, max_seq_len: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout, max_seq_len=max_seq_len)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        x = x + self.drop(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.ff(self.norm2(x))
        return x


class DecoderModel(nn.Module):
    """Decoder‑only Transformer with RoPE.  Suitable for ≤ 4k‑token contexts."""

    def __init__(
        self,
        vocab_size: int, *,
        d_model: int = 768,
        n_heads: int = 12,
        dim_ff: int | None = None,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        pad_token_id: int = 0
    ) -> None:
        super().__init__()

        if not dim_ff:
            dim_ff = 4 * d_model

        self.pad_token_id = pad_token_id
        self.token_emb = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_heads, dim_ff, dropout, max_seq_len=max_seq_len)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        del self.lm_head.weight  # Drop the original parameter
        self.lm_head.weight = self.token_emb.weight  # Weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """`input_ids`shape [B, L] → returns logits [B, L, vocab]."""
        key_padding_mask = input_ids.eq(self.pad_token_id)  # [B, L]
        x = self.token_emb(input_ids)  # [B, L, E]
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        x = self.norm_out(x)
        return self.lm_head(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = DecoderModel(vocab_size=10_000, max_seq_len=4096).to(device)
    #model = torch.compile(model, mode="reduce-overhead")
    dummy_input = torch.randint(0, 10_000, (2, 128), device=device)

    logits = model(dummy_input)
    print("logits.shape =", logits.shape)  # -> [2, 128, 10000]
