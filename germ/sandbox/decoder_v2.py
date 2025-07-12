import math
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

        # Interleave even/odd dimensions → original ordering
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
        self.drop = nn.Dropout(dropout)
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
        if pad_mask is None or not pad_mask.any():  # Fast path when no padding
            attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=True,
            )  # [B, H, L, D]
        else:  # Slow-path: need explicit mask
            # pad_mask: [B, L]  (True = PAD)
            causal = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=x.device), 1
            )  # [L, L]

            key_mask = pad_mask[:, None, None, :]  # [B, 1, 1, L]
            bool_mask = causal[None, None] | key_mask  # [B, 1, L, L]

            # Float mask: 0 for keep, -inf for mask  (FP32 is fine for all dtypes)
            attn_mask = torch.zeros_like(bool_mask, dtype=torch.float32)
            attn_mask = attn_mask.masked_fill(bool_mask, float("-inf"))  # [B, 1, L, L]

            # Expand over heads and flatten to (B*H, L, L)
            attn_mask = attn_mask.expand(B, self.n_heads, L, L) \
                .reshape(B * self.n_heads, L, L)

            # Flatten q/k/v the same way
            q_ = q.reshape(B * self.n_heads, L, self.head_dim)
            k_ = k.reshape_as(q_)
            v_ = v.reshape_as(q_)

            attn = F.scaled_dot_product_attention(
                q_, k_, v_,
                attn_mask=attn_mask,
                dropout_p=self.drop.p if self.training else 0.0,
                # is_causal must stay False when an explicit mask is given
            ).view(B, self.n_heads, L, self.head_dim)

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
        gate, value = self.w12(x).chunk(2, dim=-1)
        # From https://arxiv.org/pdf/2002.05202 (Shazeer 2020)
        # SwiGLU(x) = Swish(xW) ⊗ (xV)
        x = self.w3(F.silu(gate) * value)
        return self.drop(x)  # dropout


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float, *, max_seq_len: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout, max_seq_len=max_seq_len)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
    ):
        # Pre-compute broadcast mask once
        if key_padding_mask is not None:
            pad = key_padding_mask.unsqueeze(-1)  # [B, L, 1]
        else:
            pad = None

        # Attention block
        x_res = x
        x_norm = self.norm1(x)
        if pad is not None:
            x_norm = x_norm.masked_fill(pad, 0.0)  # zero-out pad queries
        # From https://aclanthology.org/2024.sigul-1.35.pdf
        # Apply residual dropout after the out-proj, before addition
        x = x_res + self.drop(self.attn(x_norm, key_padding_mask))

        # Feed-forward block
        x_res = x
        x_norm = self.norm2(x)
        if pad is not None:
            x_norm = x_norm.masked_fill(pad, 0.0)
        x = x_res + self.ff(x_norm)

        # Ensure next layer starts with clean PAD rows
        if pad is not None:
            x = x.masked_fill(pad, 0.0)

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
        self.token_emb_scale = math.sqrt(d_model)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_heads, dim_ff, dropout, max_seq_len=max_seq_len)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """`input_ids`shape [B, L] → returns logits [B, L, vocab]."""
        key_padding_mask = input_ids.eq(self.pad_token_id)  # [B, L]
        x = self.token_emb(input_ids) * self.token_emb_scale  # [B, L, E]
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        x = self.norm_out(x)
        return self.lm_head(x)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


if __name__ == "__main__":
    torch.manual_seed(0)

    VOCAB = 10_000
    PAD_ID = 0

    DTYPE = torch.float16  # torch.bfloat16 once MPS supports it stably
    DEVICE = torch.device("mps")
    MODEL = DecoderModel(
        vocab_size=VOCAB,
        max_seq_len=4096,
        pad_token_id=PAD_ID,
    ).to(DEVICE, DTYPE)
    MODEL.apply(_init_weights)
    #model = torch.compile(model, mode="reduce-overhead")

    MODEL.eval()


    def rand_batch(batch, seqlen, pad_prob=0.3):
        """Return (input_ids, tgt_ids) with random padding."""
        inp = torch.randint(1, VOCAB, (batch, seqlen), device=DEVICE)
        if pad_prob:
            mask = torch.rand_like(inp.float()) < pad_prob
            inp = inp.masked_fill(mask, PAD_ID)
        # Next-token prediction target (shifted left)
        tgt = inp.roll(-1, dims=1)
        tgt[:, -1] = PAD_ID  # last token unused
        return inp, tgt


    def test_forward_backward():
        """Forward + loss + backward in train mode."""
        MODEL.train()
        optim = torch.optim.AdamW(MODEL.parameters(), lr=0.01)
        for B, L in [(2, 128), (4, 257)]:  # Include odd length
            inp, tgt = rand_batch(B, L, pad_prob=0.4)
            logits = MODEL(inp)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, VOCAB),
                tgt.view(-1),
                ignore_index=PAD_ID
            )
            loss.backward()
            assert not torch.isnan(loss), "NaN loss"
            # Simple grad sanity: some grad norms must be non-zero
            gnorm = math.sqrt(sum(p.grad.pow(2).sum().item()
                                  for p in MODEL.parameters() if p.grad is not None))
            assert gnorm > 0, "no gradients propagated"
            optim.zero_grad()
        MODEL.eval()


    @torch.inference_mode()
    def test_causality():
        """Changing future tokens must not alter past logits."""
        B, L = 1, 64
        base, _ = rand_batch(B, L, pad_prob=0.0)
        mod = base.clone()
        # Alter tokens in the *future* half
        mod[:, L // 2:] = torch.randint(1, VOCAB, (B, L // 2), device=DEVICE)
        out_base = MODEL(base)[:, :L // 2]
        out_mod = MODEL(mod)[:, :L // 2]
        torch.testing.assert_close(
            out_base, out_mod, rtol=0, atol=1e-3,
            msg="future context leaked into past logits"
        )


    def test_compile():
        """Ensure model is torch.compile-able on the selected device."""
        compiled = torch.compile(MODEL, mode="reduce-overhead")
        inp, _ = rand_batch(2, 128, pad_prob=0.2)
        _ = compiled(inp)   # just a dry run


    with torch.autocast(device_type=DEVICE.type, dtype=DTYPE):
        for fn in (test_forward_backward, test_causality, test_compile):
            fn()
    print("✓ all sanity tests passed.")