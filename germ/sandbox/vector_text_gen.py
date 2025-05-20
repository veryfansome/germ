import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from germ.sandbox.e5_token_embedding_model import TokenEmbeddingModel


################################################################################
# 2.  TWO‑STAGE CONTROLLER                                                    #
################################################################################

class TwoStageController(nn.Module):
    """Predicts (i) **semantic intention vector** and (ii) insertion position."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 1024,
                 max_buffer_len: int = 32):
        super().__init__()
        self.fc = nn.Linear(embed_dim * 2, hidden_dim)
        self.token_head = nn.Linear(hidden_dim, embed_dim)
        self.pos_head = nn.Linear(hidden_dim, max_buffer_len + 1)  # +1 → append
        self.max_buffer_len = max_buffer_len

    def forward(self, target_vec: torch.Tensor, summary_vec: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.fc(torch.cat([target_vec, summary_vec], dim=-1)))
        token_vec = F.normalize(self.token_head(h), p=2, dim=-1)
        pos_logits = self.pos_head(h)
        return token_vec, pos_logits

################################################################################
# 3.  VECTOR‑CONTROLLED TEXT GENERATOR                                        #
################################################################################

class VectorControlledGenerator:
    """Maintains
        • FAISS index of seen tokens  → (id → embedding)
        • a mutable *token buffer*    → ordered list[str]
        • a *hierarchical* summary    → list[np.ndarray] (level‑0, level‑1, …)
    """

    def __init__(self,
                 embedder: TokenEmbeddingModel,
                 controller: TwoStageController,
                 device: str = "cpu",
                 top_k: int = 8,
                 min_sim: float = 0.60,
                 rollup_span: int = 16):
        self.embedder = embedder.to(device)
        self.controller = controller.to(device)
        self.device = device
        self.top_k = top_k
        self.min_sim = min_sim
        self.rollup_span = rollup_span

        self.token_buffer: List[str] = []
        self.summary_vectors: List[np.ndarray] = []  # hierarchical stack

        dim = embedder.embed_dim
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._token2id = {}
        self._id2token = {}
        self._next_id = 0

    # ---------------------------------------------------------------------
    # TOKEN BANK
    # ---------------------------------------------------------------------
    def learn_token(self, token: str):
        if token in self._token2id:
            return
        vec = self.embedder([token]).detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vec)
        tid = self._next_id
        self.index.add_with_ids(vec, np.array([tid], dtype=np.int64))
        self._token2id[token] = tid
        self._id2token[tid] = token
        self._next_id += 1

    # ---------------------------------------------------------------------
    # SUMMARY MAINTENANCE (hierarchical) ----------------------------------
    # ---------------------------------------------------------------------
    def _update_summary(self, new_vec: np.ndarray):
        """Roll up older tokens every *rollup_span* words."""
        if not self.summary_vectors:
            self.summary_vectors.append(new_vec.copy())
            return
        lvl0 = self.summary_vectors[0] * len(self.token_buffer[:-1])  # prior count
        lvl0 = (lvl0 + new_vec) / len(self.token_buffer)
        self.summary_vectors[0] = lvl0
        # Optionally roll top level into higher hierarchy (omitted for brevity)

    def _get_summary_vec(self) -> torch.Tensor:
        if not self.summary_vectors:
            return torch.zeros(1, self.embedder.embed_dim, device=self.device)
        v = self.summary_vectors[0]
        return torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)

    # ---------------------------------------------------------------------
    # MAIN GENERATION STEP -------------------------------------------------
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def step(self, target_vec: torch.Tensor) -> bool:
        """Run one *controller → retrieval → placement* cycle.

        Returns True if **completion** is predicted, else False."""
        summary_vec = self._get_summary_vec()
        tok_vec_pred, pos_logits = self.controller(target_vec, summary_vec)

        # 1‑a  Retrieve candidate tokens -----------------------------------
        q = tok_vec_pred.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(q)
        dists, ids = self.index.search(q, self.top_k)
        best_id = ids[0][0]
        sim = dists[0][0]
        if best_id == -1 or sim < self.min_sim:
            return True  # force completion
        token = self._id2token[best_id]

        # 1‑b  Choose position --------------------------------------------
        buf_len = len(self.token_buffer)
        pos = int(pos_logits.argmax(dim=-1).item())
        pos = min(pos, buf_len)  # clamp; >=len -> append

        # 1‑c  Mutate state -----------------------------------------------
        self.token_buffer.insert(pos, token)
        self._update_summary(q[0])
        return False

    # ---------------------------------------------------------------------
    # USER‑FACING HELPERS --------------------------------------------------
    # ---------------------------------------------------------------------
    def empty_token_buffer(self) -> str:
        text = "".join(self.token_buffer)
        self.token_buffer.clear()
        self.summary_vectors.clear()
        return text

    def buffer_text(self) -> str:
        return "".join(self.token_buffer)

################################################################################
# 4.  TRAINING STUB (token selector + placer)                                 #
################################################################################
#  Fine‑tuning logic is domain‑specific.  See README for a full script.  Sketch:
#
#   model = TwoStageController(embed_dim=768)
#   optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
#   for eg in dataset:
#       target_vec = ...
#       summary_vec = ...
#       gold_tok_vec = embedder([gold_token])
#       gold_pos = torch.tensor([gold_position])
#       pred_vec, pos_logits = model(target_vec, summary_vec)
#       loss_cos = 1 - F.cosine_similarity(pred_vec, gold_tok_vec).mean()
#       loss_pos = F.cross_entropy(pos_logits, gold_pos)
#       loss = loss_cos + loss_pos
#       loss.backward(); optimiser.step(); optimiser.zero_grad()
#
#  Use *hard negatives* by sampling other tokens for the same position.
################################################################################
