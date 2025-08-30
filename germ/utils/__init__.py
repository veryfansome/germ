import numpy as np
import time


class TimeBudget:
    def __init__(self, budget_ns: int):
        self.budget_ns = int(budget_ns)
        self.start_ns = time.monotonic_ns()

    @classmethod
    def from_ms(cls, budget_ms: int) -> "TimeBudget":
        return cls(budget_ms * 1_000_000)

    @classmethod
    def from_seconds(cls, budget_seconds: float) -> "TimeBudget":
        return cls(int(budget_seconds * 1_000_000_000))

    def remaining_ns(self) -> int:
        elapsed_ns = time.monotonic_ns() - self.start_ns
        rem = self.budget_ns - elapsed_ns
        return rem if rem > 0 else 0

    def remaining_ms(self) -> int:
        return self.remaining_ns() // 1_000_000

    def remaining_seconds(self) -> int:
        return self.remaining_ns() // 1_000_000_000


def find_near_dup_pairs(embs, normalize: bool = False, sim_threshold: float = 0.90):
    """
    embs: list/array of shape (n, d), float-like
    Returns:
      - near_dup_pairs: list of (i, j, sim) with sim >= threshold
    """
    embs = np.array(embs, dtype=np.float32)
    if normalize:
        embs = normalize_embeddings(embs)

    # Cosine similarity matrix
    sim_matrix = embs @ embs.T

    # Upper triangle mask (exclude diagonal)
    mask = np.triu(sim_matrix, 1) >= sim_threshold
    i_idx, j_idx = np.where(mask)
    sims = sim_matrix[i_idx, j_idx].astype(float)

    # Sort by descending similarity for stability of output
    order = np.argsort(-sims)
    near_dup_pairs = [(int(i_idx[k]), int(j_idx[k]), float(sims[k])) for k in order]

    return near_dup_pairs


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Prevent division by zero by setting zero norms to one
    return embeddings / norms
