"""
bandit.py — minimal environments for K-armed (multi-armed) bandit experiments.

This file implements a Bernoulli bandit with optional *non-stationarity* (drifting arm means).
It is deliberately tiny so that the “agent” code stays the focus.

Why Bernoulli?
- It’s the canonical didactic setting: each arm a has an unknown success probability μ_a.
- Rewards are in {0,1}, making interpretation + plotting straightforward.
- Many bandit algorithms (ε-greedy, UCB1, Thompson sampling with Beta posterior) have
  simple, closed-form updates in the Bernoulli case.

Key ideas embodied here:
- `step(arm)` returns a single stochastic reward ~ Bernoulli(μ_arm).
- `best_mean` tracks the *current* best arm’s mean — useful for computing regret curves,
  and it also adapts if the environment is drifting (non-stationary).

References (high-quality intros):
- Sutton & Barto, *Reinforcement Learning: An Introduction*, Ch. 2 (Multi-armed bandits)
  https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
- Lattimore & Szepesvári, *Bandit Algorithms*, Cambridge (free online edition)
  https://tor-lattimore.com/downloads/book/book.pdf
"""
import numpy as np


class BernoulliBandit:
    """
    A K-armed Bernoulli bandit.

    Parameters
    ----------
    probs : array-like of shape (K,)
        The true (hidden) success probabilities μ_a ∈ [0,1] for each arm a.
        Example: [0.1, 0.2, 0.35, 0.7]
    drifting : bool, default False
        If True, the environment is *non-stationary*: after each pull, each μ_a receives
        an additive Gaussian perturbation N(0, drift_std), then we clip to [0,1].
        This lets you test how stationary algorithms degrade and why discounting / windows help.
    drift_std : float, default 0.0
        The standard deviation of the Gaussian drift. Typical values: 0.001–0.01.
        Larger means faster change; too large turns the task into pure noise.
    seed : int or None
        Seed for the environment’s RNG. Fix this for reproducible simulations.

    Attributes
    ----------
    K : int
        Number of arms.
    probs : np.ndarray of shape (K,)
        Current arm success probabilities. If `drifting=True`, this will evolve over time.
    best_mean : float
        The *current* best mean, i.e., max_a μ_a. We expose this so callers can compute
        instantaneous regret μ* - r_t and cumulative regret ∑(μ* - r_t) cleanly.

    Design notes
    ------------
    - We keep the environment minimal and “pure”: it doesn’t track episodes or totals;
      it only gives you a reward on demand. This mirrors the bandit formalism.
    - Drift is implemented *after* sampling a reward, so the reward reflects the
      pre-drift μ_a at time t, and drift affects the time t+1 means. Either ordering is fine,
      but this choice matches common simulation code.
    """

    def __init__(self, probs, drifting=False, drift_std=0.0, seed=None):
        """
        probs: list/array of true success probabilities per arm in [0,1].
        drifting: if True, each step adds N(0, drift_std) then clips to [0,1].
        """
        self.rng = np.random.default_rng(seed)
        self.probs = np.array(probs, dtype=float)
        self.K = len(self.probs)
        self.drifting = drifting
        self.drift_std = float(drift_std)
        # Track the best current mean for regret calculations (updates if drifting)
        self.best_mean = self.probs.max()

    def step(self, arm):
        """
        Pull an arm and get a stochastic reward.

        Parameters
        ----------
        arm : int
            Index in [0, K). No safety checks here for speed and simplicity.

        Returns
        -------
        reward : float
            1.0 with probability μ_arm, else 0.0.

        Side effects
        ------------
        - If `drifting=True` and `drift_std>0`, all arm means are perturbed by N(0, drift_std)
          and clipped to [0,1] *after* sampling the reward for this step.
        - `best_mean` is updated to reflect the current (possibly drifted) means.

        Why float rewards?
        -------------------
        We return float for compatibility with Gaussian/general bandits if you extend this file.
        """
        p = self.probs[arm]
        r = self.rng.random() < p  # Bernoulli trial
        if self.drifting and self.drift_std > 0:
            self.probs += self.rng.normal(0.0, self.drift_std, size=self.K)
            self.probs = np.clip(self.probs, 0.0, 1.0)
            self.best_mean = self.probs.max()
        return float(r)

    def reset(self, probs=None):
        """
        Reset the environment (optionally with new means).

        Parameters
        ----------
        probs : array-like or None
            If provided, replace the arm means with this vector; otherwise reuse current means.
        """
        if probs is not None:
            self.probs = np.array(probs, dtype=float)
        self.best_mean = self.probs.max()
