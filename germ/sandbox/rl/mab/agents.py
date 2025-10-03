"""
agents.py — classic bandit agents with simple, readable implementations.

We include three widely used strategies:
1) ε-Greedy (with optional optimistic initialization)
2) UCB1 (Upper Confidence Bound)
3) Thompson Sampling for Bernoulli rewards (Beta posterior)

Each agent exposes a minimal interface:
- `select(t)` → choose an arm at time t (1-based t to match typical formulas).
- `update(arm, reward)` → incorporate the observed reward.

Why these three?
- ε-Greedy: Baseline for *exploration vs. exploitation* with almost no math.
- UCB1: Frequentist approach that adds a *confidence bonus* to optimistic actions.
- Thompson Sampling: Bayesian approach that “samples a belief” and is often very strong.

References
----------
- Sutton & Barto, Ch. 2 (ε-greedy, incremental mean updates)
  https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
- Auer, Cesa-Bianchi, Fischer (2002) — UCB1 and finite-time regret bounds
  https://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Auer%2Bal%3A2002.pdf
- Russo, Van Roy, Kazerouni, Osband, Wen (2017) — Tutorial on Thompson Sampling
  https://arxiv.org/abs/1707.02038
- Lattimore & Szepesvári (2020) — rigorous coverage of bandits & regret
  https://tor-lattimore.com/downloads/book/book.pdf
"""
import numpy as np


class AgentBase:
    """
    Shared scaffolding for bandit agents.

    Responsibilities
    ---------------
    - Hold common state:
        `n[a]` = how many times we’ve pulled arm a,
        `q[a]` = our current estimate of the mean reward for arm a (empirical mean).
    - Provide a consistent RNG for reproducibility.
    - Offer a default `update` using an *incremental mean* to avoid storing histories.

    Why incremental means?
    ----------------------
    The textbook update:
        q_new = q_old + (reward - q_old) / n
    is numerically stable, online (O(1) per step), and minimizes dependencies.

    Subclasses must implement:
    - `select(self, t)` returning an arm index in [0, K).
    """
    def __init__(self, K, seed=None):
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Reset counts and value estimates; subclasses may extend."""
        self.n = np.zeros(self.K, dtype=int)  # pull counts per arm
        self.q = np.zeros(self.K, dtype=float)  # empirical means per arm

    def select(self, t):
        """Pick an arm at 1-based time t. Must be overridden in subclasses."""
        raise NotImplementedError

    def update(self, arm, reward):
        """
        Default update: incremental (running) mean for arm `arm`.

        This keeps `q[arm]` equal to the empirical average of all rewards seen on that arm.
        Many algorithms (ε-greedy, UCB1 variants) use this estimate as their base value.
        """
        self.n[arm] += 1
        # incremental mean
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]

class EpsilonGreedy(AgentBase):
    """
    ε-Greedy action selection.

    Rule
    ----
    With probability ε, pick a *uniformly random* arm (exploration).
    With probability 1-ε, pick argmax_a q[a] (exploitation).

    Parameters
    ----------
    epsilon : float in [0,1]
        Exploration rate. Typical starters: 0.1 or 0.01.
        Higher ε explores more but sacrifices short-term reward.
    optimistic_init : float
        Initialize all q[a] to this optimistic value. For Bernoulli, 1.0 is a common trick.
        Effect: the agent is biased to try under-sampled arms early (since their estimates
        start high and then decay toward truth).
    seed : int or None
        RNG seed for reproducibility.

    Notes
    -----
    - Annealing ε (e.g., ε_t ∝ 1/t) can improve long-run performance but hides the
      simple exploration trade-off; we keep it fixed here for clarity.
    - Ties in argmax: NumPy’s argmax picks the first occurrence; behavior is deterministic
      given `q`. If you want fair tie-breaking, inject tiny noise or break ties randomly.
    """
    def __init__(self, K, epsilon=0.1, seed=None, optimistic_init=0.0):
        super().__init__(K, seed)
        self.epsilon = float(epsilon)
        self.optimistic_init = float(optimistic_init)
        self.q[:] = self.optimistic_init

    def select(self, t):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.K)  # Explore
        return int(np.argmax(self.q))  # Exploit

class UCB1(AgentBase):
    """
    UCB1 (Upper Confidence Bound) — “optimism in the face of uncertainty”.

    Idea
    ----
    Add a bonus to each empirical mean that shrinks with additional pulls:
        UCB[a] = q[a] + sqrt( (c * log t) / n[a] )
    Choose the arm with the largest UCB.

    Intuition
    ---------
    - Arms with few samples keep a large uncertainty bonus → encouraged exploration.
    - As n[a] grows, the bonus shrinks like O(√(log t / n[a])) → more exploitation.

    Parameters
    ----------
    c : float (default 2.0)
        Controls exploration strength; the original UCB1 uses a fixed constant (often 2).
        We expose it as a knob since practical performance is problem-dependent.
    seed : int or None

    Implementation details
    ----------------------
    - We force each arm to be tried *once* before using the formula (since n[a]=0 would
      divide by zero and, conceptually, each arm deserves at least one sample).
    - Our `t` is 1-based to match usual statements of the bound.
    - This is the vanilla UCB1; variants like UCB2 and UCB1-Tuned exist for variance-aware
      bonuses (see Auer et al. 2002, Section 4).

    Reference
    ---------
    Auer, Cesa-Bianchi, Fischer (2002): “Finite-time Analysis of the Multiarmed Bandit Problem”
    https://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Auer%2Bal%3A2002.pdf
    """
    def __init__(self, K, seed=None, c=2.0):
        super().__init__(K, seed)
        self.c = float(c)

    def select(self, t):
        # Ensure each arm is tried once to initialize n[a] and q[a]
        for a in range(self.K):
            if self.n[a] == 0:
                return a
        # Classic UCB1 bonus; `c` lets you over/under-explore relative to the textbook constant
        ucb = self.q + np.sqrt(self.c * np.log(max(1, t)) / self.n)
        return int(np.argmax(ucb))

class ThompsonBernoulli(AgentBase):
    """
    Thompson Sampling for Bernoulli rewards using Beta posteriors.

    Model
    -----
    For each arm a, maintain a Beta(α[a], β[a]) posterior over μ_a.
    - Prior: α0, β0 (default 1,1 = uniform over [0,1])
    - After observing reward r ∈ {0,1}:
        α[a] ← α[a] + r
        β[a] ← β[a] + (1 - r)

    Action rule
    -----------
    At each step, sample a *plausible* mean from each posterior:
        θ[a] ~ Beta(α[a], β[a])
    and pick argmax_a θ[a]. This is a randomized but *probability-matching* strategy:
    arms that are likely to be optimal under the posterior are picked more often.

    Parameters
    ----------
    alpha0, beta0 : float
        Prior hyperparameters; for Bernoulli rewards, (1,1) is uninformative and common.
        If you expect sparse successes, you can bias toward smaller μ with alpha0<beta0.

    Notes
    -----
    - We maintain `q[a]` as the empirical mean only for logging/plots; the *decision* uses
      the Beta posterior samples.
    - TS often matches or beats UCB-like methods in practice across many problem classes,
      while being simple to implement.

    Reference
    ---------
    Russo, Van Roy, Kazerouni, Osband, Wen (2017): “A Tutorial on Thompson Sampling”
    https://arxiv.org/abs/1707.02038
    """
    def __init__(self, K, seed=None, alpha0=1.0, beta0=1.0):
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        super().__init__(K, seed)

    def reset(self):
        """Reset empirical means and counts, plus the Beta priors."""
        super().reset()
        self.alpha = np.full(self.K, self.alpha0, dtype=float)
        self.beta = np.full(self.K, self.beta0, dtype=float)

    def select(self, t):
        """Draw a sample θ[a] ~ Beta(α[a], β[a]) for each arm and choose the max."""
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        # Keep `q` in sync via the base incremental-mean update
        super().update(arm, reward)
        # Conjugate Beta-Bernoulli update to the posterior for this arm
        if reward > 0.5:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0
