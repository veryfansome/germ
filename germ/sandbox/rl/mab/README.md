Awesome—let’s build a clean, self-contained **multi-armed bandit** project that teaches the core ideas fast and gives you room to experiment rigorously.

Below is a compact but *production-ready* scaffold with three canonical algorithms—**ε-greedy**, **UCB1**, and **Thompson Sampling**—plus proper evaluation (reward, regret, confidence intervals), and knobs for stationarity vs. non-stationarity.

---

# Goals (what you’ll learn)

* Formalize bandits: actions $a\in\{1,\dots,K\}$, unknown means $\mu_a$, reward $r_t$.
* **Exploration vs. exploitation** via:

  * ε-greedy (value estimates + random exploration).
  * **UCB1** (optimism in face of uncertainty).
  * **Thompson Sampling** (Bayesian posterior sampling).
* **Metrics**: average reward, **cumulative regret** $R_T = \sum_{t=1}^T (\mu^\* - \mu_{a_t})$, selection counts, calibration.
* Experimental hygiene: multiple seeds, confidence intervals, non-stationary drifts.

---

# Project layout

```
bandits/
  ├─ bandit.py            # Environments (Bernoulli, Gaussian, drifting)
  ├─ agents.py            # ε-Greedy, UCB1, Thompson
  ├─ run.py               # Experiment loop, logging, plots
  ├─ metrics.py           # Regret, CI, tables
  └─ requirements.txt     # numpy, matplotlib
```

---

# Core formulas (keep these handy)

* **Sample mean update** for arm $a$:
  $\hat\mu_a \leftarrow \hat\mu_a + \frac{1}{n_a}(r - \hat\mu_a)$
* **ε-greedy**: with prob ε pick random arm; else pick $\arg\max_a \hat\mu_a$.
* **UCB1** (Auer et al. 2002): pick
  $a_t = \arg\max_a \left[\hat\mu_a + \sqrt{\frac{2\ln t}{n_a}}\right]$  (ensure $n_a>0$ first).
* **Thompson (Bernoulli)**: maintain Beta($\alpha_a,\beta_a$); sample $\tilde\mu_a \sim \text{Beta}(\alpha_a,\beta_a)$; pick $\arg\max_a \tilde\mu_a$; update $\alpha,\beta$ with successes/failures.

---

# Minimal, solid implementation (copy–paste runnable)

## `bandit.py`

```python
import numpy as np

class BernoulliBandit:
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
        self.best_mean = self.probs.max()

    def step(self, arm):
        p = self.probs[arm]
        r = self.rng.random() < p
        if self.drifting and self.drift_std > 0:
            self.probs += self.rng.normal(0.0, self.drift_std, size=self.K)
            self.probs = np.clip(self.probs, 0.0, 1.0)
            self.best_mean = self.probs.max()
        return float(r)

    def reset(self, probs=None):
        if probs is not None:
            self.probs = np.array(probs, dtype=float)
        self.best_mean = self.probs.max()
```

## `agents.py`

```python
import numpy as np

class AgentBase:
    def __init__(self, K, seed=None):
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.n = np.zeros(self.K, dtype=int)
        self.q = np.zeros(self.K, dtype=float)  # value estimates

    def select(self, t):
        raise NotImplementedError

    def update(self, arm, reward):
        self.n[arm] += 1
        # incremental mean
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]

class EpsilonGreedy(AgentBase):
    def __init__(self, K, epsilon=0.1, seed=None, optimistic_init=0.0):
        super().__init__(K, seed)
        self.epsilon = float(epsilon)
        self.optimistic_init = float(optimistic_init)
        self.q[:] = self.optimistic_init

    def select(self, t):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.K)
        return int(np.argmax(self.q))

class UCB1(AgentBase):
    def __init__(self, K, seed=None, c=2.0):
        super().__init__(K, seed)
        self.c = float(c)

    def select(self, t):
        # ensure each arm is pulled at least once
        for a in range(self.K):
            if self.n[a] == 0:
                return a
        ucb = self.q + np.sqrt(self.c * np.log(max(1, t)) / self.n)
        return int(np.argmax(ucb))

class ThompsonBernoulli(AgentBase):
    def __init__(self, K, seed=None, alpha0=1.0, beta0=1.0):
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        super().__init__(K, seed)

    def reset(self):
        super().reset()
        self.alpha = np.full(self.K, self.alpha0, dtype=float)
        self.beta = np.full(self.K, self.beta0, dtype=float)

    def select(self, t):
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        # keep q as empirical mean for logging; Beta posterior for decisions
        super().update(arm, reward)
        if reward > 0.5:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0
```

## `metrics.py`

```python
import numpy as np

def run_episode(env, agent, T):
    rewards = np.zeros(T, dtype=float)
    chosen = np.zeros(T, dtype=int)
    best_means = np.zeros(T, dtype=float)

    agent.reset()
    for t in range(1, T + 1):
        a = agent.select(t)
        r = env.step(a)
        agent.update(a, r)
        rewards[t-1] = r
        chosen[t-1] = a
        best_means[t-1] = env.best_mean  # handles drifting envs
    return rewards, chosen, best_means

def regret_curve(rewards, best_means):
    inst_regret = best_means - rewards
    return np.cumsum(inst_regret)

def mean_ci(x, axis=0, conf=0.95):
    # Normal approx; fine for large n. For small n, bootstrap instead.
    x = np.asarray(x)
    m = x.mean(axis=axis)
    s = x.std(axis=axis, ddof=1)
    n = x.shape[axis]
    from scipy.stats import norm
    z = norm.ppf(0.5 + conf/2.0)
    half = z * s / np.sqrt(n)
    return m, half
```

## `run.py`

```python
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from agents import EpsilonGreedy, UCB1, ThompsonBernoulli
from bandit import BernoulliBandit
from metrics import run_episode, regret_curve, mean_ci

def compare(arms=[0.1, 0.2, 0.35, 0.7], T=10000, runs=20, drifting=False, drift_std=0.0, seed=42):
    rng = np.random.default_rng(seed)

    algs = [
        ("ε-greedy(0.1)", lambda K: EpsilonGreedy(K, epsilon=0.1, optimistic_init=0.0, seed=rng.integers(1<<31))),
        ("UCB1(c=2)",     lambda K: UCB1(K, c=2.0, seed=rng.integers(1<<31))),
        ("Thompson",      lambda K: ThompsonBernoulli(K, seed=rng.integers(1<<31))),
    ]

    K = len(arms)
    all_rewards = {name: [] for name,_ in algs}
    all_regrets = {name: [] for name,_ in algs}

    for r in range(runs):
        env = BernoulliBandit(arms, drifting=drifting, drift_std=drift_std, seed=rng.integers(1<<31))
        for name, ctor in algs:
            agent = ctor(K)
            rewards, _, best = run_episode(env, agent, T)
            all_rewards[name].append(rewards)
            all_regrets[name].append(regret_curve(rewards, best))

    # Plot mean reward (rolling) and cumulative regret with 95% CI
    window = max(1, T // 200)
    def rolling_mean(x, w):
        if w <= 1: return x
        c = np.convolve(x, np.ones(w)/w, mode="same")
        return c

    plt.figure()
    for name in all_rewards:
        r = np.mean(all_rewards[name], axis=0)
        plt.plot(rolling_mean(r, window), label=name)
    plt.title("Rolling Average Reward")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    for name in all_regrets:
        r = np.stack(all_regrets[name], axis=0)  # (runs, T)
        mean = r.mean(axis=0)
        plt.plot(mean, label=name)
    plt.title("Cumulative Regret")
    plt.xlabel("Timestep")
    plt.ylabel("Regret")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Final numbers table
    rows = []
    for name in all_rewards:
        R = np.stack(all_rewards[name], axis=0)
        G = np.stack(all_regrets[name], axis=0)
        avg_reward = R.mean()
        final_regret = G[:,-1]
        m_reg, ci_reg = mean_ci(final_regret[:,None], axis=0)
        rows.append({
            "algorithm": name,
            "avg_reward": avg_reward,
            "final_regret_mean": float(m_reg),
            "final_regret_CI95_halfwidth": float(ci_reg),
        })
    df = pl.DataFrame(rows).sort("final_regret_mean")
    print(df)

if __name__ == "__main__":
    # Stationary example
    compare(arms=[0.1, 0.2, 0.35, 0.7], T=10000, runs=20, drifting=False)

    # Non-stationary example (uncomment to try)
    # compare(arms=[0.3, 0.31, 0.32, 0.33], T=20000, runs=20, drifting=True, drift_std=0.002)
```

`requirements.txt`

```
numpy
matplotlib
scipy
pandas
```

---

## How to use it (quick start)

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python run.py`

You’ll get:

* Rolling average reward plot.
* Cumulative regret plot.
* A small table comparing algorithms with 95% CIs.

---

## What results to expect (stationary Bernoulli)

* **Thompson** ≈ **UCB1** in final regret; both should beat ε-greedy for modest ε (0.1).
* ε-greedy with **optimistic initialization** (e.g., `optimistic_init=1.0` for Bernoulli) often improves early exploration, narrowing the gap.
* In **non-stationary** settings, UCB1’s log bonus becomes stale; fixed-ε greedy or discounted/Sliding-Window Thompson usually fares better.

---

## Extensions (after baseline)

* **Contextual bandits**: add feature vectors $x_t$ and use LinUCB or Thompson for linear reward models.
* **Non-stationarity**: exponential decay on counts/means or sliding windows; for Thompson, use a forgetting factor on $(\alpha,\beta)$.
* **Other reward models**: Gaussian arms (unknown mean, known/unknown variance) to see UCB-V, Bayesian Normal-Inverse-Gamma TS.
* **Hyperparameter sweeps**: scan ε, UCB constant $c$, Beta priors; use median-of-means for robustness.
* **Stat tests**: pairwise **paired bootstrap** on regret trajectories to assert significance.

---

## Opinions (and alternatives)

* **Start with Thompson Sampling**: it’s simple, performs strongly, and generalizes well—**my pick for default**.
  **Opposing view:** UCB1 has stronger frequentist guarantees and no priors; it’s fully deterministic given data.
* **Avoid annealing ε** (ε → 0) as your first tweak; it helps, but tuning schedules obscures the basic tradeoffs.
* **Prefer regret** over average reward when comparing algorithms across problem instances.

**Biases/uncertainties:** I’m biased toward Bayesian methods (TS) because they’re simple to code and empirically strong on Bernoulli arms. In small-T regimes with very close arm means, UCB’s behavior can be easier to reason about; your mileage may vary.

---

## Good prompts to iterate with me next

* “Run the project with 10 arms whose means are linearly spaced from 0.1 to 0.9; compare Thompson vs UCB1 over T=50k with 50 runs; report final regret with 95% CIs.”
* “Add exponential discounting with factor 0.99 to ε-greedy and compare on a drifting bandit with drift\_std=0.003.”
* “Switch to Gaussian rewards (σ=0.1) and implement UCB-V + Gaussian Thompson; compare.”

If you want, I can add **non-stationary variants** (discounted UCB, sliding-window TS) or **contextual LinUCB** in the same scaffold next.
