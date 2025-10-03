"""
run.py — experiment harness for comparing bandit agents on a chosen environment.

This script wires together:
  - the environment (BernoulliBandit),
  - the agents (ε-Greedy, UCB1, Thompson Sampling),
  - the experiment loop (multiple runs/seeds),
  - plotting (rolling average reward and cumulative regret),
  - and a small printable table of final comparisons with 95% confidence intervals.

WHY THIS FILE EXISTS
--------------------
Separating "experiment plumbing" from the algorithm code is a best practice:
  • You can swap environments/agents without touching plotting or metrics.
  • You get reproducible multi-run evaluations (critical for stochastic algorithms).
  • Keeps algorithm files minimal and easier to read/extend.

WHAT YOU CAN TWEAK FIRST
------------------------
  • `arms`: true means for each arm (stationary case) — e.g., [0.1, 0.2, 0.35, 0.7]
  • `T`:    horizon (number of pulls/steps)
  • `runs`: number of independent trials (used to compute variability/CIs)
  • `drifting` & `drift_std`: turn on non-stationarity and control its speed

PLOTS YOU'LL SEE
----------------
  • Rolling Average Reward: short-term performance (smoothed).
  • Cumulative Regret: long-term gap to an oracle that always picks the current best arm.

KEY REFERENCES (match the codebase)
-----------------------------------
  • Sutton & Barto (multi-armed bandits, incremental means):
        https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
  • Auer, Cesa-Bianchi, Fischer (UCB1 + regret bounds):
        https://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Auer%2Bal%3A2002.pdf
  • Russo et al. (Thompson Sampling tutorial):
        https://arxiv.org/abs/1707.02038
  • Lattimore & Szepesvári (textbook depth on regret & algorithms):
        https://tor-lattimore.com/downloads/book/book.pdf
"""
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from germ.sandbox.rl.mab.bandit import BernoulliBandit
from germ.sandbox.rl.mab.agents import EpsilonGreedy, UCB1, ThompsonBernoulli
from germ.sandbox.rl.mab.metrics import run_episode, regret_curve, mean_ci

def compare(arms=(0.1, 0.2, 0.35, 0.7), T=10000, runs=20, drifting=False, drift_std=0.0, seed=42):
    """
    Run a multi-run comparison across algorithms on a K-armed bandit.

    Parameters
    ----------
    arms : list[float]
        True success probabilities μ for each arm (stationary means if drifting=False).
        Choose them to create a meaningful gap between best/second-best arms; small gaps
        make exploration harder and surface algorithmic differences.
    T : int
        Horizon (number of time steps). Larger T highlights asymptotic behavior such as
        logarithmic regret for UCB1 on “nice” instances.
    runs : int
        Number of independent trials with different RNG seeds. We average across runs and
        compute confidence intervals so that conclusions aren’t due to luck.
    drifting : bool
        If True, the environment’s means evolve with Gaussian noise each step
        (see `BernoulliBandit` for details). This tests non-stationary robustness.
    drift_std : float
        Standard deviation for the per-step Gaussian drift of the means.
        Typical range: 0.001–0.01. Higher = faster non-stationarity.
    seed : int
        Master seed for the *experiment harness*. We derive per-run/per-agent seeds from this
        master seed to ensure reproducibility while keeping streams independent.

    Returns
    -------
    None (side effects: plots to screen and prints a summary table)

    DESIGN NOTES
    ------------
    • Seeding strategy:
        We create a single RNG (np.random.default_rng(seed)) and *draw* child seeds for
        each environment and agent instantiation. This preserves independence while
        allowing the entire experiment to be reproduced from one number.
    • Rolling average reward:
        Smoothing short-term reward with a window (T/200 by default) produces readable
        curves without hiding trends. You can change the window to trade smoothness vs. detail.
    • Regret:
        We plot *mean cumulative regret* across runs. Lower is better; curves that
        grow slowly (sub-linearly) are preferable.
    • Confidence intervals:
        The printed table uses a simple normal-approximation CI on final regret, which is
        reasonable for moderate runs; for few runs or skewed data, consider bootstrap CIs.
        (See metrics.mean_ci docstring.)
    """

    # Master RNG for the harness; never used directly for coin flips, only to sample
    # per-run/per-agent seeds so that streams are independent and reproducible.
    rng = np.random.default_rng(seed)

    # Define the algorithms we want to compare.
    # We pass in *constructors* that accept K so we can recreate agents cleanly for each run.
    algs = [
        ("ε-greedy(0.1)", lambda K: EpsilonGreedy(K, epsilon=0.1, optimistic_init=0.0, seed=rng.integers(1<<31))),
        ("UCB1(c=2)",     lambda K: UCB1(K, c=2.0, seed=rng.integers(1<<31))),
        ("Thompson",      lambda K: ThompsonBernoulli(K, seed=rng.integers(1<<31))),
    ]

    K = len(arms)

    # Storage for trajectories across runs
    # We keep raw rewards and regrets so we can compute means/quantiles later if needed.
    all_rewards = {name: [] for name,_ in algs}
    all_regrets = {name: [] for name,_ in algs}

    # ---- Core experiment loop -------------------------------------------------
    for r in range(runs):
        # New environment instance per run with its own seed
        env = BernoulliBandit(arms, drifting=drifting, drift_std=drift_std, seed=rng.integers(1<<31))
        for name, ctor in algs:
            agent = ctor(K)  # fresh agent state per run (counts, priors, etc.)
            rewards, _, best = run_episode(env, agent, T)
            all_rewards[name].append(rewards)
            all_regrets[name].append(regret_curve(rewards, best))
    # ---------------------------------------------------------------------------

    # ---- Plot 1: Rolling Average Reward --------------------------------------
    # Plot mean reward (rolling) and cumulative regret with 95% CI
    # Rationale: raw reward is very spiky (Bernoulli 0/1); smoothing makes trends legible.

    window = max(1, T // 200)
    def rolling_mean(x, w):
        # Simple centered moving average via convolution.
        # For exploration studies, you might prefer a causal window (np.convolve with 'valid'),
        # but here 'same' keeps the vector length aligned with time for easy plotting.
        if w <= 1: return x
        c = np.convolve(x, np.ones(w)/w, mode="same")
        return c

    plt.figure()
    for name in all_rewards:
        # Mean over runs at each time step → a representative learning curve
        r = np.mean(all_rewards[name], axis=0)
        plt.plot(rolling_mean(r, window), label=name)
    plt.title("Rolling Average Reward")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot 2: Cumulative Regret -------------------------------------------
    # Rationale: the central objective in bandit theory. Lower is better.
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

    # ---- Summary table --------------------------------------------------------
    # We compute average reward (scalar) across all steps and runs,
    # plus the distribution of final regrets across runs with a 95% CI.    rows = []
    rows = []
    for name in all_rewards:
        R = np.stack(all_rewards[name], axis=0)  # (runs, T)
        G = np.stack(all_regrets[name], axis=0)  # (runs, T)
        avg_reward = R.mean()  # scalar: average of everything
        final_regret = G[:,-1]  # (runs,)

        # mean_ci returns (mean, halfwidth); we pass a column vector to reuse the API
        m_reg, ci_reg = mean_ci(final_regret[:,None], axis=0)
        rows.append({
            "algorithm": name,
            "avg_reward": avg_reward,
            "final_regret_mean": float(m_reg),
            "final_regret_CI95_halfwidth": float(ci_reg),
            # Optional extras you might log for dashboards:
            # "median_final_regret": float(np.median(final_regret)),
            # "best_arm_mean": float(np.max(arms) if not drifting else np.nan),
        })
    df = pl.DataFrame(rows).sort("final_regret_mean")
    print(df)
    # If you plan to automate sweeps, consider returning `df` and the raw dicts:
    # return df, all_rewards, all_regrets


if __name__ == "__main__":
    # Example 1: Stationary bandit
    # RATIONALE: canonical sanity-check. Thompson and UCB1 should both beat ε-greedy(0.1)
    # in final regret on many instances; relative gaps depend on arm gaps and horizon T.
    #compare(arms=[0.1, 0.2, 0.35, 0.7], T=10000, runs=20, drifting=False)

    # Example 2: Non-stationary bandit
    # RATIONALE: tight means + drift test robustness. Plain UCB1 may lag because its
    # confidence bonuses assume stationarity; fixed-ε greedy or discounted/windowed TS
    # tend to handle drift better. See:
    #   • Sutton & Barto §2.6 (non-stationary tracking via constant step-size)
    #   • Garivier & Moulines (2011) for discounted UCB (advanced reading):
    #       https://jmlr.org/papers/volume12/garivier11a/garivier11a.pdf
    compare(arms=[0.3, 0.31, 0.32, 0.33], T=20000, runs=20, drifting=True, drift_std=0.002)
