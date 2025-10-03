"""
metrics.py — experiment loop, regret computation, and quick confidence intervals.

What we compute
---------------
- `run_episode(env, agent, T)`:
    Runs a single length-T interaction and returns:
    * rewards[t]   = realized reward at step t
    * chosen[t]    = arm index chosen at step t (for diagnostics)
    * best_means[t]= the environment’s *current* best arm mean μ*_t
      (tracks drift if the environment is non-stationary)

- `regret_curve(rewards, best_means)`:
    Instantaneous regret at time t is μ*_t - r_t (with Bernoulli rewards).
    Cumulative regret is the running sum. This is the de facto metric in bandit literature.

- `mean_ci(x, conf=0.95)`:
    Simple normal-approximation confidence interval around the mean of `x`.
    Good enough for quick comparisons across multiple random seeds/runs.

Why regret?
-----------
Average reward can be misleading across different problem instances. Regret measures how
far we are from an oracle that always plays the *current* best arm, which is comparable
across instances and matches the theory (logarithmic in T for “nice” problems).

References
----------
- Lattimore & Szepesvári, *Bandit Algorithms* (regret as central objective)
  https://tor-lattimore.com/downloads/book/book.pdf
- Sutton & Barto, Ch. 2 (incremental implementation; non-stationary tracking)
  https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
"""
import numpy as np

def run_episode(env, agent, T):
    """
    Run a single episode of length T and collect trajectories.

    Parameters
    ----------
    env : object with .step(arm) -> reward and attribute .best_mean
        Typically `BernoulliBandit` from bandit.py. We only rely on this minimal API.
    agent : object with .select(t) -> arm and .update(arm, reward)
        Any agent from agents.py or a compatible implementation.
    T : int
        Number of interaction steps.

    Returns
    -------
    rewards : np.ndarray shape (T,)
        Realized rewards r_1, ..., r_T.
    chosen : np.ndarray shape (T,)
        Arm indices a_1, ..., a_T chosen by the agent (useful for diagnostics/plots).
    best_means : np.ndarray shape (T,)
        μ*_t = max_a μ_a at each step (after any environment drift). Used to compute regret.

    Implementation notes
    --------------------
    - We reset the agent at episode start to ensure clean state (counts, estimates, priors).
    - Time index t is 1-based in the agent API to align with common UCB/analysis notation.
    """
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
    """
    Compute cumulative regret given realized rewards and the per-step best mean.

    Parameters
    ----------
    rewards : np.ndarray shape (T,)
    best_means : np.ndarray shape (T,)
        For stationary bandits, this is constant; for drifting bandits, it changes over time.

    Returns
    -------
    cumulative_regret : np.ndarray shape (T,)
        Cumulative sum of (μ*_t - r_t).

    Notes
    -----
    If you generalize beyond Bernoulli rewards, the definition is unchanged:
    regret is always (best achievable expected reward at t) minus (your realized reward).
    """
    inst_regret = best_means - rewards
    return np.cumsum(inst_regret)

def mean_ci(x, axis=0, conf=0.95):
    """
    Compute a (symmetric) normal-approximation CI for the mean of `x`.

    Parameters
    ----------
    x : array-like
        If you pass a matrix of shape (runs, time) and set axis=0, you get a CI over runs
        for each time step; here we usually pass a 1D array (final regrets across runs).
    axis : int
        Axis along which to average.
    conf : float
        Confidence level (e.g., 0.95 for 95% CI).

    Returns
    -------
    mean : np.ndarray
    halfwidth : np.ndarray
        So your CI is [mean - halfwidth, mean + halfwidth].

    Caveats
    -------
    - This assumes approximate normality by CLT; for small sample sizes or skewed
      distributions, prefer a bootstrap CI (percentile or BCa).
    - We use the *sample* standard deviation (ddof=1).

    Implementation detail
    ---------------------
    We import `scipy.stats.norm` here to keep dependencies local; see requirements.txt.
    """
    # Normal approx; fine for large n. For small n, bootstrap instead.
    x = np.asarray(x)
    m = x.mean(axis=axis)
    s = x.std(axis=axis, ddof=1)
    n = x.shape[axis]
    from scipy.stats import norm
    z = norm.ppf(0.5 + conf/2.0)
    half = z * s / np.sqrt(n)
    return m, half
