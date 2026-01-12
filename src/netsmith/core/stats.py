"""
Core statistical functions: distributions, confidence intervals, bootstrap.
"""

from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def distributions(data: NDArray, method: str = "empirical") -> dict:
    """
    Estimate distributions from data.

    Parameters
    ----------
    data : array
        Input data
    method : str, default "empirical"
        Distribution estimation method

    Returns
    -------
    result : dict
        Dictionary with distribution parameters
    """
    # Placeholder - full implementation in engine layer
    return {}


def confidence_intervals(
    data: NDArray, alpha: float = 0.05, method: str = "normal"
) -> Tuple[float, float]:
    """
    Compute confidence intervals.

    Parameters
    ----------
    data : array
        Input data
    alpha : float, default 0.05
        Significance level
    method : str, default "normal"
        Method: "normal", "bootstrap", "percentile"

    Returns
    -------
    ci : tuple
        (lower, upper) confidence interval bounds
    """
    # Placeholder - full implementation in engine layer
    mean = np.mean(data)
    std = np.std(data)
    try:
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
    except ImportError:
        # Fallback: approximate with numpy using standard normal approximation
        # For alpha=0.05: z ≈ 1.96, for alpha=0.01: z ≈ 2.576, etc.
        # Good approximation for most use cases
        z = 1.96 if alpha == 0.05 else 2.576 if alpha == 0.01 else 1.645 if alpha == 0.10 else 1.96
    return (mean - z * std, mean + z * std)


def bootstrap(
    data: NDArray,
    statistic: Callable,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> dict:
    """
    Bootstrap resampling.

    Parameters
    ----------
    data : array
        Input data
    statistic : callable
        Function to compute statistic
    n_bootstrap : int, default 1000
        Number of bootstrap samples
    seed : int, optional
        Random seed
    alpha : float, default 0.05
        Significance level for confidence interval

    Returns
    -------
    result : dict
        Dictionary with bootstrap results
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)

    # Compute observed statistic
    observed_stat = float(statistic(data))

    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.integers(0, len(data), size=len(data))
        bootstrap_sample = data[indices]
        bootstrap_stat = float(statistic(bootstrap_sample))
        bootstrap_stats.append(bootstrap_stat)

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute confidence interval (percentile method)
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return {
        "statistic": observed_stat,
        "bootstrap_mean": float(np.mean(bootstrap_stats)),
        "bootstrap_std": float(np.std(bootstrap_stats)),
        "ci": (ci_lower, ci_upper),
        "n_bootstrap": n_bootstrap,
    }
