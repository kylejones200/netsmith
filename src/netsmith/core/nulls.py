"""
Core null models and permutation tests.
"""

from typing import Optional, Dict, Callable
import numpy as np
from numpy.typing import NDArray

from .graph import Graph


def null_models(
    graph: Graph,
    method: str = "configuration",
    n_samples: int = 100,
    seed: Optional[int] = None
) -> Dict:
    """
    Generate null model graphs.
    
    Parameters
    ----------
    graph : Graph
        Input graph
    method : str, default "configuration"
        Null model method: "configuration", "erdos_renyi", "degree_preserving"
    n_samples : int, default 100
        Number of null model samples
    seed : int, optional
        Random seed
    
    Returns
    -------
    result : dict
        Dictionary with null model graphs
    """
    # Placeholder - full implementation in engine layer
    return {"graphs": []}


def permutation_tests(
    graph: Graph,
    statistic: Callable,
    n_permutations: int = 1000,
    seed: Optional[int] = None
) -> Dict:
    """
    Permutation test for graph statistics.
    
    Parameters
    ----------
    graph : Graph
        Input graph
    statistic : callable
        Function that computes a statistic from a graph
    n_permutations : int, default 1000
        Number of permutations
    seed : int, optional
        Random seed
    
    Returns
    -------
    result : dict
        Dictionary with test results
    """
    # Placeholder - full implementation in engine layer
    return {"p_value": 1.0, "statistic": 0.0}

