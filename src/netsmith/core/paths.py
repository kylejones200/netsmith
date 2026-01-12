"""
Core path algorithms: walk metrics.

Note: shortest_paths and reachability have been moved to api.paths
to fix architecture violations (Core layer should not import from Engine).
This module is kept for backward compatibility and re-exports from API.
"""

from typing import Dict

# Re-export from API layer to maintain backward compatibility
from ..api.paths import reachability, shortest_paths  # noqa: F401
from .graph import Graph


def walk_metrics(graph: Graph, length: int = 1) -> Dict:
    """
    Compute random walk metrics.

    Parameters
    ----------
    graph : Graph
        Input graph
    length : int, default 1
        Walk length

    Returns
    -------
    metrics : dict
        Dictionary with walk metrics
    """
    raise NotImplementedError(
        "walk_metrics is not yet implemented. " "This feature is planned for a future release."
    )
