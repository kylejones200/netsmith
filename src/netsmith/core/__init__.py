"""
Core layer: Pure math, no I/O, no global state.

This layer contains the fundamental graph types, metrics, and algorithms.
"""

from .community import label_propagation_hooks, louvain_hooks, modularity
from .graph import Graph, GraphView
from .metrics import (
    assortativity,
    centrality,
    clustering,
    components,
    degree,
    k_core,
    strength,
)
from .nulls import null_models, permutation_tests
from .paths import reachability, shortest_paths, walk_metrics
from .stats import bootstrap, confidence_intervals, distributions

__all__ = [
    "Graph",
    "GraphView",
    "degree",
    "strength",
    "centrality",
    "assortativity",
    "clustering",
    "k_core",
    "components",
    "shortest_paths",
    "reachability",
    "walk_metrics",
    "modularity",
    "louvain_hooks",
    "label_propagation_hooks",
    "null_models",
    "permutation_tests",
    "distributions",
    "confidence_intervals",
    "bootstrap",
]
