"""
Core layer: Pure math, no I/O, no global state.

This layer contains the fundamental graph types, metrics, and algorithms.
"""

from .graph import Graph, GraphView
from .metrics import (
    degree,
    strength,
    centrality,
    assortativity,
    clustering,
    k_core,
    components,
)
from .paths import shortest_paths, reachability, walk_metrics
from .community import modularity, louvain_hooks, label_propagation_hooks
from .nulls import null_models, permutation_tests
from .stats import distributions, confidence_intervals, bootstrap

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

