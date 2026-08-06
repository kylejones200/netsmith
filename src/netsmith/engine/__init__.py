"""
Engine layer: Performance and execution.

Provides two backends: Pure Python plus Rust.
The API does not change.
"""

from .contracts import EdgeList, GraphData
from .dispatch import (
    compute_betweenness,
    compute_clustering,
    compute_communities,
    compute_components,
    compute_core_numbers,
    compute_degree,
    compute_label_propagation,
    compute_pagerank,
    compute_shortest_paths,
    compute_shortest_paths_multi,
)

__all__ = [
    "compute_degree",
    "compute_pagerank",
    "compute_betweenness",
    "compute_communities",
    "compute_core_numbers",
    "compute_label_propagation",
    "compute_clustering",
    "compute_components",
    "compute_shortest_paths",
    "compute_shortest_paths_multi",
    "EdgeList",
    "GraphData",
]
