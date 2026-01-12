"""
Engine layer: Performance and execution.

Provides two backends: Pure Python plus Rust.
The API does not change.
"""

from .dispatch import (
    compute_degree,
    compute_pagerank,
    compute_communities,
    compute_clustering,
    compute_components,
    compute_shortest_paths,
)
from .contracts import EdgeList, GraphData

__all__ = [
    "compute_degree",
    "compute_pagerank",
    "compute_communities",
    "compute_clustering",
    "compute_components",
    "compute_shortest_paths",
    "EdgeList",
    "GraphData",
]

