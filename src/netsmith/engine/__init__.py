"""
Engine layer: Performance and execution.

Provides two backends: Pure Python plus Rust.
The API does not change.
"""

from .contracts import EdgeList, GraphData
from .dispatch import (
    compute_clustering,
    compute_communities,
    compute_components,
    compute_degree,
    compute_pagerank,
    compute_shortest_paths,
)

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
