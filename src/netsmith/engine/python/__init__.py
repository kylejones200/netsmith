"""
Python backend: Reference implementations.

These mirror the Rust kernels, including the failure policy: an edge naming a
node that does not exist is an error, never a dropped edge.
"""

from .centrality import betweenness_python
from .clustering import clustering_python
from .communities import communities_python, louvain_python, modularity_python
from .components import components_python
from .degree import degree_python
from .pagerank import pagerank_python
from .paths import mean_shortest_path_python, shortest_paths_python

__all__ = [
    "degree_python",
    "betweenness_python",
    "pagerank_python",
    "communities_python",
    "louvain_python",
    "modularity_python",
    "clustering_python",
    "components_python",
    "shortest_paths_python",
    "mean_shortest_path_python",
]
