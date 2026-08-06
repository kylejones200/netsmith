"""
Python backend: Reference implementations.

These mirror the Rust kernels, including the failure policy: an edge naming a
node that does not exist is an error, never a dropped edge.
"""

from .centrality import betweenness_python
from .clustering import clustering_python
from .communities import (
    communities_python,
    label_propagation_python,
    louvain_python,
    modularity_python,
)
from .components import components_python
from .degree import degree_python
from .kcore import core_numbers_python
from .nulls import (
    configuration_model_python,
    degree_preserving_rewire_python,
    erdos_renyi_python,
)
from .pagerank import pagerank_python
from .paths import (
    mean_shortest_path_python,
    shortest_paths_multi_python,
    shortest_paths_python,
)

__all__ = [
    "degree_python",
    "betweenness_python",
    "pagerank_python",
    "communities_python",
    "louvain_python",
    "label_propagation_python",
    "core_numbers_python",
    "configuration_model_python",
    "degree_preserving_rewire_python",
    "erdos_renyi_python",
    "modularity_python",
    "clustering_python",
    "components_python",
    "shortest_paths_python",
    "shortest_paths_multi_python",
    "mean_shortest_path_python",
]
