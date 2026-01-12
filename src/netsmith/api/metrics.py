"""
API metric functions: clustering, components.

These functions operate on Graph objects and delegate to Engine layer.
"""

from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..core.graph import Graph
from ..engine.contracts import EdgeList
from ..engine.dispatch import compute_clustering, compute_components
from ..exceptions import ValidationError

Backend = Literal["auto", "python", "rust"]


def clustering(graph: Graph, node: Optional[int] = None, backend: Backend = "auto") -> Union[NDArray, float]:
    """
    Compute clustering coefficient.

    Parameters
    ----------
    graph : Graph
        Input graph
    node : int, optional
        If provided, return clustering for this node only
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    clustering : array or float
        Clustering coefficients or single value
    """
    if node is not None:
        # Validate node index
        if not isinstance(node, (int, np.integer)):
            raise ValidationError(f"node must be integer, got {type(node)}")
        if node < 0 or node >= graph.n_nodes:
            raise ValidationError(f"node {node} is out of range [0, {graph.n_nodes})")

    edges = graph.to_edge_list()

    clustering_values = compute_clustering(edges, backend=backend)

    if node is not None:
        return float(clustering_values[node])
    return clustering_values


def components(graph: Graph, return_labels: bool = True, backend: Backend = "auto") -> Union[int, NDArray]:
    """
    Compute connected components.

    Parameters
    ----------
    graph : Graph
        Input graph
    return_labels : bool, default True
        If True, return component labels for each node.
        If False, return number of components.
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    labels : array (n_nodes,) or int
        Component labels or number of components
    """
    edges = graph.to_edge_list()

    n_components, labels = compute_components(edges, backend=backend)

    if return_labels:
        return labels
    return n_components

