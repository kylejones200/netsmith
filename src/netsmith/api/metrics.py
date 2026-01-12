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


def clustering(
    graph: Graph, node: Optional[int] = None, backend: Backend = "auto"
) -> Union[NDArray[np.float64], float]:
    """
    Compute clustering coefficient (transitivity).

    Parameters
    ----------
    graph : Graph
        Input graph
    node : int, optional
        If provided, returns clustering coefficient for this node only.
        If None, returns clustering coefficients for all nodes.
    backend : Backend, default "auto"
        Computation backend: "auto" (prefer Rust), "python", or "rust"

    Returns
    -------
    clustering : NDArray[np.float64] or float
        If node is None: array (n_nodes,) with clustering coefficient for each node.
        If node is specified: float with clustering coefficient for that node.
        Values range from 0.0 (no triangles) to 1.0 (complete clustering).

    Raises
    ------
    ValidationError
        If node is out of range [0, graph.n_nodes)

    Notes
    -----
    Clustering coefficient measures the fraction of triangles around a node.
    For a node i: C_i = (number of triangles) / (number of possible triangles).
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


def components(
    graph: Graph, return_labels: bool = True, backend: Backend = "auto"
) -> Union[int, NDArray[np.int64]]:
    """
    Compute connected components.

    Parameters
    ----------
    graph : Graph
        Input graph
    return_labels : bool, default True
        If True, returns component label array for each node.
        If False, returns only the number of connected components.
    backend : Backend, default "auto"
        Computation backend: "auto" (prefer Rust), "python", or "rust"

    Returns
    -------
    labels : NDArray[np.int64] or int
        If return_labels=True: array (n_nodes,) with component ID for each node.
        Nodes in the same component have the same label.
        If return_labels=False: integer count of connected components.

    Notes
    -----
    For undirected graphs, finds all connected components.
    For directed graphs, finds weakly connected components.
    Component labels are integers starting from 0.
    """
    edges = graph.to_edge_list()

    n_components, labels = compute_components(edges, backend=backend)

    if return_labels:
        return labels
    return n_components

