"""
API metric functions: clustering, components.

These functions operate on Graph objects and delegate to Engine layer.
"""

from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..core.graph import Graph
from ..engine.dispatch import compute_betweenness, compute_clustering, compute_components
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


def betweenness(
    graph: Graph,
    node: Optional[int] = None,
    normalized: bool = True,
    weight: Optional[bool] = None,
    backend: Backend = "auto",
) -> Union[NDArray[np.float64], float]:
    """
    Compute betweenness centrality.

    Parameters
    ----------
    graph : Graph
        Input graph. Self-loops are ignored; parallel edges collapse to the
        lightest one. Directed graphs follow edges only from source to target.
    node : int, optional
        If provided, returns the score for this node only.
    normalized : bool, default True
        Divide by the number of ordered pairs excluding the node, giving
        scores in [0, 1]. When False, an undirected graph is halved instead,
        since each pair is swept from both endpoints.
    weight : bool, optional
        Read edge weights as shortest-path distances — a heavier edge is a
        longer step. Defaults to using weights when the graph has them.
        Weights must be strictly positive.
    backend : Backend, default "auto"
        Computation backend: "auto" (prefer Rust), "python", or "rust"

    Returns
    -------
    betweenness : NDArray[np.float64] or float
        Share of shortest paths between other pairs of nodes that run through
        each node. Endpoints are excluded and unreachable pairs contribute
        nothing.

    Raises
    ------
    ValidationError
        If node is out of range [0, graph.n_nodes)

    Notes
    -----
    Betweenness identifies brokers: nodes whose removal would lengthen or cut
    the routes between others. Computed with Brandes' algorithm, O(n·m) for
    unweighted graphs. Matches `networkx.betweenness_centrality`.
    """
    if node is not None:
        if not isinstance(node, (int, np.integer)):
            raise ValidationError(f"node must be integer, got {type(node)}")
        if node < 0 or node >= graph.n_nodes:
            raise ValidationError(f"node {node} is out of range [0, {graph.n_nodes})")

    edges = graph.to_edge_list()
    scores = compute_betweenness(edges, normalized=normalized, weight=weight, backend=backend)

    if node is not None:
        return float(scores[node])
    return scores


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
