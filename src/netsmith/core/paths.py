"""
Core path algorithms: shortest paths, reachability, walk metrics.
"""

from typing import Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .graph import Graph


def shortest_paths(
    graph: Graph,
    source: Optional[int] = None,
    target: Optional[int] = None,
    weight: Optional[str] = None,
) -> Union[NDArray[np.int64], Dict]:
    """
    Compute shortest paths.

    Parameters
    ----------
    graph : Graph
        Input graph
    source : int, optional
        Source node (if None, compute all pairs)
    target : int, optional
        Target node
    weight : str, optional
        Edge weight attribute (if graph is weighted)

    Returns
    -------
    result : array or dict
        Distance array or path information
    """
    from ..engine.contracts import EdgeList
    from ..engine.dispatch import compute_shortest_paths

    src, dst, w = graph.edges_coo()
    edges = EdgeList(u=src, v=dst, w=w, directed=graph.directed, n_nodes=graph.n_nodes)

    return compute_shortest_paths(
        edges, source=source, target=target, weight=weight, backend="auto"
    )


def reachability(graph: Graph, source: int) -> NDArray:
    """
    Compute reachable nodes from source.

    Parameters
    ----------
    graph : Graph
        Input graph
    source : int
        Source node

    Returns
    -------
    reachable : array
        Boolean array indicating reachable nodes
    """
    from ..engine.contracts import EdgeList
    from ..engine.dispatch import compute_shortest_paths

    src, dst, w = graph.edges_coo()
    edges = EdgeList(u=src, v=dst, w=w, directed=graph.directed, n_nodes=graph.n_nodes)

    dist = compute_shortest_paths(edges, source=source, backend="auto")
    if isinstance(dist, dict):
        # Fallback if dict returned
        return np.ones(graph.n_nodes, dtype=bool)
    # Convert to boolean: reachable if distance is not max
    max_val = np.iinfo(np.int64).max
    return (dist != max_val).astype(bool)


def walk_metrics(graph: Graph, length: int = 1) -> Dict:
    """
    Compute random walk metrics.

    Parameters
    ----------
    graph : Graph
        Input graph
    length : int, default 1
        Walk length

    Returns
    -------
    metrics : dict
        Dictionary with walk metrics
    """
    # Placeholder - full implementation in engine layer
    return {}
