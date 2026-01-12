"""
Core path algorithms: shortest paths, reachability, walk metrics.
"""

from typing import Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..exceptions import ValidationError
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
    # Validate source and target if provided
    if source is not None:
        if not isinstance(source, (int, np.integer)):
            raise ValidationError(f"source must be integer, got {type(source)}")
        if source < 0 or source >= graph.n_nodes:
            raise ValidationError(f"source {source} is out of range [0, {graph.n_nodes})")

    if target is not None:
        if not isinstance(target, (int, np.integer)):
            raise ValidationError(f"target must be integer, got {type(target)}")
        if target < 0 or target >= graph.n_nodes:
            raise ValidationError(f"target {target} is out of range [0, {graph.n_nodes})")

    from ..engine.dispatch import compute_shortest_paths

    edges = graph.to_edge_list()

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
    # Validate source node
    if not isinstance(source, (int, np.integer)):
        raise ValidationError(f"source must be integer, got {type(source)}")
    if source < 0 or source >= graph.n_nodes:
        raise ValidationError(f"source {source} is out of range [0, {graph.n_nodes})")

    from ..engine.dispatch import compute_shortest_paths

    edges = graph.to_edge_list()

    dist = compute_shortest_paths(edges, source=source, backend="auto")
    if isinstance(dist, dict):
        # This should not happen when source is specified
        # If it does, it indicates a backend implementation issue
        raise ValueError(
            f"Unexpected dict return from compute_shortest_paths with source={source}. "
            f"This indicates a backend implementation issue."
        )
    # Convert to boolean: reachable if distance is not max
    # Use the actual dtype's max value, not hardcoded int64.max
    max_val = np.iinfo(dist.dtype).max
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
    raise NotImplementedError(
        "walk_metrics is not yet implemented. "
        "This feature is planned for a future release."
    )
