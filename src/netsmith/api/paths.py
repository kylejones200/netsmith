"""
API path functions: shortest paths, reachability.

These functions operate on Graph objects and delegate to Engine layer.
"""

from typing import Dict, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..core.graph import Graph
from ..engine.contracts import EdgeList
from ..engine.dispatch import compute_shortest_paths
from ..exceptions import BackendError, ValidationError

Backend = Literal["auto", "python", "rust"]


def shortest_paths(
    graph: Graph,
    source: Optional[int] = None,
    target: Optional[int] = None,
    weight: Optional[str] = None,
    backend: Backend = "auto",
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
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

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

    edges = graph.to_edge_list()

    return compute_shortest_paths(
        edges, source=source, target=target, weight=weight, backend=backend
    )


def reachability(graph: Graph, source: int, backend: Backend = "auto") -> NDArray:
    """
    Compute reachable nodes from source.

    Parameters
    ----------
    graph : Graph
        Input graph
    source : int
        Source node
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

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

    edges = graph.to_edge_list()

    dist = compute_shortest_paths(edges, source=source, backend=backend)
    if isinstance(dist, dict):
        # This should not happen when source is specified
        # If it does, it indicates a backend implementation issue
        raise BackendError(
            f"Unexpected dict return from compute_shortest_paths with source={source}. "
            f"This indicates a backend implementation issue."
        )
    # Convert to boolean: reachable if distance is not max
    # Use the actual dtype's max value, not hardcoded int64.max
    max_val = np.iinfo(dist.dtype).max
    return (dist != max_val).astype(bool)

