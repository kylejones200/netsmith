"""
API path functions: shortest paths, reachability.

These functions operate on Graph objects and delegate to Engine layer.
"""

from typing import Dict, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..core.graph import Graph
from ..engine.dispatch import compute_shortest_paths, compute_shortest_paths_multi
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
    Compute shortest paths in a graph.

    Parameters
    ----------
    graph : Graph
        Input graph
    source : int, optional
        Source node index. If None, computes all-pairs shortest paths.
    target : int, optional
        Target node index. If specified with source, returns shortest path
        between source and target.
    weight : str, optional
        Edge weight attribute name (currently not fully supported)
    backend : Backend, default "auto"
        Computation backend: "auto" (prefer Rust), "python", or "rust"

    Returns
    -------
    result : NDArray[np.int64] or Dict
        If source is specified: distance array (n_nodes,) with distances from source.
        If source is None: dictionary with path information.

    Raises
    ------
    ValidationError
        If source or target are out of range [0, graph.n_nodes)

    Notes
    -----
    Unreachable nodes have distance equal to the maximum value for the array dtype.
    For all-pairs computation (source=None), the return format may vary by backend.
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


def shortest_paths_multi(graph: Graph, sources, backend: Backend = "auto") -> NDArray[np.int64]:
    """
    Compute hop distances from several sources at once.

    Calling `shortest_paths` in a loop rebuilds the adjacency list every time,
    and that setup dominates the cost when the graph is fixed. This builds it
    once; the Rust backend also sweeps the sources in parallel.

    Parameters
    ----------
    graph : Graph
        Input graph
    sources : sequence of int
        Node indices to search from
    backend : Backend, default "auto"
        Computation backend: "auto" (prefer Rust), "python", or "rust"

    Returns
    -------
    distances : NDArray[np.int64], shape (len(sources), n_nodes)
        Row i holds the hop distances from `sources[i]`. Unreachable nodes hold
        the maximum int64.

    Raises
    ------
    ValidationError
        If any source is out of range [0, graph.n_nodes)

    Examples
    --------
    >>> G = Graph(edges=[(0, 1), (1, 2)], n_nodes=3)
    >>> shortest_paths_multi(G, [0, 2]).shape
    (2, 3)
    """
    sources = np.asarray(sources, dtype=np.int64).ravel()
    out_of_range = (sources < 0) | (sources >= graph.n_nodes)
    if out_of_range.any():
        bad = int(sources[np.flatnonzero(out_of_range)[0]])
        raise ValidationError(f"source {bad} is out of range [0, {graph.n_nodes})")

    return compute_shortest_paths_multi(graph.to_edge_list(), sources, backend=backend)


def reachability(graph: Graph, source: int, backend: Backend = "auto") -> NDArray[np.bool_]:
    """
    Compute reachable nodes from a source node.

    Parameters
    ----------
    graph : Graph
        Input graph
    source : int
        Source node index
    backend : Backend, default "auto"
        Computation backend: "auto" (prefer Rust), "python", or "rust"

    Returns
    -------
    reachable : NDArray[np.bool_]
        Boolean array (n_nodes,) where reachable[i] is True if node i is
        reachable from source, False otherwise. The source node itself is
        always reachable.

    Raises
    ------
    ValidationError
        If source is out of range [0, graph.n_nodes)
    BackendError
        If backend returns unexpected format (implementation issue)
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
