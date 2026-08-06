"""
Core community detection: modularity, Louvain hooks, label propagation hooks.
"""

import logging
from typing import Dict, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .graph import Graph

logger = logging.getLogger(__name__)

CommunityBackend = Literal["auto", "rust", "python"]


def _resolve_backend(backend: CommunityBackend) -> str:
    """Pick the backend to run on, preferring the Rust kernel."""
    if backend not in ("auto", "rust", "python"):
        raise ValueError(f"unknown backend: {backend!r} (expected 'auto', 'rust' or 'python')")
    if backend != "auto":
        if backend == "rust":
            from ..engine.rust import _RUST_AVAILABLE

            if not _RUST_AVAILABLE:
                raise ImportError("Rust backend requested but netsmith_rs is not available")
        return backend

    from ..engine.rust import _RUST_AVAILABLE

    return "rust" if _RUST_AVAILABLE else "python"


def modularity(
    graph: Graph,
    communities: NDArray,
    weight: Optional[str] = None,
    backend: CommunityBackend = "auto",
) -> float:
    """
    Compute modularity score for community assignments.

    Parameters
    ----------
    graph : Graph
        Input graph
    communities : NDArray
        Array (n_nodes,) with community ID for each node.
        Nodes with the same ID are in the same community.
    weight : str, optional
        Edge weight attribute name (currently ignored; uses graph weights if available)
    backend : str, default "auto"
        "auto" (Rust if available, else Python), "rust", "python", or
        All backends use the same definition and agree to floating-point
        tolerance.

    Returns
    -------
    modularity : float
        Modularity score in range [-0.5, 1.0]. Higher values indicate
        stronger community structure. Values >0.3 typically indicate
        meaningful communities.

    Notes
    -----
    Modularity measures the quality of community assignments by comparing
    the fraction of edges within communities to the expected fraction in
    a random graph with the same degree sequence.
    """
    backend_name = _resolve_backend(backend)

    if backend_name in ("rust", "python"):
        edges = graph.to_edge_list()
        if backend_name == "rust":
            from ..engine.rust import modularity_rust

            return modularity_rust(edges, np.asarray(communities))

        from ..engine.python import modularity_python

        return modularity_python(edges, np.asarray(communities))


def louvain_hooks(
    graph: Graph,
    resolution: float = 1.0,
    seed: Optional[int] = None,
    backend: CommunityBackend = "auto",
) -> Dict:
    """
    Louvain community detection hooks.

    Parameters
    ----------
    graph : Graph
        Input graph. Directed graphs are treated as undirected; parallel and
        reciprocal edges are merged by summing their weights.
    resolution : float, default 1.0
        Resolution parameter. Higher values yield smaller communities.
    seed : int, optional
        Random seed for the node visit order. Each backend is reproducible for
        a given seed, but the same seed does not carry across backends.
    backend : str, default "auto"
        "auto" (Rust if available, else Python), "rust", or "python".

    Returns
    -------
    result : dict
        Dictionary with keys "communities" (array of community ids),
        "modularity", and "n_communities". The Rust and Python backends also
        report "n_levels", the number of aggregation levels performed.
    """
    backend_name = _resolve_backend(backend)

    if backend_name in ("rust", "python"):
        edges = graph.to_edge_list()
        if backend_name == "rust":
            from ..engine.rust import louvain_rust

            return louvain_rust(edges, resolution=resolution, seed=seed)

        from ..engine.python import louvain_python

        return louvain_python(edges, resolution=resolution, seed=seed)


def label_propagation_hooks(
    graph: Graph,
    seed: Optional[int] = None,
    max_iter: int = 100,
    backend: CommunityBackend = "auto",
) -> Dict:
    """
    Detect communities using asynchronous label propagation.

    Parameters
    ----------
    graph : Graph
        Input graph. Treated as undirected; parallel and reciprocal edges are
        merged by summing their weights.
    seed : int, optional
        Random seed for the visit order and tie-breaks. Each backend is
        reproducible for a given seed, but the same seed does not carry across
        backends.
    max_iter : int, default 100
        Cap on passes
    backend : str, default "auto"
        "auto" (Rust if available, else Python), "rust", or "python".

    Returns
    -------
    result : dict
        Dictionary containing:
        - "communities": NDArray[np.int64] (n_nodes,) with community IDs
        - "n_communities": int number of communities found

    Notes
    -----
    Label propagation is fast and parameter-free, but it optimizes nothing
    explicitly: on graphs without clear structure it can collapse everything
    into one community. Prefer `louvain_hooks` when you want a partition you
    can defend by its modularity.
    """
    from ..engine.dispatch import compute_label_propagation

    _resolve_backend(backend)
    return compute_label_propagation(
        graph.to_edge_list(), seed=seed, max_iter=max_iter, backend=backend
    )
