"""
Core community detection: modularity, Louvain hooks, label propagation hooks.
"""

import logging
from typing import Dict, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .graph import Graph

logger = logging.getLogger(__name__)

CommunityBackend = Literal["auto", "rust", "python", "networkx"]


def _resolve_backend(backend: CommunityBackend) -> str:
    """Pick the backend to run on, preferring the Rust kernel."""
    if backend not in ("auto", "rust", "python", "networkx"):
        raise ValueError(
            f"unknown backend: {backend!r} (expected 'auto', 'rust', 'python' or 'networkx')"
        )
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
        "networkx". All backends use the same definition and agree to
        floating-point tolerance.

    Returns
    -------
    modularity : float
        Modularity score in range [-0.5, 1.0]. Higher values indicate
        stronger community structure. Values >0.3 typically indicate
        meaningful communities.

    Raises
    ------
    ImportError
        If the "networkx" backend is selected and NetworkX is not installed

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

    # Convert to NetworkX for modularity computation
    try:
        import networkx  # noqa: F401
        from networkx.algorithms import community
    except ImportError:
        raise ImportError(
            "networkx is required for modularity computation. Install with: pip install networkx"
        )

    nx_graph = graph.as_networkx()

    # Convert communities array to list of sets
    n_communities = int(np.max(communities) + 1)
    community_sets = [set() for _ in range(n_communities)]
    for node, comm_id in enumerate(communities):
        community_sets[int(comm_id)].add(node)

    # Compute modularity
    if graph.weighted:
        modularity_score = community.modularity(nx_graph, community_sets, weight="weight")
    else:
        modularity_score = community.modularity(nx_graph, community_sets)

    return float(modularity_score)


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
        "auto" (Rust if available, else Python), "rust", "python", or
        "networkx".

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

    # Convert to NetworkX
    try:
        import networkx  # noqa: F401
        from networkx.algorithms import community
    except ImportError:
        raise ImportError(
            "networkx is required for Louvain community detection. "  # noqa: E501
            "Install with: pip install networkx"
        )

    nx_graph = graph.as_networkx()

    # Convert to undirected for community detection
    if nx_graph.is_directed():
        nx_graph = nx_graph.to_undirected()

    # Detect communities using Louvain
    try:
        communities_generator = community.louvain_communities(
            nx_graph,
            weight="weight" if graph.weighted else None,
            resolution=resolution,
            seed=seed,
        )
        communities = list(communities_generator)
    except AttributeError:
        # Fallback for older NetworkX versions
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(
                nx_graph,
                weight="weight" if graph.weighted else None,
                random_state=seed,
            )
            # Convert partition dict to list of sets
            n_communities = max(partition.values()) + 1
            communities = [set() for _ in range(n_communities)]
            for node, comm_id in partition.items():
                communities[comm_id].add(node)
        except ImportError:
            raise ImportError(
                "python-louvain is required for Louvain community detection. "  # noqa: E501
                "Install with: pip install python-louvain"
            )

    # Compute modularity
    if graph.weighted:
        modularity_score = community.modularity(nx_graph, communities, weight="weight")
    else:
        modularity_score = community.modularity(nx_graph, communities)

    # Convert communities to array
    community_array = np.zeros(graph.n_nodes, dtype=np.int64)
    for comm_id, comm_set in enumerate(communities):
        for node in comm_set:
            community_array[node] = comm_id

    return {
        "communities": community_array,
        "modularity": float(modularity_score),
        "n_communities": len(communities),
    }


def label_propagation_hooks(graph: Graph, seed: Optional[int] = None) -> Dict:
    """
    Detect communities using asynchronous label propagation.

    Parameters
    ----------
    graph : Graph
        Input graph (converted to undirected for community detection)
    seed : int, optional
        Random seed for reproducibility (algorithm has stochastic elements)

    Returns
    -------
    result : dict
        Dictionary containing:
        - "communities": NDArray[np.int64] (n_nodes,) with community IDs
        - "n_communities": int number of communities found

    Raises
    ------
    ImportError
        If NetworkX is not installed (required for label propagation)

    Notes
    -----
    Label propagation is a fast, local algorithm. Nodes iteratively adopt
    the label most common among their neighbors. The algorithm converges
    when labels stabilize. Works well for graphs with clear community structure.
    """
    # Convert to NetworkX
    try:
        import networkx  # noqa: F401
        from networkx.algorithms import community
    except ImportError:
        raise ImportError(
            "networkx is required for label propagation. Install with: pip install networkx"
        )

    nx_graph = graph.as_networkx()

    # Convert to undirected
    if nx_graph.is_directed():
        nx_graph = nx_graph.to_undirected()

    # Detect communities using label propagation
    communities_generator = community.asyn_lpa_communities(
        nx_graph, weight="weight" if graph.weighted else None, seed=seed
    )
    communities = list(communities_generator)

    # Convert communities to array
    community_array = np.zeros(graph.n_nodes, dtype=np.int64)
    for comm_id, comm_set in enumerate(communities):
        for node in comm_set:
            community_array[node] = comm_id

    return {"communities": community_array, "n_communities": len(communities)}
