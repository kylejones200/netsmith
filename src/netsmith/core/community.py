"""
Core community detection: modularity, Louvain hooks, label propagation hooks.
"""

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .graph import Graph


def modularity(graph: Graph, communities: NDArray, weight: Optional[str] = None) -> float:
    """
    Compute modularity.

    Parameters
    ----------
    graph : Graph
        Input graph
    communities : array (n_nodes,)
        Community assignment for each node
    weight : str, optional
        Edge weight attribute (ignored, uses graph weights if available)

    Returns
    -------
    modularity : float
        Modularity score
    """
    # Convert to NetworkX for modularity computation
    try:
        import networkx as nx
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


def louvain_hooks(graph: Graph, resolution: float = 1.0, seed: Optional[int] = None) -> Dict:
    """
    Louvain community detection hooks.

    Parameters
    ----------
    graph : Graph
        Input graph
    resolution : float, default 1.0
        Resolution parameter
    seed : int, optional
        Random seed

    Returns
    -------
    result : dict
        Dictionary with community assignments and modularity
    """
    # Convert to NetworkX
    try:
        import networkx  # noqa: F401
        from networkx.algorithms import community
    except ImportError:
        raise ImportError(
            "networkx is required for Louvain community detection. Install with: pip install networkx"
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
                "python-louvain is required for Louvain community detection. Install with: pip install python-louvain"
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
    Label propagation community detection hooks.

    Parameters
    ----------
    graph : Graph
        Input graph
    seed : int, optional
        Random seed

    Returns
    -------
    result : dict
        Dictionary with community assignments
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
