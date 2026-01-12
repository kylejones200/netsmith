"""
Core null models and permutation tests.
"""

from typing import Callable, Dict, Literal, Optional

import numpy as np

from .graph import Graph


def null_models(
    graph: Graph,
    method: Literal["configuration", "erdos_renyi", "degree_preserving"] = "configuration",
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> Dict:
    """
    Generate null model graphs.

    Parameters
    ----------
    graph : Graph
        Input graph
    method : str, default "configuration"
        Null model method: "configuration", "erdos_renyi", "degree_preserving"
    n_samples : int, default 100
        Number of null model samples
    seed : int, optional
        Random seed

    Returns
    -------
    result : dict
        Dictionary with null model graphs
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for null model generation. Install with: pip install networkx"
        )

    rng = np.random.default_rng(seed)
    nx_graph = graph.as_networkx()

    # Convert to undirected for null models
    if nx_graph.is_directed():
        nx_graph = nx_graph.to_undirected()

    null_graphs = []

    if method == "configuration":
        # Configuration model: preserve degree sequence
        degree_seq = [d for n, d in nx_graph.degree()]
        for _ in range(n_samples):
            try:
                null_g = nx.configuration_model(degree_seq, seed=rng)
                # Remove self-loops and parallel edges
                null_g = nx.Graph(null_g)
                null_g.remove_edges_from(nx.selfloop_edges(null_g))
                null_graphs.append(null_g)
            except Exception:
                # If configuration model fails, skip this sample
                continue

    elif method == "erdos_renyi":
        # Erdos-Renyi: same number of nodes and edges
        n = nx_graph.number_of_nodes()
        m = nx_graph.number_of_edges()
        p = 2 * m / (n * (n - 1)) if n > 1 else 0.0
        for _ in range(n_samples):
            null_g = nx.erdos_renyi_graph(n, p, seed=rng)
            null_graphs.append(null_g)

    elif method == "degree_preserving":
        # Degree-preserving randomization (double edge swap)
        for _ in range(n_samples):
            null_g = nx_graph.copy()
            m = null_g.number_of_edges()
            if m > 0:
                try:
                    nx.double_edge_swap(null_g, nswap=5 * m, max_tries=100 * m, seed=rng)
                except Exception:
                    pass
            null_graphs.append(null_g)

    else:
        raise ValueError(f"Unknown null model method: {method}")

    # Convert back to Graph objects
    from .graph import Graph as GraphClass

    graph_list = []
    for null_g in null_graphs:
        edges = list(null_g.edges())
        if null_g.number_of_nodes() > 0:
            n_nodes = max(max(u, v) for u, v in edges) + 1 if edges else null_g.number_of_nodes()
        else:
            n_nodes = 0
        graph_list.append(GraphClass(edges=edges, n_nodes=n_nodes, directed=False, weighted=False))

    return {"graphs": graph_list, "method": method, "n_samples": len(graph_list)}


def permutation_tests(
    graph: Graph, statistic: Callable, n_permutations: int = 1000, seed: Optional[int] = None
) -> Dict:
    """
    Permutation test for graph statistics.

    Parameters
    ----------
    graph : Graph
        Input graph
    statistic : callable
        Function that computes a statistic from a graph
    n_permutations : int, default 1000
        Number of permutations
    seed : int, optional
        Random seed

    Returns
    -------
    result : dict
        Dictionary with test results
    """
    rng = np.random.default_rng(seed)

    # Compute observed statistic
    observed_stat = float(statistic(graph))

    # Generate permuted graphs and compute statistics
    null_stats = []
    for _ in range(n_permutations):
        # Create a permuted graph by shuffling node labels
        perm = rng.permutation(graph.n_nodes)
        src, dst, w = graph.edges_coo()

        # Apply permutation
        perm_src = perm[src]
        perm_dst = perm[dst]

        # Create new graph with permuted edges
        from .graph import Graph as GraphClass

        perm_edges = list(zip(perm_src, perm_dst))
        if w is not None:
            perm_edges = [(u, v, w[i]) for i, (u, v) in enumerate(perm_edges)]

        perm_graph = GraphClass(
            edges=perm_edges,
            n_nodes=graph.n_nodes,
            directed=graph.directed,
            weighted=graph.weighted,
        )

        null_stat = float(statistic(perm_graph))
        null_stats.append(null_stat)

    null_stats = np.array(null_stats)

    # Compute p-value (two-tailed)
    p_value = float(np.mean(np.abs(null_stats) >= np.abs(observed_stat)))

    return {
        "statistic": observed_stat,
        "null_mean": float(np.mean(null_stats)),
        "null_std": float(np.std(null_stats)),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }
