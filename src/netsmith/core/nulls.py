"""
Core null models and permutation tests.
"""

from typing import Callable, Dict, Literal, Optional

import numpy as np

from .graph import Graph

Backend = Literal["auto", "python", "rust"]


def _degree_preserving_rust(
    graph: Graph, n_samples: int, seed: Optional[int], swaps_per_edge: int, backend: str
) -> Optional[Dict]:
    """
    Generate degree-preserving nulls with the Rust kernel.

    Returns None when the Rust backend is not selected or not available, so the
    caller can fall back to NetworkX. Never returns a partially randomized
    sample without saying so.
    """
    from ..engine.dispatch import _rust_kernel

    kernel = _rust_kernel("rewire_degree_preserving_rust", backend)
    if kernel is None:
        return None

    edges = graph.to_edge_list()
    n_edges = len(edges.u)
    target_swaps = swaps_per_edge * n_edges
    # The same allowance NetworkX's double_edge_swap uses.
    max_attempts = 100 * n_edges if n_edges else 0
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**63))

    rewired, swaps, _attempts = kernel(
        edges, n_samples, target_swaps, max_attempts, int(seed) % (2**64)
    )

    short = [i for i, done in enumerate(swaps) if done < target_swaps]
    if short:
        raise ValueError(
            f"degree-preserving randomization reached only {swaps[short[0]]} of "
            f"{target_swaps} swaps on sample {short[0]}. Graphs with too few "
            f"swappable edges have no meaningful degree-preserving null, and "
            f"returning the observed graph as its own null would make any test "
            f"against it meaningless."
        )

    graphs = [
        Graph(
            edges=[(int(u), int(v)) for u, v in sample],
            n_nodes=graph.n_nodes,
            directed=False,
        )
        for sample in rewired
    ]
    return {"graphs": graphs, "method": "degree_preserving", "n_samples": len(graphs)}


def null_models(
    graph: Graph,
    method: Literal["configuration", "erdos_renyi", "degree_preserving"] = "configuration",
    n_samples: int = 100,
    seed: Optional[int] = None,
    backend: Backend = "auto",
    swaps_per_edge: int = 5,
) -> Dict:
    """
    Generate null model graphs for statistical comparison.

    Parameters
    ----------
    graph : Graph
        Input graph to generate null models from
    method : {"configuration", "erdos_renyi", "degree_preserving"}, default "configuration"
        Null model method:
        - "configuration": Preserve degree sequence, randomize connections
        - "erdos_renyi": Random graph with same number of nodes and edges
        - "degree_preserving": Preserve degrees, randomize via edge swaps
    n_samples : int, default 100
        Number of null model graphs to generate
    seed : int, optional
        Random seed for reproducibility
    backend : {"auto", "python", "rust"}, default "auto"
        Only "degree_preserving" has a Rust kernel; the other methods are
        NetworkX-backed regardless. "rust" is an error if the extension is
        not built.
    swaps_per_edge : int, default 5
        For "degree_preserving": how many successful swaps to aim for, as a
        multiple of the edge count. Higher means further from the observed
        wiring.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "graphs": List of Graph objects (null model samples)
        - "method": String name of the method used
        - "n_samples": Number of graphs actually generated (may be < n_samples
          if some samples failed for configuration model)

    Raises
    ------
    ImportError
        If NetworkX is needed for the chosen method and is not installed. The
        Rust "degree_preserving" path needs no NetworkX.
    ValueError
        If method is not recognized, or if a graph cannot be randomized —
        returning the observed graph as its own null model would make any
        test against it meaningless

    Notes
    -----
    Null models are used for statistical significance testing. They preserve
    certain properties (e.g., degree sequence) while randomizing others.
    The configuration model may skip some samples if degree sequences are
    invalid (this is expected behavior).
    """
    if method not in ("configuration", "erdos_renyi", "degree_preserving"):
        raise ValueError(f"Unknown null model method: {method}")

    if method == "degree_preserving":
        rewired = _degree_preserving_rust(graph, n_samples, seed, swaps_per_edge, backend)
        if rewired is not None:
            return rewired

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
            # No try/except: a sample that cannot be generated is a real
            # failure, and quietly returning fewer samples than asked for
            # would skew whatever significance test consumes them.
            null_g = nx.configuration_model(degree_seq, seed=rng)
            # Remove self-loops and parallel edges
            null_g = nx.Graph(null_g)
            null_g.remove_edges_from(nx.selfloop_edges(null_g))
            null_graphs.append(null_g)

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
                except nx.NetworkXAlgorithmError as e:
                    # Too few swappable edges to randomize: the "null" graph
                    # would just be the observed one, which is not a null model.
                    raise ValueError(
                        f"degree-preserving randomization failed on this graph: {e}. "
                        f"Graphs with too few swappable edges have no meaningful "
                        f"degree-preserving null."
                    ) from e
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
