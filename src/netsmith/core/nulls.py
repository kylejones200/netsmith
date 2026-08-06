"""
Core null models and permutation tests.
"""

from typing import Callable, Dict, Literal, Optional

import numpy as np

from .graph import Graph

Backend = Literal["auto", "python", "rust"]


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

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**63))
    edges = graph.to_edge_list()

    if method == "degree_preserving":
        return _degree_preserving(graph, edges, n_samples, seed, swaps_per_edge, backend)
    if method == "configuration":
        return _configuration(graph, edges, n_samples, seed, backend)
    return _erdos_renyi(graph, edges, n_samples, seed, backend)


def _sample_graphs(graph: Graph, edge_arrays) -> list:
    """Wrap sampled edge arrays back into Graph objects."""
    return [
        Graph(
            edges=[(int(u), int(v)) for u, v in sample],
            n_nodes=graph.n_nodes,
            directed=False,
        )
        for sample in edge_arrays
    ]


def _configuration(graph: Graph, edges, n_samples: int, seed: int, backend: str) -> Dict:
    """Sample graphs with the observed degree sequence.

    Simplifying the stub pairing drops self-loops and repeated edges, so the
    realized degrees can fall a little short. The shortfall is reported rather
    than left for the caller to discover.
    """
    from ..engine.dispatch import _rust_kernel

    degrees = graph.degree_sequence()
    kernel = _rust_kernel("configuration_model_rust", backend)

    samples = []
    discarded_total = 0
    for i in range(n_samples):
        if kernel is not None:
            sample, discarded = kernel(degrees, seed + i)
        else:
            from ..engine.python import configuration_model_python

            sample, discarded = configuration_model_python(degrees, seed=seed + i)
        samples.append(sample)
        discarded_total += discarded

    return {
        "graphs": _sample_graphs(graph, samples),
        "method": "configuration",
        "n_samples": len(samples),
        "discarded_pairings": discarded_total,
    }


def _erdos_renyi(graph: Graph, edges, n_samples: int, seed: int, backend: str) -> Dict:
    """Sample graphs with the observed node and edge counts, wired at random."""
    from ..engine.dispatch import _rust_kernel

    n_edges = len(edges.u)
    kernel = _rust_kernel("erdos_renyi_rust", backend)

    samples = []
    for i in range(n_samples):
        if kernel is not None:
            samples.append(kernel(graph.n_nodes, n_edges, seed + i))
        else:
            from ..engine.python import erdos_renyi_python

            samples.append(erdos_renyi_python(graph.n_nodes, n_edges, seed=seed + i))

    return {
        "graphs": _sample_graphs(graph, samples),
        "method": "erdos_renyi",
        "n_samples": len(samples),
    }


def _degree_preserving(
    graph: Graph, edges, n_samples: int, seed: int, swaps_per_edge: int, backend: str
) -> Dict:
    """Randomize by double edge swap, keeping every degree."""
    from ..engine.dispatch import _rust_kernel

    n_edges = len(edges.u)
    target_swaps = swaps_per_edge * n_edges
    max_attempts = 100 * n_edges

    kernel = _rust_kernel("rewire_degree_preserving_rust", backend)
    if kernel is not None:
        rewired, swaps, _attempts = kernel(edges, n_samples, target_swaps, max_attempts, seed)
    else:
        from ..engine.python import degree_preserving_rewire_python

        rewired, swaps, _attempts = degree_preserving_rewire_python(
            edges, n_samples, target_swaps, max_attempts, seed
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

    return {
        "graphs": _sample_graphs(graph, rewired),
        "method": "degree_preserving",
        "n_samples": len(rewired),
    }


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
