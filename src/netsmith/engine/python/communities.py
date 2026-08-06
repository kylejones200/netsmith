"""
Python implementation of community detection (Louvain).

Mirrors the algorithm and conventions of the Rust kernel in
``rust/crates/netsmith-core/src/community.rs``: the graph is treated as
weighted and undirected, parallel edges (and the reciprocal edges of a directed
input) are merged by summing their weights, and a self-loop of weight ``w``
contributes ``2w`` to its node's degree.

The two backends use different RNGs for the node visit order, so the same
`seed` does not guarantee the same partition across backends -- only that each
backend is reproducible on its own. Modularity values are comparable.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..contracts import EdgeList

# Guard against float-noise moves cycling forever; strict improvement means
# real runs stop long before this.
MAX_PASSES = 1000

# Internal graph: adjacency (self-loops excluded), self-loop weights, degrees,
# and total edge weight m.
_Graph = Tuple[List[Dict[int, float]], List[float], List[float], float]


def _build_graph(n: int, u: NDArray, v: NDArray, w: Optional[NDArray]) -> _Graph:
    """Build the internal weighted undirected graph from an edge list."""
    adj: List[Dict[int, float]] = [{} for _ in range(n)]
    self_loops = [0.0] * n

    for idx in range(len(u)):
        a = int(u[idx])
        b = int(v[idx])
        if a >= n or b >= n or a < 0 or b < 0:
            raise ValueError(f"edge ({a}, {b}) references a node id outside [0, {n})")
        weight = 1.0 if w is None else float(w[idx])
        if weight < 0.0:
            raise ValueError(
                "weights must be non-negative (modularity is undefined for negative weights)"
            )
        if a == b:
            self_loops[a] += weight
        else:
            adj[a][b] = adj[a].get(b, 0.0) + weight
            adj[b][a] = adj[b].get(a, 0.0) + weight

    degrees = [2.0 * self_loops[i] + sum(adj[i].values()) for i in range(n)]
    total_weight = 0.5 * sum(degrees)
    return adj, self_loops, degrees, total_weight


def _modularity(graph: _Graph, labels: List[int], resolution: float) -> float:
    """Modularity of a partition of the internal graph."""
    adj, self_loops, degrees, m = graph
    n = len(adj)
    if m <= 0.0 or n == 0:
        return 0.0

    n_comms = max(labels) + 1
    internal = [0.0] * n_comms
    tot = [0.0] * n_comms

    for i in range(n):
        ci = labels[i]
        tot[ci] += degrees[i]
        internal[ci] += self_loops[i]
        for j, weight in adj[i].items():
            if labels[j] == ci:
                # Each intra-community edge is seen from both endpoints.
                internal[ci] += 0.5 * weight

    two_m = 2.0 * m
    return sum(internal[c] / m - resolution * (tot[c] / two_m) ** 2 for c in range(n_comms))


def _local_moving(
    graph: _Graph, labels: List[int], resolution: float, rng: Optional[random.Random]
) -> None:
    """Move single nodes to the best neighbouring community until none moves."""
    adj, _self_loops, degrees, m = graph
    n = len(adj)
    if m <= 0.0:
        return
    two_m = 2.0 * m

    tot = [0.0] * n
    for i in range(n):
        tot[labels[i]] += degrees[i]

    order = list(range(n))
    if rng is not None:
        rng.shuffle(order)

    for _pass in range(MAX_PASSES):
        moves = 0
        for i in order:
            k_i = degrees[i]
            old_comm = labels[i]

            weight_to: Dict[int, float] = {}
            for j, weight in adj[i].items():
                cj = labels[j]
                weight_to[cj] = weight_to.get(cj, 0.0) + weight

            # Isolate the node before scoring, so the "stay" option is scored
            # on the same footing as every move.
            tot[old_comm] -= k_i

            stay_gain = weight_to.get(old_comm, 0.0) - resolution * k_i * tot[old_comm] / two_m
            best_comm = old_comm
            best_gain = stay_gain
            for c, weight in weight_to.items():
                if c == old_comm:
                    continue
                gain = weight - resolution * k_i * tot[c] / two_m
                if gain > best_gain:
                    best_gain = gain
                    best_comm = c

            tot[best_comm] += k_i
            labels[i] = best_comm
            if best_comm != old_comm:
                moves += 1

        if moves == 0:
            break


def _renumber(labels: List[int]) -> int:
    """Relabel in place to consecutive ids ordered by first appearance."""
    mapping: Dict[int, int] = {}
    for idx, label in enumerate(labels):
        if label not in mapping:
            mapping[label] = len(mapping)
        labels[idx] = mapping[label]
    return len(mapping)


def _aggregate(graph: _Graph, labels: List[int], n_comms: int) -> _Graph:
    """Collapse each community into a single node, preserving degrees and m."""
    adj, self_loops, _degrees, m = graph
    new_adj: List[Dict[int, float]] = [{} for _ in range(n_comms)]
    new_self_loops = [0.0] * n_comms

    for i in range(len(adj)):
        ci = labels[i]
        new_self_loops[ci] += self_loops[i]
        for j, weight in adj[i].items():
            cj = labels[j]
            if ci == cj:
                # Seen once from each endpoint; halve to count the edge once.
                new_self_loops[ci] += 0.5 * weight
            else:
                new_adj[ci][cj] = new_adj[ci].get(cj, 0.0) + weight

    new_degrees = [2.0 * new_self_loops[c] + sum(new_adj[c].values()) for c in range(n_comms)]
    return new_adj, new_self_loops, new_degrees, m


def louvain_python(
    edges: EdgeList,
    resolution: float = 1.0,
    seed: Optional[int] = None,
    max_levels: int = 20,
) -> Dict:
    """
    Detect communities with the Louvain method (Python backend).

    Parameters
    ----------
    edges : EdgeList
        Edge list. Treated as undirected; duplicate and reciprocal edges are
        summed. Weights, if present, must be non-negative.
    resolution : float, default 1.0
        Resolution parameter gamma. Higher values yield smaller communities.
    seed : int, optional
        Randomizes the node visit order deterministically. When omitted, nodes
        are visited in index order (also deterministic).
    max_levels : int, default 20
        Cap on aggregation levels.

    Returns
    -------
    result : dict
        - "communities": array (n_nodes,) of community ids in 0..n_communities
        - "modularity": float modularity of the partition
        - "n_communities": int
        - "n_levels": int aggregation levels performed

    Notes
    -----
    A graph with no edges (or all-zero weights) yields one community per node
    and modularity 0.0.
    """
    if not resolution > 0.0 or not np.isfinite(resolution):
        raise ValueError("resolution must be a positive finite number")
    if max_levels < 1:
        raise ValueError("max_levels must be >= 1")

    n = int(edges.n_nodes)
    if n == 0:
        return {
            "communities": np.zeros(0, dtype=np.int64),
            "modularity": 0.0,
            "n_communities": 0,
            "n_levels": 0,
        }

    original = _build_graph(n, edges.u, edges.v, edges.w)
    node_labels = list(range(n))

    if original[3] <= 0.0:
        return {
            "communities": np.arange(n, dtype=np.int64),
            "modularity": 0.0,
            "n_communities": n,
            "n_levels": 0,
        }

    rng = random.Random(seed) if seed is not None else None
    graph = _build_graph(n, edges.u, edges.v, edges.w)
    n_levels = 0

    for _ in range(max_levels):
        labels = list(range(len(graph[0])))
        _local_moving(graph, labels, resolution, rng)
        n_comms = _renumber(labels)

        # No community merged: further levels cannot change anything.
        if n_comms == len(graph[0]):
            break

        node_labels = [labels[label] for label in node_labels]
        n_levels += 1
        graph = _aggregate(graph, labels, n_comms)

    n_communities = _renumber(node_labels)
    return {
        "communities": np.asarray(node_labels, dtype=np.int64),
        "modularity": float(_modularity(original, node_labels, resolution)),
        "n_communities": n_communities,
        "n_levels": n_levels,
    }


def modularity_python(edges: EdgeList, labels: NDArray, resolution: float = 1.0) -> float:
    """
    Compute the modularity of a partition (Python backend).

    Parameters
    ----------
    edges : EdgeList
        Edge list, treated as undirected
    labels : NDArray
        Community id per node, length n_nodes
    resolution : float, default 1.0
        Resolution parameter gamma

    Returns
    -------
    modularity : float
        Modularity score; 0.0 for a graph with no edge weight.
    """
    n = int(edges.n_nodes)
    labels_list = [int(c) for c in np.asarray(labels).ravel()]
    if len(labels_list) != n:
        raise ValueError(f"labels length ({len(labels_list)}) must equal n_nodes ({n})")
    if n == 0:
        return 0.0
    if min(labels_list) < 0:
        raise ValueError("community ids must be non-negative")

    graph = _build_graph(n, edges.u, edges.v, edges.w)
    return float(_modularity(graph, labels_list, resolution))


def label_propagation_python(
    edges: EdgeList, seed: Optional[int] = None, max_iter: int = 100
) -> Dict:
    """
    Detect communities by asynchronous label propagation (Python backend).

    Every node starts alone and repeatedly adopts whichever label carries the
    most edge weight among its neighbours, updating in place so changes spread
    within a pass. Ties are broken at random, so `seed` decides the outcome on
    any graph with ties.

    Parameters
    ----------
    edges : EdgeList
        Edge list, treated as undirected
    seed : int, optional
        Random seed for the visit order and tie-breaks
    max_iter : int, default 100
        Cap on passes

    Returns
    -------
    result : dict
        "communities" (array of ids) and "n_communities"

    Notes
    -----
    Faster than Louvain and needs no resolution parameter, but it optimizes
    nothing explicitly: on graphs without clear structure it can collapse into
    a single community. Prefer `louvain_python` for a partition you can defend
    by its modularity.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    n = int(edges.n_nodes)
    if n == 0:
        return {"communities": np.zeros(0, dtype=np.int64), "n_communities": 0}

    adj, _self_loops, _degrees, _m = _build_graph(n, edges.u, edges.v, edges.w)
    labels = list(range(n))
    rng = random.Random(seed)
    order = list(range(n))

    for _ in range(max_iter):
        rng.shuffle(order)
        changed = False

        for node in order:
            if not adj[node]:
                continue

            weight_to: Dict[int, float] = {}
            for neighbour, weight in adj[node].items():
                label = labels[neighbour]
                weight_to[label] = weight_to.get(label, 0.0) + weight

            best_weight = max(weight_to.values())
            # Choose uniformly among ties, so the visit order alone does not
            # decide it.
            best = rng.choice([lab for lab, w in weight_to.items() if w == best_weight])

            if best != labels[node]:
                labels[node] = best
                changed = True

        if not changed:
            break

    n_communities = _renumber(labels)
    return {
        "communities": np.asarray(labels, dtype=np.int64),
        "n_communities": n_communities,
    }


def communities_python(edges: EdgeList, method: str = "louvain") -> NDArray[np.int64]:
    """
    Compute community assignments (Python backend).

    Parameters
    ----------
    edges : EdgeList
        Edge list, treated as undirected
    method : str, default "louvain"
        Community detection method. Only "louvain" is implemented.

    Returns
    -------
    communities : array (n_nodes,)
        Community id per node

    Raises
    ------
    ValueError
        If `method` is not supported
    """
    if method != "louvain":
        raise ValueError(f"unsupported community detection method: {method!r} (expected 'louvain')")
    return louvain_python(edges)["communities"]
