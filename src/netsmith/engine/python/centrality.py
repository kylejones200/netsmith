"""
Python implementation of centrality measures.

Mirrors the Rust kernel in ``rust/crates/netsmith-core/src/centrality.rs``:
Brandes' algorithm, one shortest-path sweep per source followed by a
back-propagation of dependencies. Self-loops are ignored and parallel edges
collapse to their lightest member, since neither can lie on a shortest path
between two distinct nodes.

The Rust kernel runs the sweeps in parallel; this one is serial, so it is the
fallback rather than the fast path. Both produce the same scores.
"""

import heapq
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..contracts import EdgeList

# Adjacency: for each node, a list of (neighbour, distance) pairs.
_Adjacency = List[List[Tuple[int, float]]]


def _build_adjacency(edges: EdgeList, weighted: bool) -> _Adjacency:
    """Build a directed adjacency list, dropping self-loops."""
    n = int(edges.n_nodes)
    lightest: List[Dict[int, float]] = [{} for _ in range(n)]

    for idx in range(len(edges.u)):
        u = int(edges.u[idx])
        v = int(edges.v[idx])
        if u < 0 or v < 0 or u >= n or v >= n or u == v:
            continue
        distance = float(edges.w[idx]) if weighted else 1.0
        if weighted and not distance > 0.0:
            raise ValueError(
                "weights must be strictly positive when used as shortest-path distances"
            )
        for a, b in ((u, v),) if edges.directed else ((u, v), (v, u)):
            current = lightest[a].get(b)
            if current is None or distance < current:
                lightest[a][b] = distance

    return [sorted(nbrs.items()) for nbrs in lightest]


def _sweep_unweighted(adjacency: _Adjacency, source: int, n: int):
    """BFS from `source`, returning visit order, path counts and predecessors."""
    sigma = [0.0] * n
    sigma[source] = 1.0
    hops = [-1] * n
    hops[source] = 0
    predecessors: List[List[int]] = [[] for _ in range(n)]
    order: List[int] = []

    queue = deque([source])
    while queue:
        v = queue.popleft()
        order.append(v)
        for w, _ in adjacency[v]:
            if hops[w] < 0:
                hops[w] = hops[v] + 1
                queue.append(w)
            if hops[w] == hops[v] + 1:
                sigma[w] += sigma[v]
                predecessors[w].append(v)

    return order, sigma, predecessors


def _sweep_weighted(adjacency: _Adjacency, source: int, n: int):
    """Dijkstra from `source`, returning visit order, path counts and predecessors."""
    sigma = [0.0] * n
    sigma[source] = 1.0
    tentative = [np.inf] * n
    tentative[source] = 0.0
    settled = [False] * n
    predecessors: List[List[int]] = [[] for _ in range(n)]
    order: List[int] = []

    heap: List[Tuple[float, int, int]] = [(0.0, source, source)]
    while heap:
        distance, predecessor, v = heapq.heappop(heap)
        if settled[v]:
            continue
        if v != source:
            sigma[v] += sigma[predecessor]
        settled[v] = True
        order.append(v)

        for w, weight in adjacency[v]:
            if settled[w]:
                continue
            through_v = distance + weight
            if through_v < tentative[w]:
                # A strictly better route discards everything found so far.
                tentative[w] = through_v
                sigma[w] = 0.0
                predecessors[w] = [v]
                heapq.heappush(heap, (through_v, v, w))
            elif through_v == tentative[w]:
                # An equally short route: another way to reach w.
                sigma[w] += sigma[v]
                predecessors[w].append(v)

    return order, sigma, predecessors


def _rescale(betweenness: NDArray, n: int, normalized: bool, directed: bool) -> NDArray:
    """Scale raw betweenness the way NetworkX does."""
    if normalized:
        scale = None if n <= 2 else 1.0 / ((n - 1) * (n - 2))
    elif not directed:
        # Every pair is swept from both endpoints.
        scale = 0.5
    else:
        scale = None

    if scale is not None:
        betweenness *= scale
    return betweenness


def betweenness_python(
    edges: EdgeList, normalized: bool = True, weight: Optional[bool] = None
) -> NDArray[np.float64]:
    """
    Compute betweenness centrality (Python backend).

    Parameters
    ----------
    edges : EdgeList
        Edge list. Self-loops are ignored; parallel edges collapse to the
        lightest one. A directed edge list is followed only from `u` to `v`.
    normalized : bool, default True
        Divide by the number of ordered pairs excluding the node,
        `1/((n-1)(n-2))`, giving scores in [0, 1]. When False, an undirected
        graph is halved instead, since each pair is swept twice.
    weight : bool, optional
        Whether to read `edges.w` as shortest-path distances. Defaults to using
        weights when the edge list carries them. Weights must be strictly
        positive.

    Returns
    -------
    betweenness : array (n_nodes,)
        Share of shortest paths between other pairs that run through each node.

    Notes
    -----
    Endpoints are excluded, and pairs with no path between them contribute
    nothing. Matches `networkx.betweenness_centrality` for the same arguments.
    """
    n = int(edges.n_nodes)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    weighted = (edges.w is not None) if weight is None else bool(weight)
    if weighted and edges.w is None:
        raise ValueError("weight=True requires an edge list with weights")

    adjacency = _build_adjacency(edges, weighted)
    sweep = _sweep_weighted if weighted else _sweep_unweighted
    totals = np.zeros(n, dtype=np.float64)

    for source in range(n):
        order, sigma, predecessors = sweep(adjacency, source, n)

        delta = [0.0] * n
        for w in reversed(order):
            coefficient = (1.0 + delta[w]) / sigma[w]
            for v in predecessors[w]:
                delta[v] += sigma[v] * coefficient
            if w != source:
                totals[w] += delta[w]

    return _rescale(totals, n, normalized, edges.directed)
