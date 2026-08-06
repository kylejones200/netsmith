"""
Python implementation of shortest paths.
"""

from collections import deque
from typing import Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..contracts import UNREACHABLE, EdgeList


def _build_adjacency(edges: EdgeList):
    """Build the adjacency list, rejecting edges that name absent nodes."""
    n = int(edges.n_nodes)
    adj = [[] for _ in range(n)]
    for i in range(len(edges.u)):
        u, v = int(edges.u[i]), int(edges.v[i])
        if u < 0 or v < 0 or u >= n or v >= n:
            raise ValueError(f"edge {i} ({u}, {v}) names a node id outside [0, {n})")
        adj[u].append(v)
        if not edges.directed:
            adj[v].append(u)
    return adj


def _breadth_first(adj, source: int, distances, unreachable: int) -> None:
    """Fill `distances` with hop counts from `source`, in place."""
    distances[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if distances[v] == unreachable:
                distances[v] = distances[u] + 1
                queue.append(v)


def shortest_paths_python(
    edges: EdgeList,
    source: Optional[int] = None,
    target: Optional[int] = None,
    weight: Optional[str] = None,
) -> Union[NDArray, Dict]:
    """Compute shortest paths (Python backend)."""
    n = edges.n_nodes
    adj = _build_adjacency(edges)

    if source is not None:
        if source < 0 or source >= n:
            raise ValueError(f"source {source} is not a node of the graph [0, {n})")
        unreachable = UNREACHABLE
        dist = np.full(n, unreachable, dtype=np.int64)
        _breadth_first(adj, int(source), dist, unreachable)

        if target is not None:
            return {"distance": int(dist[target]) if dist[target] != UNREACHABLE else -1}
        return dist
    else:
        # All pairs - use mean shortest path
        msp = mean_shortest_path_python(edges)
        return {"mean_shortest_path": msp}


def shortest_paths_multi_python(edges: EdgeList, sources) -> NDArray[np.int64]:
    """
    Compute hop distances from each of several sources (Python backend).

    Builds the adjacency list once rather than per source, which is what makes
    this cheaper than calling `shortest_paths_python` in a loop.

    Parameters
    ----------
    edges : EdgeList
        Edge list
    sources : sequence of int
        Node indices to search from

    Returns
    -------
    distances : array (len(sources), n_nodes)
        Row i holds the distances from sources[i]. Unreachable nodes hold
        `np.iinfo(np.int64).max`.

    Raises
    ------
    ValueError
        If any edge or any source names a node outside [0, n_nodes)
    """
    n = int(edges.n_nodes)
    adj = _build_adjacency(edges)

    sources = np.asarray(sources, dtype=np.int64).ravel()
    out_of_range = (sources < 0) | (sources >= n)
    if out_of_range.any():
        bad = int(sources[np.flatnonzero(out_of_range)[0]])
        raise ValueError(f"source {bad} is not a node of the graph [0, {n})")

    unreachable = UNREACHABLE
    distances = np.full((len(sources), n), unreachable, dtype=np.int64)
    for row, source in enumerate(sources):
        _breadth_first(adj, int(source), distances[row], unreachable)
    return distances


def mean_shortest_path_python(edges: EdgeList) -> float:
    """Compute mean shortest path length (Python backend)."""
    n = edges.n_nodes
    adj = _build_adjacency(edges)
    unreachable = UNREACHABLE

    total = 0
    pairs = 0

    for s in range(n):
        dist = np.full(n, unreachable, dtype=np.int64)
        _breadth_first(adj, s, dist, unreachable)

        reached = dist[s + 1 :]
        reached = reached[reached != unreachable]
        total += int(reached.sum())
        pairs += int(reached.size)

    return (total / pairs) if pairs > 0 else np.nan
