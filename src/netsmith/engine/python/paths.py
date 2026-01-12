"""
Python implementation of shortest paths.
"""

from collections import deque
from typing import Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..contracts import EdgeList


def shortest_paths_python(
    edges: EdgeList,
    source: Optional[int] = None,
    target: Optional[int] = None,
    weight: Optional[str] = None,
) -> Union[NDArray, Dict]:
    """Compute shortest paths (Python backend)."""
    n = edges.n_nodes

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(len(edges.u)):
        u, v = int(edges.u[i]), int(edges.v[i])
        if u < n and v < n:
            adj[u].append(v)
            if not edges.directed:
                adj[v].append(u)

    if source is not None:
        # Single source shortest paths
        dist = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
        queue = deque([source])
        dist[source] = 0

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == np.iinfo(np.int64).max:
                    dist[v] = dist[u] + 1
                    queue.append(v)

        if target is not None:
            return {"distance": int(dist[target]) if dist[target] != np.iinfo(np.int64).max else -1}
        return dist
    else:
        # All pairs - use mean shortest path
        msp = mean_shortest_path_python(edges)
        return {"mean_shortest_path": msp}


def mean_shortest_path_python(edges: EdgeList) -> float:
    """Compute mean shortest path length (Python backend)."""
    n = edges.n_nodes

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(len(edges.u)):
        u, v = int(edges.u[i]), int(edges.v[i])
        if u < n and v < n:
            adj[u].append(v)
            if not edges.directed:
                adj[v].append(u)

    total = 0
    pairs = 0

    for s in range(n):
        dist = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
        queue = deque([s])
        dist[s] = 0

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == np.iinfo(np.int64).max:
                    dist[v] = dist[u] + 1
                    queue.append(v)

        for t in range(s + 1, n):
            if dist[t] != np.iinfo(np.int64).max:
                total += dist[t]
                pairs += 1

    return (total / pairs) if pairs > 0 else np.nan
