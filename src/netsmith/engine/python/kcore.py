"""
Python implementation of k-core decomposition.
"""

import numpy as np
from numpy.typing import NDArray

from ..contracts import EdgeList


def core_numbers_python(edges: EdgeList) -> NDArray[np.int64]:
    """
    Compute the core number of every node (Python backend).

    A node's core number is the largest k for which it survives in the k-core:
    the maximal subgraph where every node has at least k neighbours inside it.
    Found by repeatedly peeling the lowest-degree node — the peel level never
    decreases, so each node's core number is the level at which it comes off.

    Direction is ignored, and a self-loop is not a neighbour.

    Parameters
    ----------
    edges : EdgeList
        Edge list

    Returns
    -------
    core_numbers : array (n_nodes,)
        Core number for each node

    Raises
    ------
    ValueError
        If any edge names a node outside [0, n_nodes)

    Notes
    -----
    This scans the remaining nodes to find the minimum each round, which is
    quadratic. The Rust kernel uses a bucket queue and is linear; this is the
    fallback for installs without the extension.
    """
    n = int(edges.n_nodes)
    neighbours = [set() for _ in range(n)]
    for i in range(len(edges.u)):
        u, v = int(edges.u[i]), int(edges.v[i])
        if u < 0 or v < 0 or u >= n or v >= n:
            raise ValueError(f"edge {i} ({u}, {v}) names a node id outside [0, {n})")
        if u != v:
            neighbours[u].add(v)
            neighbours[v].add(u)

    degree = np.array([len(nbrs) for nbrs in neighbours], dtype=np.int64)
    core = np.zeros(n, dtype=np.int64)
    peeled = np.zeros(n, dtype=bool)
    level = 0

    for _ in range(n):
        remaining = np.flatnonzero(~peeled)
        if remaining.size == 0:
            break

        node = int(remaining[np.argmin(degree[remaining])])
        level = max(level, int(degree[node]))
        core[node] = level
        peeled[node] = True

        for other in neighbours[node]:
            if not peeled[other] and degree[other] > 0:
                degree[other] -= 1

    return core
