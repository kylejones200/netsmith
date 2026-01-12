"""
Python implementation of PageRank.
"""

import numpy as np
from numpy.typing import NDArray

from ..contracts import EdgeList


def pagerank_python(
    edges: EdgeList, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 200
) -> NDArray[np.float64]:
    """Compute PageRank (Python backend)."""
    n = edges.n_nodes
    degrees = np.zeros(n, dtype=np.int64)

    # Compute out-degrees
    for i in range(len(edges.u)):
        u = edges.u[i]
        degrees[u] += 1

    # Initialize PageRank
    pr = np.ones(n, dtype=np.float64) / n

    # Iterate
    for _ in range(max_iter):
        pr_new = np.zeros(n, dtype=np.float64)

        for i in range(len(edges.u)):
            u, v = edges.u[i], edges.v[i]
            if degrees[u] > 0:
                pr_new[v] += alpha * pr[u] / degrees[u]

        # Add teleportation
        pr_new += (1 - alpha) / n

        # Check convergence
        if np.linalg.norm(pr_new - pr) < tol:
            break

        pr = pr_new

    return pr
