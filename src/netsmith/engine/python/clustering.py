"""
Python implementation of clustering coefficient.
"""

import numpy as np
from numpy.typing import NDArray

from ..contracts import EdgeList


def clustering_python(edges: EdgeList) -> NDArray[np.float64]:
    """Compute local clustering coefficients (Python backend)."""
    n = edges.n_nodes
    clustering = np.zeros(n, dtype=np.float64)
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(len(edges.u)):
        u, v = int(edges.u[i]), int(edges.v[i])
        if u < n and v < n:
            adj[u].add(v)
            if not edges.directed:
                adj[v].add(u)
    
    # Compute clustering for each node
    for u in range(n):
        neighbors = list(adj[u])
        k = len(neighbors)
        if k < 2:
            clustering[u] = 0.0
            continue
        
        # Count triangles
        triangles = 0
        for i in range(k):
            for j in range(i + 1, k):
                if neighbors[j] in adj[neighbors[i]]:
                    triangles += 1
        
        clustering[u] = (2.0 * triangles) / (k * (k - 1))
    
    return clustering

