"""
Python implementation of connected components.
"""

import numpy as np
from numpy.typing import NDArray
from collections import deque

from ..contracts import EdgeList


def components_python(edges: EdgeList) -> tuple[int, NDArray[np.int64]]:
    """Compute connected components (Python backend)."""
    n = edges.n_nodes
    
    # Build adjacency list (always undirected for components)
    adj = [[] for _ in range(n)]
    for i in range(len(edges.u)):
        u, v = int(edges.u[i]), int(edges.v[i])
        if u < n and v < n:
            adj[u].append(v)
            adj[v].append(u)
    
    # BFS to find components
    labels = np.full(n, -1, dtype=np.int64)
    component_id = 0
    
    for start in range(n):
        if labels[start] != -1:
            continue
        
        queue = deque([start])
        labels[start] = component_id
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if labels[v] == -1:
                    labels[v] = component_id
                    queue.append(v)
        
        component_id += 1
    
    return component_id, labels

