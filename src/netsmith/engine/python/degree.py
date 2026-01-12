"""
Python implementation of degree computation.
"""

import numpy as np
from numpy.typing import NDArray

from ..contracts import EdgeList


def degree_python(edges: EdgeList) -> NDArray[np.int64]:
    """Compute degree sequence (Python backend)."""
    n = edges.n_nodes
    degrees = np.zeros(n, dtype=np.int64)

    u = edges.u
    v = edges.v

    for i in range(len(u)):
        u_idx, v_idx = int(u[i]), int(v[i])
        if u_idx < n:
            degrees[u_idx] += 1
        if not edges.directed and v_idx < n and u_idx != v_idx:
            degrees[v_idx] += 1

    return degrees
