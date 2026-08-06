"""
Python implementation of degree computation.
"""

import numpy as np
from numpy.typing import NDArray

from ..contracts import EdgeList


def degree_python(edges: EdgeList) -> NDArray[np.int64]:
    """
    Compute degree sequence (Python backend).

    A self-loop counts once, matching the Rust kernel. Edges naming a node
    outside the graph are an error, not a silently dropped edge.

    Raises
    ------
    ValueError
        If any edge names a node outside [0, n_nodes)
    """
    n = int(edges.n_nodes)
    u = np.asarray(edges.u, dtype=np.int64)
    v = np.asarray(edges.v, dtype=np.int64)

    if u.size:
        out_of_range = (u < 0) | (u >= n) | (v < 0) | (v >= n)
        if out_of_range.any():
            first = int(np.flatnonzero(out_of_range)[0])
            raise ValueError(
                f"edge {first} ({u[first]}, {v[first]}) names a node id outside [0, {n})"
            )

    endpoints = u if edges.directed else np.concatenate([u, v[u != v]])
    return np.bincount(endpoints, minlength=n).astype(np.int64)
