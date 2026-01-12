"""
Input validation: Check inputs and produce clear errors.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray



def validate_edges(
    u: NDArray, v: NDArray, w: Optional[NDArray] = None, n_nodes: Optional[int] = None
) -> None:
    """
    Validate edge list inputs.

    Parameters
    ----------
    u : array
        Source nodes
    v : array
        Destination nodes
    w : array, optional
        Edge weights
    n_nodes : int, optional
        Number of nodes

    Raises
    ------
    ValueError
        If validation fails
    """
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)

    if len(u) != len(v):
        raise ValueError(f"u and v must have same length: {len(u)} != {len(v)}")

    if np.any(u < 0) or np.any(v < 0):
        raise ValueError("Node indices must be non-negative")

    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        if len(w) != len(u):
            raise ValueError(f"w must have same length as u and v: {len(w)} != {len(u)}")
        if np.any(~np.isfinite(w)):
            raise ValueError("Edge weights must be finite")

    if n_nodes is not None:
        max_node = max(np.max(u), np.max(v))
        if max_node >= n_nodes:
            raise ValueError(
                f"Node index {max_node} >= n_nodes {n_nodes}. "
                f"Node indices must be in [0, n_nodes)"
            )
