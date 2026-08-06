"""
Python implementation of PageRank.
"""

import warnings

import numpy as np
from numpy.typing import NDArray

from ..contracts import EdgeList


def pagerank_python(
    edges: EdgeList, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 200
) -> NDArray[np.float64]:
    """
    Compute PageRank (Python backend).

    Parameters
    ----------
    edges : EdgeList
        Edge list. An undirected edge list is followed in both directions;
        a directed one only from `u` to `v`. Edge weights, when present, make
        a link proportionally more likely to be followed. Weights must be
        non-negative.
    alpha : float, default 0.85
        Damping factor: the probability of following a link rather than
        teleporting to a uniformly random node.
    tol : float, default 1e-6
        Convergence tolerance. Iteration stops when the L1 change in the
        score vector falls below `n_nodes * tol`.
    max_iter : int, default 200
        Maximum power-iteration steps.

    Returns
    -------
    pagerank : array (n_nodes,)
        Scores summing to 1. Every node scores at least `(1 - alpha) / n`.

    Notes
    -----
    Nodes with no outgoing edges ("dangling" nodes) would otherwise leak their
    score out of the system; their mass is redistributed uniformly at each
    step, which is the standard formulation and what NetworkX does.

    A graph with no edges yields a uniform distribution.
    """
    n = int(edges.n_nodes)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    src = np.asarray(edges.u, dtype=np.int64)
    dst = np.asarray(edges.v, dtype=np.int64)
    if src.size and (src.min() < 0 or dst.min() < 0):
        raise ValueError("edge endpoints must be non-negative")

    if edges.w is None:
        weight = np.ones(src.size, dtype=np.float64)
    else:
        weight = np.asarray(edges.w, dtype=np.float64)
        if weight.size and weight.min() < 0.0:
            raise ValueError("PageRank requires non-negative edge weights")

    # Drop edges pointing outside the node range, matching the other kernels.
    in_range = (src < n) & (dst < n)
    src, dst, weight = src[in_range], dst[in_range], weight[in_range]

    # An undirected edge is a link in both directions; a self-loop is still
    # just one link.
    if not edges.directed:
        off_diagonal = src != dst
        src, dst, weight = (
            np.concatenate([src, dst[off_diagonal]]),
            np.concatenate([dst, src[off_diagonal]]),
            np.concatenate([weight, weight[off_diagonal]]),
        )

    out_strength = np.bincount(src, weights=weight, minlength=n)
    dangling = out_strength == 0.0

    # Share of each edge's source score that flows along it.
    if src.size:
        share = weight / out_strength[src]
    else:
        share = weight

    pagerank = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - alpha) / n

    for _ in range(max_iter):
        # bincount over an empty edge set yields an integer array, so pin dtype.
        inflow = np.bincount(dst, weights=share * pagerank[src], minlength=n).astype(
            np.float64, copy=False
        )
        # Dangling nodes have nowhere to send their score; spread it evenly.
        inflow += pagerank[dangling].sum() / n
        updated = alpha * inflow + teleport

        if np.abs(updated - pagerank).sum() < n * tol:
            return updated
        pagerank = updated

    warnings.warn(
        f"PageRank did not converge within max_iter={max_iter} "
        f"(tol={tol}); returning the last iterate",
        RuntimeWarning,
        stacklevel=2,
    )
    return pagerank
