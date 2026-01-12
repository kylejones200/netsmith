"""
Public compute functions with stable signatures.
"""

from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..api.load import load_edges
from ..engine.contracts import EdgeList
from ..engine.dispatch import compute_communities, compute_degree, compute_pagerank

Backend = Literal["auto", "python", "rust"]


def degree(
    edges: Union[EdgeList, str, np.ndarray],
    n_nodes: Optional[int] = None,
    directed: bool = False,
    weight: Optional[str] = None,
    backend: Backend = "auto",
) -> NDArray[np.int64]:
    """
    Compute degree sequence for nodes in a graph.

    Parameters
    ----------
    edges : EdgeList, str, or np.ndarray
        Edge list representation. Can be:
        - EdgeList: Direct edge list object
        - str: File path (parquet, csv) to load edges from
        - np.ndarray: 2D array with shape (n_edges, 2) or (n_edges, 3) for weighted
    n_nodes : int, optional
        Number of nodes. If not provided, inferred from edge indices.
    directed : bool, default False
        Whether the graph is directed
    weight : str, optional
        Column name for edge weights (when loading from file)
    backend : Backend, default "auto"
        Computation backend: "auto" (prefer Rust), "python", or "rust"

    Returns
    -------
    degrees : NDArray[np.int64]
        Array (n_nodes,) with degree of each node.
        For directed graphs, returns out-degree.
        For undirected graphs, returns total degree.

    Notes
    -----
    If edges is a string or array, it will be loaded/converted to EdgeList format.
    The backend parameter allows selecting Python or Rust implementation.
    """
    if not isinstance(edges, EdgeList):
        edges = load_edges(edges, directed=directed, n_nodes=n_nodes)

    return compute_degree(edges, backend=backend)


def pagerank(
    edges: Union[EdgeList, str, np.ndarray],
    n_nodes: Optional[int] = None,
    alpha: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 200,
    backend: Backend = "auto",
) -> NDArray[np.float64]:
    """
    Compute PageRank centrality scores.

    Parameters
    ----------
    edges : EdgeList, str, or np.ndarray
        Edge list representation (see degree() for format details)
    n_nodes : int, optional
        Number of nodes. If not provided, inferred from edge indices.
    alpha : float, default 0.85
        Damping factor (probability of following links vs. random jump).
        Must be in [0, 1]. Typical values: 0.85 (web), 0.9 (social networks).
    tol : float, default 1e-6
        Convergence tolerance. Algorithm stops when change < tol.
    max_iter : int, default 200
        Maximum number of iterations before stopping.
    backend : Backend, default "auto"
        Computation backend: "auto" (prefer Rust), "python", or "rust"

    Returns
    -------
    pagerank : NDArray[np.float64]
        Array (n_nodes,) with PageRank score for each node.
        Scores sum to approximately 1.0 (may vary slightly due to convergence).

    Notes
    -----
    PageRank measures the importance of nodes based on link structure.
    Higher scores indicate more "important" or "central" nodes.
    The algorithm iteratively updates scores until convergence.
    """
    if not isinstance(edges, EdgeList):
        edges = load_edges(edges, n_nodes=n_nodes)

    return compute_pagerank(edges, alpha=alpha, tol=tol, max_iter=max_iter, backend=backend)


def communities(
    edges: Union[EdgeList, str, np.ndarray],
    method: str = "louvain",
    n_nodes: Optional[int] = None,
    backend: Backend = "auto",
) -> NDArray[np.int64]:
    """
    Compute community assignments using community detection algorithms.

    Parameters
    ----------
    edges : EdgeList, str, or np.ndarray
        Edge list representation (see degree() for format details)
    method : str, default "louvain"
        Community detection method. Currently supports "louvain".
    n_nodes : int, optional
        Number of nodes. If not provided, inferred from edge indices.
    backend : Backend, default "auto"
        Computation backend: "auto" (prefer Rust), "python", or "rust"
        Note: Community detection may fall back to Python/NetworkX implementation.

    Returns
    -------
    communities : NDArray[np.int64]
        Array (n_nodes,) with community ID for each node.
        Nodes in the same community have the same ID.
        Community IDs are integers starting from 0.

    Notes
    -----
    Community detection finds groups of nodes with dense internal connections.
    The Louvain method optimizes modularity to find communities.
    """
    if not isinstance(edges, EdgeList):
        edges = load_edges(edges, n_nodes=n_nodes)

    return compute_communities(edges, method=method, backend=backend)
