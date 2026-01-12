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
    Compute degree sequence.

    Parameters
    ----------
    edges : EdgeList, str, or array
        Edge list (or source to load from)
    n_nodes : int, optional
        Number of nodes
    directed : bool, default False
        Whether graph is directed
    weight : str, optional
        Edge weight column (if loading from file)
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    degrees : array (n_nodes,)
        Degree sequence
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
    Compute PageRank.

    Parameters
    ----------
    edges : EdgeList, str, or array
        Edge list (or source to load from)
    n_nodes : int, optional
        Number of nodes
    alpha : float, default 0.85
        Damping factor
    tol : float, default 1e-6
        Convergence tolerance
    max_iter : int, default 200
        Maximum iterations
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    pagerank : array (n_nodes,)
        PageRank scores
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
    Compute community assignments.

    Parameters
    ----------
    edges : EdgeList, str, or array
        Edge list (or source to load from)
    method : str, default "louvain"
        Community detection method
    n_nodes : int, optional
        Number of nodes
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    communities : array (n_nodes,)
        Community assignments
    """
    if not isinstance(edges, EdgeList):
        edges = load_edges(edges, n_nodes=n_nodes)

    return compute_communities(edges, method=method, backend=backend)
