"""
Backend dispatch: Selects Python or Rust backend at runtime.

Failure policy
--------------
``backend="auto"`` may quietly use whichever backend is available — that is
what "auto" means. ``backend="rust"`` never falls back: if the extension is not
installed, or has no kernel for the requested computation, that is an error.
Returning the Python result under a Rust label would misreport what ran.
"""

import logging
from typing import Callable, Dict, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..exceptions import BackendError
from .contracts import EdgeList

logger = logging.getLogger(__name__)

Backend = Literal["auto", "python", "rust"]


def _detect_backend(preference: Backend = "auto") -> str:
    """
    Resolve a backend preference against what is actually installed.

    Raises
    ------
    ImportError
        If "rust" was requested and the extension is not importable
    ValueError
        If the preference is not a known backend name
    """
    if preference not in ("auto", "python", "rust"):
        raise ValueError(f"unknown backend: {preference!r} (expected 'auto', 'python' or 'rust')")
    if preference == "python":
        return "python"

    try:
        import netsmith_rs  # type: ignore  # noqa: F401

        return "rust"
    except ImportError:
        if preference == "rust":
            raise ImportError(
                "Rust backend requested but netsmith_rs is not available. "
                "Build it with: maturin develop --release -m rust/Cargo.toml"
            )
        return "python"


def _rust_kernel(name: str, backend: Backend) -> Optional[Callable]:
    """
    Return the named Rust kernel, or None when Python should run instead.

    Raises
    ------
    BackendError
        If the Rust backend was demanded but has no kernel by that name
    """
    if _detect_backend(backend) != "rust":
        return None

    from . import rust

    kernel = getattr(rust, name, None)
    if kernel is None:
        if backend == "rust":
            raise BackendError(
                f"the Rust backend has no {name!r} kernel; use backend='python' or 'auto'"
            )
        logger.debug("No Rust %s kernel, using Python", name)
    return kernel


def compute_degree(edges: EdgeList, backend: Backend = "auto") -> NDArray[np.int64]:
    """
    Compute degree sequence.

    Parameters
    ----------
    edges : EdgeList
        Edge list
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    degrees : array (n_nodes,)
        Degree sequence
    """
    kernel = _rust_kernel("degree_rust", backend)
    if kernel is not None:
        return kernel(edges)

    from .python import degree_python

    return degree_python(edges)


def compute_pagerank(
    edges: EdgeList,
    alpha: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 200,
    backend: Backend = "auto",
) -> NDArray[np.float64]:
    """
    Compute PageRank.

    Parameters
    ----------
    edges : EdgeList
        Edge list
    alpha : float, default 0.85
        Damping factor
    tol : float, default 1e-6
        Convergence tolerance
    max_iter : int, default 200
        Maximum iterations
    backend : str, default "auto"
        Backend: "auto", "python", or "rust". There is no Rust PageRank kernel
        yet, so "rust" is an error rather than a silent Python computation.

    Returns
    -------
    pagerank : array (n_nodes,)
        PageRank scores
    """
    kernel = _rust_kernel("pagerank_rust", backend)
    if kernel is not None:
        return kernel(edges, alpha, tol, max_iter)

    from .python import pagerank_python

    return pagerank_python(edges, alpha=alpha, tol=tol, max_iter=max_iter)


def compute_clustering(edges: EdgeList, backend: Backend = "auto") -> NDArray[np.float64]:
    """
    Compute clustering coefficients.

    Parameters
    ----------
    edges : EdgeList
        Edge list
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    clustering : array (n_nodes,)
        Local clustering coefficients
    """
    kernel = _rust_kernel("clustering_rust", backend)
    if kernel is not None:
        return kernel(edges)

    from .python import clustering_python

    return clustering_python(edges)


def compute_components(edges: EdgeList, backend: Backend = "auto") -> tuple[int, NDArray[np.int64]]:
    """
    Compute connected components.

    Parameters
    ----------
    edges : EdgeList
        Edge list
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    n_components : int
        Number of connected components
    labels : array (n_nodes,)
        Component labels for each node
    """
    kernel = _rust_kernel("components_rust", backend)
    if kernel is not None:
        labels = kernel(edges)
        n_components = int(np.max(labels) + 1) if len(labels) > 0 else 0
        return n_components, labels

    from .python import components_python

    return components_python(edges)


def compute_shortest_paths(
    edges: EdgeList,
    source: Optional[int] = None,
    target: Optional[int] = None,
    weight: Optional[str] = None,
    backend: Backend = "auto",
) -> Union[NDArray[np.int64], Dict[str, Union[float, int]]]:
    """
    Compute shortest paths, measured in hops.

    Parameters
    ----------
    edges : EdgeList
        Edge list
    source : int, optional
        Source node. When omitted, returns the mean shortest path length.
    target : int, optional
        Target node
    weight : str, optional
        Not supported. Weighted shortest paths need Dijkstra, which this
        kernel does not implement; passing a weight raises rather than
        returning hop counts that look like distances.
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    dist : array or dict
        Distance array, or path information when source is omitted

    Raises
    ------
    NotImplementedError
        If `weight` is given
    """
    if weight is not None:
        raise NotImplementedError(
            "weighted shortest paths are not implemented; these distances are hop "
            "counts. Pass weight=None, or use betweenness() for weighted routing."
        )

    if source is not None:
        kernel = _rust_kernel("shortest_paths_rust", backend)
        if kernel is not None:
            return kernel(edges, source, edges.directed)

    from .python import shortest_paths_python

    return shortest_paths_python(edges, source, target, weight)


def compute_betweenness(
    edges: EdgeList,
    normalized: bool = True,
    weight: Optional[bool] = None,
    backend: Backend = "auto",
) -> NDArray[np.float64]:
    """
    Compute betweenness centrality (Brandes).

    Parameters
    ----------
    edges : EdgeList
        Edge list. Self-loops are ignored; parallel edges collapse to the
        lightest one.
    normalized : bool, default True
        Divide by the number of ordered pairs excluding the node, giving
        scores in [0, 1]
    weight : bool, optional
        Read edge weights as shortest-path distances. Defaults to using them
        when the edge list carries them; weights must be strictly positive.
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    betweenness : array (n_nodes,)
        Share of shortest paths between other pairs running through each node
    """
    kernel = _rust_kernel("betweenness_rust", backend)
    if kernel is not None:
        return kernel(edges, normalized=normalized, weight=weight)

    from .python import betweenness_python

    return betweenness_python(edges, normalized=normalized, weight=weight)


def compute_communities(
    edges: EdgeList, method: str = "louvain", backend: Backend = "auto"
) -> NDArray[np.int64]:
    """
    Compute community assignments.

    Parameters
    ----------
    edges : EdgeList
        Edge list
    method : str, default "louvain"
        Community detection method. Only "louvain" is implemented.
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    communities : array (n_nodes,)
        Community assignments
    """
    kernel = _rust_kernel("communities_rust", backend)
    if kernel is not None:
        return kernel(edges, method)

    from .python import communities_python

    return communities_python(edges, method)
