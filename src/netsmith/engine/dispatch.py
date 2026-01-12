"""
Backend dispatch: Selects Python or Rust backend at runtime.
"""

import logging
from typing import Dict, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..exceptions import BackendError
from .contracts import EdgeList

logger = logging.getLogger(__name__)

Backend = Literal["auto", "python", "rust"]


def _detect_backend(preference: Backend = "auto") -> str:
    """Detect available backend."""
    if preference == "rust":
        try:
            import netsmith_rs  # type: ignore  # noqa: F401

            return "rust"
        except ImportError:
            if preference == "rust":
                raise ImportError("Rust backend requested but not available")
            return "python"
    elif preference == "python":
        return "python"
    else:  # auto
        try:
            import netsmith_rs  # type: ignore  # noqa: F401

            return "rust"
        except ImportError:
            return "python"


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
    backend_name = _detect_backend(backend)

    if backend_name == "rust":
        try:
            from .rust import degree_rust

            return degree_rust(edges)
        except ImportError:
            # Expected: Rust backend not available, fall back silently
            logger.debug("Rust backend not available for degree computation, using Python")
            pass
        except RuntimeError as e:
            # Unexpected: Rust backend failed, log and raise
            logger.error(f"Rust backend error in degree computation: {e}", exc_info=True)
            raise BackendError(f"Rust backend failed: {e}") from e

    # Python backend
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
        Backend: "auto", "python", or "rust"

    Returns
    -------
    pagerank : array (n_nodes,)
        PageRank scores
    """
    backend_name = _detect_backend(backend)

    if backend_name == "rust":
        try:
            from .rust import pagerank_rust

            return pagerank_rust(edges, alpha, tol, max_iter)
        except ImportError:
            pass

    from .python import pagerank_python

    return pagerank_python(edges, alpha, tol, max_iter)


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
    backend_name = _detect_backend(backend)

    if backend_name == "rust":
        try:
            from .rust import clustering_rust

            return clustering_rust(edges)
        except ImportError:
            # Expected: Rust backend not available, fall back silently
            logger.debug("Rust backend not available for clustering computation, using Python")
            pass
        except RuntimeError as e:
            # Unexpected: Rust backend failed, log and raise
            logger.error(f"Rust backend error in clustering computation: {e}", exc_info=True)
            raise BackendError(f"Rust backend failed: {e}") from e

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
    backend_name = _detect_backend(backend)

    if backend_name == "rust":
        try:
            from .rust import components_rust

            labels = components_rust(edges)
            n_components = int(np.max(labels) + 1) if len(labels) > 0 else 0
            return n_components, labels
        except (ImportError, RuntimeError):
            pass

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
    Compute shortest paths.

    Parameters
    ----------
    edges : EdgeList
        Edge list
    source : int, optional
        Source node
    target : int, optional
        Target node
    weight : str, optional
        Edge weight attribute (not yet supported)
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    dist : array or dict
        Distance array or path information
    """
    backend_name = _detect_backend(backend)

    if backend_name == "rust" and source is not None:
        try:
            from .rust import shortest_paths_rust

            return shortest_paths_rust(edges, source, edges.directed)
        except ImportError:
            # Expected: Rust backend not available, fall back silently
            logger.debug("Rust backend not available for shortest paths computation, using Python")
            pass
        except RuntimeError as e:
            # Unexpected: Rust backend failed, log and raise
            logger.error(f"Rust backend error in shortest paths computation: {e}", exc_info=True)
            raise BackendError(f"Rust backend failed: {e}") from e

    from .python import shortest_paths_python

    return shortest_paths_python(edges, source, target, weight)


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
        Community detection method
    backend : str, default "auto"
        Backend: "auto", "python", or "rust"

    Returns
    -------
    communities : array (n_nodes,)
        Community assignments
    """
    backend_name = _detect_backend(backend)

    if backend_name == "rust":
        try:
            from .rust import communities_rust

            return communities_rust(edges, method)
        except ImportError:
            pass

    from .python import communities_python

    return communities_python(edges, method)
