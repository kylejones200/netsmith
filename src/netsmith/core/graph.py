"""
Core graph types and adapters.

Edge lists are the primary exchange format. Supports node and edge attributes.
Supports directed, undirected, weighted, multigraph as explicit modes.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class Graph:
    """
    Lightweight graph representation.

    Primary storage is edges + optional adjacency matrix.
    NetworkX conversion is lazy and optional.

    Attributes
    ----------
    edges : list of (int, int) or (int, int, float)
        Edge list (unweighted or weighted)
    n_nodes : int
        Number of nodes
    directed : bool
        Whether graph is directed
    weighted : bool
        Whether edges have weights

    Examples
    --------
    >>> G = Graph(edges=[(0,1), (1,2)], n_nodes=3)
    >>> G.n_edges
    2
    >>> G.degree_sequence()
    array([1, 2, 1])
    """

    edges: List[Tuple]
    n_nodes: int
    directed: bool = False
    weighted: bool = False
    _adjacency: Optional = None  # numpy array (dense) or scipy sparse matrix
    _degrees: Optional[NDArray] = None
    _in_degrees: Optional[NDArray] = None
    _out_degrees: Optional[NDArray] = None

    @property
    def n_edges(self) -> int:
        """Number of edges"""
        if hasattr(self, "_n_edges_cached"):
            return self._n_edges_cached
        return len(self.edges)

    def degree_sequence(self) -> NDArray[np.int64]:
        """
        Degree sequence (cached).

        For undirected graphs, returns total degree.
        For directed graphs, returns out-degree.

        Returns
        -------
        degrees : array (n_nodes,)
            Degree of each node
        """
        if self.directed:
            return self.out_degree_sequence()
        else:
            if self._degrees is None:
                degrees = np.zeros(self.n_nodes, dtype=np.int64)
                for edge in self.edges:
                    i, j = edge[0], edge[1]
                    degrees[i] += 1
                    if i != j:
                        degrees[j] += 1
                self._degrees = degrees
            return self._degrees

    def in_degree_sequence(self) -> NDArray[np.int64]:
        """In-degree sequence for directed graphs (cached)."""
        if not self.directed:
            raise ValueError("in_degree_sequence() only valid for directed graphs")
        if self._in_degrees is None:
            in_degrees = np.zeros(self.n_nodes, dtype=np.int64)
            for edge in self.edges:
                j = edge[1]
                in_degrees[j] += 1
            self._in_degrees = in_degrees
        return self._in_degrees

    def out_degree_sequence(self) -> NDArray[np.int64]:
        """Out-degree sequence for directed graphs (cached)."""
        if not self.directed:
            raise ValueError("out_degree_sequence() only valid for directed graphs")
        if self._out_degrees is None:
            out_degrees = np.zeros(self.n_nodes, dtype=np.int64)
            for edge in self.edges:
                i = edge[0]
                out_degrees[i] += 1
            self._out_degrees = out_degrees
        return self._out_degrees

    def adjacency_matrix(self, format: str = "dense"):
        """
        Adjacency matrix (lazy, dense by default for small graphs).

        Parameters
        ----------
        format : str, default "dense"
            Output format: "dense" (numpy array), "sparse" (CSR), or "coo"
            Note: "sparse" and "coo" require scipy to be installed.

        Returns
        -------
        A : numpy.ndarray, scipy.sparse.csr_matrix, or scipy.sparse.coo_matrix
            Adjacency matrix. Returns dense array by default (no scipy required).
            For sparse formats, scipy must be installed: pip install netsmith[scipy]
        """
        # Always allow dense format (pure numpy)
        if format == "dense":
            if self._adjacency is None or not isinstance(self._adjacency, np.ndarray):
                adj = np.zeros(
                    (self.n_nodes, self.n_nodes), dtype=np.float64 if self.weighted else np.int64
                )
                if self.weighted:
                    for edge in self.edges:
                        adj[edge[0], edge[1]] = edge[2]
                        if not self.directed:
                            adj[edge[1], edge[0]] = edge[2]
                else:
                    for edge in self.edges:
                        adj[edge[0], edge[1]] = 1
                        if not self.directed:
                            adj[edge[1], edge[0]] = 1
                self._adjacency = adj
            return self._adjacency

        # Sparse formats require scipy
        if format in ("sparse", "coo"):
            try:
                from scipy import sparse as sp
            except ImportError:
                raise ImportError(
                    "scipy is required for sparse matrix formats. "
                    "Install with: pip install netsmith[scipy] or pip install scipy"
                )

            if self.n_nodes > 50_000 and format != "sparse":
                raise ValueError(
                    f"Refusing to build dense adjacency matrix for n={self.n_nodes} nodes. "
                    f"This would require ~{self.n_nodes**2 * 8 / 1e9:.1f} GB of memory. "
                    f"Use format='sparse' or format='coo' instead."
                )

            if self._adjacency is None or not isinstance(self._adjacency, sp.spmatrix):
                if len(self.edges) == 0:
                    self._adjacency = sp.coo_matrix((self.n_nodes, self.n_nodes))
                else:
                    if self.weighted:
                        rows = [e[0] for e in self.edges]
                        cols = [e[1] for e in self.edges]
                        data = [e[2] for e in self.edges]
                    else:
                        rows = [e[0] for e in self.edges]
                        cols = [e[1] for e in self.edges]
                        data = [1.0] * len(self.edges)

                    if not self.directed:
                        reverse_rows = cols.copy()
                        reverse_cols = rows.copy()
                        rows = rows + reverse_rows
                        cols = cols + reverse_cols
                        data = data + data

                    self._adjacency = sp.coo_matrix(
                        (data, (rows, cols)), shape=(self.n_nodes, self.n_nodes)
                    )

            if format == "coo":
                return self._adjacency.tocoo()
            else:
                return self._adjacency.tocsr()

        raise ValueError(f"Unknown format: {format}. Use 'dense', 'sparse', or 'coo'")

    def edges_coo(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """
        Return edges in COO format (coordinate arrays).

        Returns
        -------
        src : array (n_edges,)
            Source node indices
        dst : array (n_edges,)
            Destination node indices
        weight : array (n_edges,) or None
            Edge weights (if weighted), None otherwise
        """
        if len(self.edges) == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                None if not self.weighted else np.array([]),
            )

        try:
            if self.weighted:
                src = np.array([e[0] for e in self.edges], dtype=np.int64)
                dst = np.array([e[1] for e in self.edges], dtype=np.int64)
                weight = np.array([e[2] for e in self.edges], dtype=np.float64)
                return src, dst, weight
            else:
                src = np.array([e[0] for e in self.edges], dtype=np.int64)
                dst = np.array([e[1] for e in self.edges], dtype=np.int64)
                return src, dst, None
        except (IndexError, TypeError):
            import warnings

            warnings.warn(
                f"Edge format issue in edges_coo(). "
                f"First edge: {self.edges[0] if self.edges else 'empty'}",
                UserWarning,
            )
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                None if not self.weighted else np.array([]),
            )

    def as_networkx(self, force: bool = False):
        """Convert to NetworkX graph (optional dependency)."""
        if not force and self.n_nodes > 200_000:
            raise ValueError(
                f"Refusing NetworkX conversion for n={self.n_nodes} nodes. "
                f"Use force=True to override."
            )

        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required. Install with: pip install networkx")

        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        G.add_nodes_from(range(self.n_nodes))

        if self.weighted:
            G.add_weighted_edges_from(self.edges)
        else:
            G.add_edges_from(self.edges)

        return G

    def __repr__(self) -> str:
        return f"Graph(n_nodes={self.n_nodes}, n_edges={self.n_edges}, directed={self.directed})"


class GraphView:
    """
    Read-only view of a graph with optional filtering.

    Useful for creating subgraphs without copying data.
    """

    def __init__(self, graph: Graph, node_mask: Optional[NDArray] = None):
        self.graph = graph
        self.node_mask = node_mask

    @property
    def n_nodes(self) -> int:
        if self.node_mask is not None:
            return int(np.sum(self.node_mask))
        return self.graph.n_nodes

    @property
    def n_edges(self) -> int:
        if self.node_mask is not None:
            # Count edges where both endpoints are in mask
            count = 0
            for edge in self.graph.edges:
                u, v = edge[0], edge[1]
                if self.node_mask[u] and self.node_mask[v]:
                    count += 1
            return count
        return self.graph.n_edges

    def edges_coo(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Return filtered edges in COO format."""
        src, dst, weight = self.graph.edges_coo()

        if self.node_mask is not None:
            mask = self.node_mask[src] & self.node_mask[dst]
            src = src[mask]
            dst = dst[mask]
            if weight is not None:
                weight = weight[mask]

        return src, dst, weight
