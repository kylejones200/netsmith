"""
Core network metrics: degree, centrality, assortativity, clustering, k-core, components.
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..exceptions import ValidationError
from .graph import Graph


def degree(graph: Graph, node: Optional[int] = None, mode: str = "out") -> Union[NDArray, int]:
    """
    Compute degree sequence or single node degree.

    Parameters
    ----------
    graph : Graph
        Input graph
    node : int, optional
        If provided, return degree for this node only
    mode : str, default "out"
        For directed graphs: "in", "out", or "total"

    Returns
    -------
    degrees : array or int
        Degree sequence or single degree value
    """
    if node is not None:
        # Validate node index
        if not isinstance(node, (int, np.integer)):
            raise ValidationError(f"node must be integer, got {type(node)}")
        if node < 0 or node >= graph.n_nodes:
            raise ValidationError(f"node {node} is out of range [0, {graph.n_nodes})")

        if graph.directed:
            if mode == "in":
                return int(graph.in_degree_sequence()[node])
            elif mode == "out":
                return int(graph.out_degree_sequence()[node])
            else:  # total
                return int(graph.in_degree_sequence()[node] + graph.out_degree_sequence()[node])
        else:
            return int(graph.degree_sequence()[node])

    if graph.directed:
        if mode == "in":
            return graph.in_degree_sequence()
        elif mode == "out":
            return graph.out_degree_sequence()
        else:  # total
            return graph.in_degree_sequence() + graph.out_degree_sequence()
    else:
        return graph.degree_sequence()


def strength(graph: Graph, node: Optional[int] = None, mode: str = "out") -> Union[NDArray, float]:
    """
    Compute strength (sum of edge weights) sequence or single node strength.

    Parameters
    ----------
    graph : Graph
        Input graph (must be weighted)
    node : int, optional
        If provided, return strength for this node only
    mode : str, default "out"
        For directed graphs: "in", "out", or "total"

    Returns
    -------
    strengths : array or float
        Strength sequence or single strength value
    """
    if not graph.weighted:
        # Fall back to degree if unweighted
        return degree(graph, node, mode)

    src, dst, weight = graph.edges_coo()
    if weight is None:
        return degree(graph, node, mode)

    n = graph.n_nodes
    strengths = np.zeros(n, dtype=np.float64)

    if graph.directed:
        if mode in ("out", "total"):
            for i in range(len(src)):
                strengths[src[i]] += weight[i]
        if mode in ("in", "total"):
            for i in range(len(dst)):
                strengths[dst[i]] += weight[i]
    else:
        for i in range(len(src)):
            strengths[src[i]] += weight[i]
            strengths[dst[i]] += weight[i]

    if node is not None:
        return float(strengths[node])
    return strengths


def centrality(graph: Graph, method: str = "degree", **kwargs) -> NDArray:
    """
    Compute centrality measures for nodes in a graph.

    Parameters
    ----------
    graph : Graph
        Input graph
    method : str, default "degree"
        Centrality method. Currently supported: "degree".
        Planned: "betweenness", "closeness", "eigenvector", "pagerank"
    **kwargs
        Additional arguments for specific methods (currently unused)

    Returns
    -------
    centrality : NDArray
        Array (n_nodes,) with centrality scores for each node.
        For "degree" method, returns degree centrality (normalized degrees).

    Raises
    ------
    NotImplementedError
        If method is not yet implemented

    Notes
    -----
    Centrality measures identify important nodes in a network.
    Degree centrality is the simplest measure (normalized degree).
    Other methods will be added in future releases.
    """
    # For now, implement degree centrality
    # Other methods will be added in engine layer
    if method == "degree":
        return degree(graph, mode="total" if graph.directed else "out")
    else:
        raise NotImplementedError(f"Centrality method '{method}' not yet implemented")


def assortativity(graph: Graph, attribute: Optional[NDArray] = None) -> float:
    """
    Compute assortativity coefficient (homophily measure).

    Parameters
    ----------
    graph : Graph
        Input graph
    attribute : NDArray, optional
        Node attribute array (n_nodes,) to compute assortativity on.
        If None, uses degree as the attribute (degree assortativity).

    Returns
    -------
    assortativity : float
        Assortativity coefficient in range [-1, 1].
        - Positive values: similar nodes tend to connect (assortative)
        - Negative values: dissimilar nodes tend to connect (disassortative)
        - Zero: no preference (random mixing)

    Notes
    -----
    Assortativity measures the tendency of nodes to connect to similar nodes.
    Degree assortativity (default) measures whether high-degree nodes connect
    to other high-degree nodes. The coefficient is the Pearson correlation
    of attribute values at edge endpoints.
    """
    if attribute is None:
        attribute = degree(graph)

    src, dst, weight = graph.edges_coo()

    if len(src) == 0:
        return 0.0

    # Compute assortativity
    if graph.directed:
        # For directed: use out-degree for source, in-degree for target
        src_attr = attribute[src]
        dst_attr = attribute[dst]
    else:
        src_attr = attribute[src]
        dst_attr = attribute[dst]

    # Pearson correlation of attributes at edge endpoints
    if len(src_attr) < 2:
        return 0.0

    return float(np.corrcoef(src_attr, dst_attr)[0, 1])


def clustering(graph: Graph, node: Optional[int] = None) -> Union[NDArray, float]:
    """
    Compute clustering coefficient.

    Parameters
    ----------
    graph : Graph
        Input graph
    node : int, optional
        If provided, return clustering for this node only

    Returns
    -------
    clustering : array or float
        Clustering coefficients or single value

    Note
    ----
    This function is a wrapper around api.metrics.clustering to maintain
    backward compatibility. The implementation has been moved to the API layer
    to fix architecture violations (Core layer should not import from Engine).
    """
    # Delegate to API layer (which handles Engine dispatch)
    from ..api.metrics import clustering as api_clustering

    return api_clustering(graph, node=node, backend="auto")


def k_core(graph: Graph, k: int) -> NDArray:
    """
    Compute k-core decomposition.

    Parameters
    ----------
    graph : Graph
        Input graph
    k : int
        Minimum degree for k-core

    Returns
    -------
    core_numbers : array (n_nodes,)
        Core number for each node
    """
    # Convert to NetworkX for k-core computation
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for k-core computation. Install with: pip install networkx"
        )

    nx_graph = graph.as_networkx()

    # Convert to undirected for k-core
    if nx_graph.is_directed():
        nx_graph = nx_graph.to_undirected()

    # Compute core numbers
    core_dict = nx.core_number(nx_graph)

    # Convert to array
    core_numbers = np.zeros(graph.n_nodes, dtype=np.int64)
    for node, core_num in core_dict.items():
        core_numbers[node] = core_num

    return core_numbers


def components(graph: Graph, return_labels: bool = True) -> Union[int, NDArray]:
    """
    Compute connected components.

    Parameters
    ----------
    graph : Graph
        Input graph
    return_labels : bool, default True
        If True, return component labels for each node.
        If False, return number of components.

    Returns
    -------
    labels : array (n_nodes,) or int
        Component labels or number of components

    Note
    ----
    This function is a wrapper around api.metrics.components to maintain
    backward compatibility. The implementation has been moved to the API layer
    to fix architecture violations (Core layer should not import from Engine).
    """
    # Delegate to API layer (which handles Engine dispatch)
    from ..api.metrics import components as api_components

    return api_components(graph, return_labels=return_labels, backend="auto")
