"""
Basic Network Analysis Example

This example demonstrates basic usage of NetSmith with edge lists.
Shows how to load graphs, compute metrics, and perform network analysis.
"""

import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging  # noqa: E402

import numpy as np  # noqa: E402

from netsmith.core import Graph  # noqa: E402
from netsmith.core.metrics import clustering, components, degree  # noqa: E402
from netsmith.core.paths import shortest_paths  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_sample_graph():
    """Create a sample undirected graph from an edge list."""
    # Create a simple triangle plus a connected edge
    # Nodes: 0-1-2 form a triangle, 2-3 is an additional edge
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),  # Triangle
        (2, 3),  # Additional edge
    ]

    graph = Graph(edges=edges, n_nodes=4, directed=False, weighted=False)

    return graph


def main():
    """Main example function."""
    logger.info("NetSmith: Basic Network Analysis Example\n")

    # Create a sample graph
    graph = create_sample_graph()
    logger.info(f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    logger.info(f"Directed: {graph.directed}, Weighted: {graph.weighted}\n")

    # Compute degree sequence
    logger.info("Degree Sequence:")
    deg = degree(graph)
    logger.info(f"  Degrees: {deg}")
    logger.info(f"  Mean degree: {np.mean(deg):.2f}\n")

    # Compute clustering
    logger.info("Clustering Coefficients:")
    clust = clustering(graph)
    logger.info(f"  Clustering: {clust}")
    logger.info(f"  Mean clustering: {np.mean(clust):.2f}\n")

    # Compute connected components
    logger.info("Connected Components:")
    comp_labels = components(graph, return_labels=True)
    n_components = len(np.unique(comp_labels))
    logger.info(f"  Number of components: {n_components}")
    logger.info(f"  Component labels: {comp_labels}\n")

    # Compute shortest paths from node 0
    logger.info("Shortest Paths from Node 0:")
    paths = shortest_paths(graph, source=0)
    logger.info(f"  Distances: {paths}\n")

    # Convert to NetworkX (optional)
    try:
        import networkx  # noqa: F401

        nx_graph = graph.as_networkx()
        logger.info("NetworkX Conversion:")
        logger.info(f"  Type: {type(nx_graph).__name__}")
        logger.info(f"  Nodes: {nx_graph.number_of_nodes()}")
        logger.info(f"  Edges: {nx_graph.number_of_edges()}")
    except ImportError:
        logger.info("NetworkX not available (install with: pip install networkx)")


if __name__ == "__main__":
    main()
