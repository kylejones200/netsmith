"""
Network Metrics Example

This example demonstrates comprehensive network metrics computation using NetSmith.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging  # noqa: E402

import numpy as np  # noqa: E402

from netsmith.core import Graph  # noqa: E402
from netsmith.core.metrics import (  # noqa: E402
    assortativity,
    clustering,
    components,
    degree,
    k_core,
    strength,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_weighted_graph():
    """Create a weighted, undirected graph."""
    edges = [
        (0, 1, 0.5),
        (1, 2, 1.0),
        (2, 0, 0.8),
        (2, 3, 0.3),
        (3, 4, 0.7),
    ]

    graph = Graph(edges=edges, n_nodes=5, directed=False, weighted=True)

    return graph


def main():
    """Main example function."""
    logger.info("NetSmith: Network Metrics Example\n")

    # Create a weighted graph
    graph = create_weighted_graph()
    logger.info(f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    logger.info(f"Weighted: {graph.weighted}\n")

    # Degree sequence
    logger.info("Degree Sequence:")
    deg = degree(graph)
    logger.info(f"  {deg}\n")

    # Strength sequence (for weighted graphs)
    if graph.weighted:
        logger.info("Strength Sequence:")
        str_seq = strength(graph)
        logger.info(f"  {str_seq}\n")

    # Clustering coefficients
    logger.info("Clustering Coefficients:")
    clust = clustering(graph)
    logger.info(f"  {clust}")
    logger.info(f"  Mean: {np.mean(clust):.3f}\n")

    # K-core decomposition
    logger.info("K-Core Decomposition:")
    # Get core numbers for k=1, 2, 3
    for k in [1, 2, 3]:
        core_nums = k_core(graph, k=k)
        in_core = np.where(core_nums >= k)[0]
        logger.info(f"  {k}-core: nodes {list(in_core)}")
    logger.info("")

    # Connected components
    logger.info("Connected Components:")
    comp_labels = components(graph, return_labels=True)
    n_components = len(np.unique(comp_labels))
    logger.info(f"  Number of components: {n_components}")
    for comp_id in range(n_components):
        nodes = np.where(comp_labels == comp_id)[0]
        logger.info(f"  Component {comp_id}: nodes {list(nodes)}")
    logger.info("")

    # Assortativity
    logger.info("Assortativity:")
    assort = assortativity(graph)
    logger.info(f"  Degree assortativity: {assort:.3f}")


if __name__ == "__main__":
    main()
