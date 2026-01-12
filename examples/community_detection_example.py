"""
Community Detection Example

This example demonstrates community detection algorithms in NetSmith.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import numpy as np

from netsmith.core import Graph
from netsmith.core.community import label_propagation_hooks, louvain_hooks, modularity

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_community_graph():
    """Create a graph with two clear communities."""
    # Two triangles connected by a single edge
    edges = [
        # First community (nodes 0, 1, 2)
        (0, 1),
        (1, 2),
        (2, 0),
        # Second community (nodes 3, 4, 5)
        (3, 4),
        (4, 5),
        (5, 3),
        # Connection between communities
        (2, 3),
    ]

    graph = Graph(edges=edges, n_nodes=6, directed=False, weighted=False)

    return graph


def main():
    """Main example function."""
    logger.info("NetSmith: Community Detection Example\n")

    # Create a graph with communities
    graph = create_community_graph()
    logger.info(f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges\n")

    # Louvain community detection
    logger.info("Louvain Community Detection:")
    try:
        result = louvain_hooks(graph, resolution=1.0, seed=42)
        communities = result["communities"]
        mod = result["modularity"]
        n_comm = result["n_communities"]

        logger.info(f"  Number of communities: {n_comm}")
        logger.info(f"  Modularity: {mod:.3f}")
        logger.info(f"  Community assignments: {communities}")

        # Show nodes in each community
        for comm_id in range(n_comm):
            nodes = np.where(communities == comm_id)[0]
            logger.info(f"    Community {comm_id}: nodes {list(nodes)}")
        logger.info("")
    except Exception as e:
        logger.warning(f"  Louvain failed: {e}\n")

    # Label propagation
    logger.info("Label Propagation:")
    try:
        result = label_propagation_hooks(graph, seed=42)
        communities = result["communities"]
        n_comm = result["n_communities"]

        logger.info(f"  Number of communities: {n_comm}")
        logger.info(f"  Community assignments: {communities}")

        # Show nodes in each community
        for comm_id in range(n_comm):
            nodes = np.where(communities == comm_id)[0]
            logger.info(f"    Community {comm_id}: nodes {list(nodes)}")
        logger.info("")
    except Exception as e:
        logger.warning(f"  Label propagation failed: {e}\n")

    # Manual modularity computation
    logger.info("Modularity Computation:")
    try:
        # Use Louvain communities
        result = louvain_hooks(graph, resolution=1.0, seed=42)
        communities = result["communities"]

        mod = modularity(graph, communities)
        logger.info(f"  Modularity: {mod:.3f}")
    except Exception as e:
        logger.warning(f"  Modularity computation failed: {e}")


if __name__ == "__main__":
    main()
