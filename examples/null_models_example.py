"""
Null Models Example

This example demonstrates null model generation and statistical testing in NetSmith.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from netsmith.core import Graph
from netsmith.core.nulls import null_models, permutation_tests
from netsmith.core.metrics import clustering

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_sample_graph():
    """Create a sample graph."""
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),  # Triangle
        (2, 3),
        (3, 4),  # Path
    ]

    graph = Graph(edges=edges, n_nodes=5, directed=False, weighted=False)

    return graph


def compute_mean_clustering(graph):
    """Helper function to compute mean clustering coefficient."""
    clust = clustering(graph)
    return float(np.mean(clust))


def main():
    """Main example function."""
    logger.info("NetSmith: Null Models Example\n")

    # Create a sample graph
    graph = create_sample_graph()
    logger.info(f"Original graph: {graph.n_nodes} nodes, {graph.n_edges} edges\n")

    # Generate null models
    logger.info("Null Model Generation:")
    logger.info("  Configuration model (preserves degree sequence):")
    try:
        result = null_models(graph, method="configuration", n_samples=10, seed=42)
        null_graphs = result["graphs"]
        logger.info(f"    Generated {len(null_graphs)} null graphs")

        # Compute clustering on null models
        null_clustering = [compute_mean_clustering(g) for g in null_graphs[:5]]
        logger.info(f"    Mean clustering (first 5): {[f'{c:.3f}' for c in null_clustering]}")
        logger.info("")
    except Exception as e:
        logger.warning(f"    Configuration model failed: {e}\n")

    # Erdos-Renyi null model
    logger.info("  Erdos-Renyi model (same n, m):")
    try:
        result = null_models(graph, method="erdos_renyi", n_samples=10, seed=42)
        null_graphs = result["graphs"]
        logger.info(f"    Generated {len(null_graphs)} null graphs")
        logger.info("")
    except Exception as e:
        logger.warning(f"    Erdos-Renyi model failed: {e}\n")

    # Permutation test
    logger.info("Permutation Test (mean clustering):")
    try:
        result = permutation_tests(
            graph, statistic=compute_mean_clustering, n_permutations=100, seed=42
        )

        logger.info(f"  Observed statistic: {result['statistic']:.3f}")
        logger.info(f"  Null mean: {result['null_mean']:.3f}")
        logger.info(f"  Null std: {result['null_std']:.3f}")
        logger.info(f"  p-value: {result['p_value']:.3f}")

        if result["p_value"] < 0.05:
            logger.info("  Result: Significant (p < 0.05)")
        else:
            logger.info("  Result: Not significant (p >= 0.05)")
    except Exception as e:
        logger.warning(f"  Permutation test failed: {e}")


if __name__ == "__main__":
    main()
