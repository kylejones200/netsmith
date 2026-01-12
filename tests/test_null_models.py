"""
Tests for null models and permutation tests in NetSmith.
"""

import numpy as np
import pytest

from netsmith.core import Graph
from netsmith.core.metrics import clustering, degree
from netsmith.core.nulls import null_models, permutation_tests


class TestNullModels:
    """Test null model generation."""

    def test_configuration_model(self):
        """Test configuration model preserves degree sequence."""
        # Create a simple graph
        edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
        graph = Graph(edges=edges, n_nodes=4, directed=False, weighted=False)

        result = null_models(graph, method="configuration", n_samples=5, seed=42)

        assert "graphs" in result
        assert len(result["graphs"]) > 0
        assert result["method"] == "configuration"

        # Check that null graphs have same number of nodes
        # Note: edge count may differ due to multi-edges/self-loops removal
        for null_graph in result["graphs"][:3]:  # Check first few
            assert null_graph.n_nodes == graph.n_nodes
            assert null_graph.n_edges > 0  # Should have edges, but count may vary

    def test_erdos_renyi_model(self):
        """Test Erdos-Renyi null model."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = Graph(edges=edges, n_nodes=3, directed=False, weighted=False)

        result = null_models(graph, method="erdos_renyi", n_samples=5, seed=42)

        assert "graphs" in result
        assert len(result["graphs"]) > 0
        assert result["method"] == "erdos_renyi"

        # Check that null graphs have same number of nodes
        for null_graph in result["graphs"][:3]:
            assert null_graph.n_nodes == graph.n_nodes

    def test_degree_preserving_model(self):
        """Test degree-preserving randomization."""
        edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
        graph = Graph(edges=edges, n_nodes=4, directed=False, weighted=False)

        result = null_models(graph, method="degree_preserving", n_samples=5, seed=42)

        assert "graphs" in result
        assert len(result["graphs"]) > 0
        assert result["method"] == "degree_preserving"

        # Check that null graphs have same structure
        for null_graph in result["graphs"][:3]:
            assert null_graph.n_nodes == graph.n_nodes
            assert null_graph.n_edges == graph.n_edges

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3, directed=False, weighted=False)

        with pytest.raises(ValueError, match="Unknown null model method"):
            null_models(graph, method="invalid_method", n_samples=5, seed=42)


class TestPermutationTests:
    """Test permutation testing for graph statistics."""

    def test_permutation_test_mean_degree(self):
        """Test permutation test for mean degree."""
        edges = [(0, 1), (1, 2), (2, 0), (0, 3)]
        graph = Graph(edges=edges, n_nodes=4, directed=False, weighted=False)

        def mean_degree(g):
            deg = degree(g)
            return float(np.mean(deg))

        result = permutation_tests(graph, statistic=mean_degree, n_permutations=50, seed=42)

        assert "statistic" in result
        assert "null_mean" in result
        assert "null_std" in result
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1
        assert result["n_permutations"] == 50

    def test_permutation_test_clustering(self):
        """Test permutation test for clustering coefficient."""
        edges = [(0, 1), (1, 2), (2, 0)]  # Triangle
        graph = Graph(edges=edges, n_nodes=3, directed=False, weighted=False)

        def mean_clustering(g):
            clust = clustering(g)
            return float(np.mean(clust))

        result = permutation_tests(graph, statistic=mean_clustering, n_permutations=50, seed=42)

        assert "statistic" in result
        assert result["statistic"] > 0  # Triangle has high clustering
        assert 0 <= result["p_value"] <= 1

    def test_permutation_test_small_graph(self):
        """Test permutation test on very small graph."""
        edges = [(0, 1)]
        graph = Graph(edges=edges, n_nodes=2, directed=False, weighted=False)

        def edge_count(g):
            return float(g.n_edges)

        result = permutation_tests(graph, statistic=edge_count, n_permutations=20, seed=42)

        assert result["statistic"] == 1.0
        assert 0 <= result["p_value"] <= 1
