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
        """Degree-preserving randomization keeps degrees but moves the edges."""
        # Big enough that double_edge_swap has valid swaps to make. On a tiny
        # graph it cannot randomize at all, which is now an error rather than
        # a silent copy of the input.
        import networkx as nx

        source = nx.barabasi_albert_graph(60, 3, seed=7)
        graph = Graph(edges=list(source.edges()), n_nodes=60, directed=False, weighted=False)

        result = null_models(graph, method="degree_preserving", n_samples=5, seed=42)

        assert result["method"] == "degree_preserving"
        assert len(result["graphs"]) == 5

        observed_degrees = sorted(graph.degree_sequence())
        observed_edges = {frozenset(e[:2]) for e in graph.edges}
        for null_graph in result["graphs"]:
            assert null_graph.n_nodes == graph.n_nodes
            assert null_graph.n_edges == graph.n_edges
            # The defining property: same degree sequence...
            assert sorted(null_graph.degree_sequence()) == observed_degrees
            # ...different wiring. A "null" identical to the observed graph
            # would make any significance test against it meaningless.
            assert {frozenset(e[:2]) for e in null_graph.edges} != observed_edges

    def test_degree_preserving_reports_when_it_cannot_randomize(self):
        """A graph too constrained to rewire raises instead of returning itself."""
        # Triangle plus a pendant: double_edge_swap has essentially no valid
        # swap, so there is no degree-preserving null to report.
        edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
        graph = Graph(edges=edges, n_nodes=4, directed=False, weighted=False)

        with pytest.raises(ValueError, match="degree-preserving"):
            null_models(graph, method="degree_preserving", n_samples=5, seed=42)

    def test_degree_preserving_backends_agree_on_the_invariant(self):
        """Both backends preserve degrees and move the wiring.

        They use different RNGs, so the graphs differ — what has to match is
        the property that makes them null models at all.
        """
        import networkx as nx

        source = nx.barabasi_albert_graph(60, 3, seed=7)
        graph = Graph(edges=list(source.edges()), n_nodes=60, directed=False)
        observed_degrees = sorted(graph.degree_sequence())
        observed_edges = {frozenset(e[:2]) for e in graph.edges}

        for backend in ("rust", "python"):
            result = null_models(
                graph, method="degree_preserving", n_samples=3, seed=42, backend=backend
            )
            assert len(result["graphs"]) == 3
            for null_graph in result["graphs"]:
                assert sorted(null_graph.degree_sequence()) == observed_degrees
                assert {frozenset(e[:2]) for e in null_graph.edges} != observed_edges

    def test_degree_preserving_is_reproducible(self):
        """The same seed gives the same null models."""
        import networkx as nx

        source = nx.barabasi_albert_graph(40, 3, seed=3)
        graph = Graph(edges=list(source.edges()), n_nodes=40, directed=False)

        def wirings(seed):
            result = null_models(
                graph, method="degree_preserving", n_samples=2, seed=seed, backend="rust"
            )
            return [sorted(map(tuple, g.edges)) for g in result["graphs"]]

        assert wirings(11) == wirings(11)
        assert wirings(11) != wirings(12)

    def test_degree_preserving_samples_are_independent(self):
        """Samples in one call are different draws, not copies."""
        import networkx as nx

        source = nx.barabasi_albert_graph(50, 3, seed=5)
        graph = Graph(edges=list(source.edges()), n_nodes=50, directed=False)

        result = null_models(graph, method="degree_preserving", n_samples=4, seed=9, backend="rust")
        wirings = {tuple(sorted(map(tuple, g.edges))) for g in result["graphs"]}
        assert len(wirings) == 4

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
