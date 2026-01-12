"""
Tests for core metrics functions.
"""

import numpy as np
import pytest

from netsmith.core.graph import Graph
from netsmith.core.metrics import clustering, components, degree, strength


class TestDegree:
    """Tests for degree function."""

    def test_degree_undirected(self):
        """Test degree computation for undirected graph."""
        edges = [(0, 1), (1, 2), (2, 0)]  # Triangle
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        degrees = degree(graph, mode="out")

        assert len(degrees) == 3
        assert np.all(degrees == 2)

    def test_degree_directed(self):
        """Test degree computation for directed graph."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3, directed=True)

        out_degrees = degree(graph, mode="out")
        in_degrees = degree(graph, mode="in")

        assert out_degrees[0] == 1
        assert out_degrees[1] == 1
        assert out_degrees[2] == 0

        assert in_degrees[0] == 0
        assert in_degrees[1] == 1
        assert in_degrees[2] == 1

    def test_degree_single_node(self):
        """Test degree for a single node."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        deg = degree(graph, node=0, mode="out")

        assert isinstance(deg, (int, np.integer))
        assert deg == 1

    def test_degree_empty_graph(self):
        """Test degree for empty graph."""
        graph = Graph(edges=[], n_nodes=3)

        degrees = degree(graph)

        assert len(degrees) == 3
        assert np.all(degrees == 0)


class TestStrength:
    """Tests for strength function."""

    def test_strength_weighted(self):
        """Test strength computation for weighted graph."""
        edges = [(0, 1, 0.5), (1, 2, 1.5), (2, 0, 2.0)]
        graph = Graph(edges=edges, n_nodes=3, weighted=True, directed=False)

        strengths = strength(graph)

        assert len(strengths) == 3
        # Each node has two edges
        assert np.isclose(strengths[0], 0.5 + 2.0)
        assert np.isclose(strengths[1], 0.5 + 1.5)
        assert np.isclose(strengths[2], 1.5 + 2.0)

    def test_strength_unweighted(self):
        """Test strength for unweighted graph (should equal degree)."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        strengths = strength(graph)
        degrees = degree(graph)

        assert np.allclose(strengths, degrees)


class TestClustering:
    """Tests for clustering function."""

    def test_clustering_triangle(self):
        """Test clustering for triangle (all should be 1.0)."""
        edges = [(0, 1), (1, 2), (2, 0)]  # Triangle
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        clustering_coeffs = clustering(graph)

        assert len(clustering_coeffs) == 3
        assert np.allclose(clustering_coeffs, 1.0)

    def test_clustering_path(self):
        """Test clustering for path (should be 0)."""
        edges = [(0, 1), (1, 2)]  # Path
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        clustering_coeffs = clustering(graph)

        assert len(clustering_coeffs) == 3
        assert np.allclose(clustering_coeffs, 0.0)

    def test_clustering_single_node(self):
        """Test clustering for a single node."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        coeff = clustering(graph, node=0)

        assert isinstance(coeff, float)
        assert coeff == 1.0


class TestComponents:
    """Tests for components function."""

    def test_components_connected(self):
        """Test components for connected graph."""
        edges = [(0, 1), (1, 2), (2, 0)]  # Triangle
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        result = components(graph, return_labels=True)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert np.all(result == result[0])  # All in same component

    def test_components_count(self):
        """Test component count."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        n_components = components(graph, return_labels=False)

        assert isinstance(n_components, (int, np.integer))
        assert n_components == 1

    def test_components_disconnected(self):
        """Test components for disconnected graph."""
        # Two isolated triangles
        edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
        graph = Graph(edges=edges, n_nodes=6, directed=False)

        labels = components(graph, return_labels=True)

        assert len(labels) == 6
        # First triangle
        assert labels[0] == labels[1] == labels[2]
        # Second triangle
        assert labels[3] == labels[4] == labels[5]
        # Different components
        assert labels[0] != labels[3]
