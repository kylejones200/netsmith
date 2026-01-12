"""
Tests for core Graph class.
"""

import networkx as nx
import numpy as np
import pytest

from netsmith.core.graph import Graph


class TestGraph:
    """Tests for Graph class."""

    def test_create_simple_graph(self):
        """Test creating a simple undirected graph."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        assert graph.n_nodes == 3
        assert graph.n_edges == 3
        assert graph.directed is False
        assert graph.weighted is False

    def test_create_directed_graph(self):
        """Test creating a directed graph."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3, directed=True)

        assert graph.directed is True
        assert graph.n_edges == 2

    def test_create_weighted_graph(self):
        """Test creating a weighted graph."""
        edges = [(0, 1, 0.5), (1, 2, 1.5)]
        graph = Graph(edges=edges, n_nodes=3, weighted=True)

        assert graph.weighted is True
        assert graph.n_edges == 2

    def test_degree_sequence(self):
        """Test computing degree sequence."""
        edges = [(0, 1), (1, 2), (2, 0)]  # Triangle
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        degrees = graph.degree_sequence()

        assert len(degrees) == 3
        assert np.all(degrees == 2)  # All nodes have degree 2

    def test_degree_sequence_directed(self):
        """Test computing degree sequence for directed graph."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = Graph(edges=edges, n_nodes=3, directed=True)

        out_degrees = graph.out_degree_sequence()
        in_degrees = graph.in_degree_sequence()

        assert len(out_degrees) == 3
        assert len(in_degrees) == 3
        assert np.all(out_degrees == 1)  # Each node has one outgoing edge
        assert np.all(in_degrees == 1)  # Each node has one incoming edge

    def test_edges_coo(self):
        """Test converting edges to COO format."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3)

        u, v, w = graph.edges_coo()

        assert np.array_equal(u, np.array([0, 1]))
        assert np.array_equal(v, np.array([1, 2]))
        assert w is None  # Unweighted

    def test_edges_coo_weighted(self):
        """Test converting weighted edges to COO format."""
        edges = [(0, 1, 0.5), (1, 2, 1.5)]
        graph = Graph(edges=edges, n_nodes=3, weighted=True)

        u, v, w = graph.edges_coo()

        assert np.array_equal(u, np.array([0, 1]))
        assert np.array_equal(v, np.array([1, 2]))
        assert w is not None
        assert np.allclose(w, np.array([0.5, 1.5]))

    def test_adjacency_matrix_sparse(self):
        """Test creating sparse adjacency matrix."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3)

        adj = graph.adjacency_matrix(format="sparse")

        # Should return scipy sparse matrix
        assert hasattr(adj, "toarray")
        dense = adj.toarray()
        assert dense.shape == (3, 3)
        assert dense[0, 1] == 1
        assert dense[1, 2] == 1
        assert dense[1, 0] == 1  # Undirected

    def test_adjacency_matrix_dense(self):
        """Test creating dense adjacency matrix."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3)

        adj = graph.adjacency_matrix(format="dense")

        assert isinstance(adj, np.ndarray)
        assert adj.shape == (3, 3)

    def test_as_networkx(self):
        """Test converting to NetworkX graph."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        G = graph.as_networkx()

        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert 0 in G and 1 in G and 2 in G

    def test_as_networkx_directed(self):
        """Test converting directed graph to NetworkX."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3, directed=True)

        G = graph.as_networkx()

        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2

    def test_empty_graph(self):
        """Test creating an empty graph."""
        graph = Graph(edges=[], n_nodes=3)

        assert graph.n_edges == 0
        assert graph.n_nodes == 3
        degrees = graph.degree_sequence()
        assert np.all(degrees == 0)
