"""
Tests for core paths functions.
"""

import numpy as np
import pytest

from netsmith.core.graph import Graph
from netsmith.core.paths import reachability, shortest_paths


class TestShortestPaths:
    """Tests for shortest_paths function."""

    def test_shortest_paths_from_source(self):
        """Test shortest paths from a source node."""
        # Path: 0-1-2
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        dist = shortest_paths(graph, source=0)

        assert isinstance(dist, np.ndarray)
        assert len(dist) == 3
        assert dist[0] == 0
        assert dist[1] == 1
        assert dist[2] == 2

    def test_shortest_paths_all_pairs(self):
        """Test shortest paths for all pairs (no source)."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        result = shortest_paths(graph)

        # Should return a dict or array
        assert isinstance(result, (dict, np.ndarray))

    def test_shortest_paths_disconnected(self):
        """Test shortest paths with disconnected nodes."""
        # Two isolated edges: 0-1 and 2-3
        edges = [(0, 1), (2, 3)]
        graph = Graph(edges=edges, n_nodes=4, directed=False)

        dist = shortest_paths(graph, source=0)

        assert len(dist) == 4
        assert dist[0] == 0
        assert dist[1] == 1
        # Nodes 2 and 3 should have large distance (unreachable)
        assert dist[2] > 1000 or dist[2] == np.iinfo(np.int64).max
        assert dist[3] > 1000 or dist[3] == np.iinfo(np.int64).max

    def test_shortest_paths_directed(self):
        """Test shortest paths for directed graph."""
        edges = [(0, 1), (1, 2)]  # Directed path
        graph = Graph(edges=edges, n_nodes=3, directed=True)

        dist = shortest_paths(graph, source=0)

        assert dist[0] == 0
        assert dist[1] == 1
        assert dist[2] == 2

        # Reverse direction should not work
        dist_rev = shortest_paths(graph, source=2)
        assert dist_rev[0] > 1000 or dist_rev[0] == np.iinfo(np.int64).max


class TestReachability:
    """Tests for reachability function."""

    def test_reachability_connected(self):
        """Test reachability for connected graph."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3, directed=False)

        reachable = reachability(graph, source=0)

        assert isinstance(reachable, np.ndarray)
        assert len(reachable) == 3
        assert reachable[0] == True  # Self-reachable
        assert reachable[1] == True
        assert reachable[2] == True

    def test_reachability_disconnected(self):
        """Test reachability for disconnected graph."""
        edges = [(0, 1), (2, 3)]
        graph = Graph(edges=edges, n_nodes=4, directed=False)

        reachable = reachability(graph, source=0)

        assert len(reachable) == 4
        assert reachable[0] == True
        assert reachable[1] == True
        assert reachable[2] == False  # Not reachable
        assert reachable[3] == False  # Not reachable

    def test_reachability_directed(self):
        """Test reachability for directed graph."""
        edges = [(0, 1), (1, 2)]
        graph = Graph(edges=edges, n_nodes=3, directed=True)

        reachable = reachability(graph, source=0)

        assert reachable[0] == True
        assert reachable[1] == True
        assert reachable[2] == True

        # Reverse direction
        reachable_rev = reachability(graph, source=2)
        assert reachable_rev[0] == False  # Can't reach 0 from 2
        assert reachable_rev[2] == True
