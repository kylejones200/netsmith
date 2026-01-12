"""
Tests for engine dispatch (backend selection).
"""

import numpy as np
import pytest

from netsmith.engine.contracts import EdgeList
from netsmith.engine.dispatch import (
    _detect_backend,
    compute_clustering,
    compute_components,
    compute_degree,
    compute_shortest_paths,
)


class TestBackendDetection:
    """Tests for backend detection."""

    def test_detect_backend_auto_rust_available(self):
        """Test auto backend detection when Rust is available."""
        # This will use Rust if available, Python otherwise
        backend = _detect_backend("auto")
        assert backend in ["rust", "python"]

    def test_detect_backend_auto_python_fallback(self):
        """Test auto backend falls back to Python."""
        # Always works
        backend = _detect_backend("auto")
        assert backend in ["rust", "python"]

    def test_detect_backend_python_explicit(self):
        """Test explicit Python backend selection."""
        backend = _detect_backend("python")
        assert backend == "python"

    def test_detect_backend_rust_explicit_unavailable(self):
        """Test explicit Rust backend when unavailable."""
        # If Rust is not available, this should raise ImportError
        # If Rust is available, it should return "rust"
        try:
            backend = _detect_backend("rust")
            assert backend == "rust"
        except ImportError:
            # Rust not available, which is OK for testing
            pass


class TestComputeDegree:
    """Tests for compute_degree dispatch."""

    def test_compute_degree_simple(self):
        """Test computing degree for simple graph."""
        u = np.array([0, 1, 2], dtype=np.int64)
        v = np.array([1, 2, 0], dtype=np.int64)
        edges = EdgeList(u=u, v=v, directed=False, n_nodes=3)

        degrees = compute_degree(edges, backend="auto")

        assert len(degrees) == 3
        assert np.all(degrees == 2)  # Triangle, all degree 2

    def test_compute_degree_directed(self):
        """Test computing degree for directed graph."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 2], dtype=np.int64)
        edges = EdgeList(u=u, v=v, directed=True, n_nodes=3)

        degrees = compute_degree(edges, backend="auto")

        assert len(degrees) == 3
        assert degrees[0] == 1  # Out-degree
        assert degrees[1] == 1
        assert degrees[2] == 0

    def test_compute_degree_python_backend(self):
        """Test computing degree with Python backend."""
        # For undirected graph, use single edge (0, 1)
        u = np.array([0], dtype=np.int64)
        v = np.array([1], dtype=np.int64)
        edges = EdgeList(u=u, v=v, directed=False, n_nodes=2)

        degrees = compute_degree(edges, backend="python")

        assert len(degrees) == 2
        assert np.all(degrees == 1)


class TestComputeClustering:
    """Tests for compute_clustering dispatch."""

    def test_compute_clustering_triangle(self):
        """Test clustering for triangle graph (all nodes have clustering=1)."""
        # Triangle: 0-1-2-0
        u = np.array([0, 1, 2], dtype=np.int64)
        v = np.array([1, 2, 0], dtype=np.int64)
        edges = EdgeList(u=u, v=v, directed=False, n_nodes=3)

        clustering = compute_clustering(edges, backend="auto")

        assert len(clustering) == 3
        # In a triangle, clustering coefficient should be 1.0
        assert np.allclose(clustering, 1.0)

    def test_compute_clustering_path(self):
        """Test clustering for path graph (should be 0)."""
        # Path: 0-1-2
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 2], dtype=np.int64)
        edges = EdgeList(u=u, v=v, directed=False, n_nodes=3)

        clustering = compute_clustering(edges, backend="auto")

        assert len(clustering) == 3
        # In a path, no triangles, so clustering should be 0
        assert np.allclose(clustering, 0.0)


class TestComputeComponents:
    """Tests for compute_components dispatch."""

    def test_compute_components_connected(self):
        """Test components for connected graph."""
        # Triangle: 0-1-2-0 (one component)
        u = np.array([0, 1, 2], dtype=np.int64)
        v = np.array([1, 2, 0], dtype=np.int64)
        edges = EdgeList(u=u, v=v, directed=False, n_nodes=3)

        n_components, labels = compute_components(edges, backend="auto")

        assert n_components == 1
        assert len(labels) == 3
        assert np.all(labels == labels[0])  # All same component

    def test_compute_components_disconnected(self):
        """Test components for disconnected graph."""
        # Two isolated edges: 0-1 and 2-3
        u = np.array([0, 2], dtype=np.int64)
        v = np.array([1, 3], dtype=np.int64)
        edges = EdgeList(u=u, v=v, directed=False, n_nodes=4)

        n_components, labels = compute_components(edges, backend="auto")

        assert n_components == 2
        assert len(labels) == 4
        assert labels[0] == labels[1]  # 0 and 1 in same component
        assert labels[2] == labels[3]  # 2 and 3 in same component
        assert labels[0] != labels[2]  # Different components


class TestComputeShortestPaths:
    """Tests for compute_shortest_paths dispatch."""

    def test_compute_shortest_paths_from_source(self):
        """Test shortest paths from a source node."""
        # Path: 0-1-2
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 2], dtype=np.int64)
        edges = EdgeList(u=u, v=v, directed=False, n_nodes=3)

        dist = compute_shortest_paths(edges, source=0, backend="auto")

        assert len(dist) == 3
        assert dist[0] == 0  # Distance to self
        assert dist[1] == 1
        assert dist[2] == 2

    def test_compute_shortest_paths_disconnected(self):
        """Test shortest paths with disconnected nodes."""
        # Two isolated edges: 0-1 and 2-3
        u = np.array([0, 2], dtype=np.int64)
        v = np.array([1, 3], dtype=np.int64)
        edges = EdgeList(u=u, v=v, directed=False, n_nodes=4)

        dist = compute_shortest_paths(edges, source=0, backend="auto")

        assert len(dist) == 4
        assert dist[0] == 0  # Distance to self
        assert dist[1] == 1  # Connected
        # Nodes 2 and 3 should have large distance (unreachable)
        assert dist[2] > 1000 or dist[2] == np.iinfo(np.int64).max
        assert dist[3] > 1000 or dist[3] == np.iinfo(np.int64).max
