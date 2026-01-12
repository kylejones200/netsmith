"""
Tests for engine contracts (EdgeList, GraphData).
"""

import pytest
import numpy as np
from netsmith.engine.contracts import EdgeList, GraphData


class TestEdgeList:
    """Tests for EdgeList data contract."""
    
    def test_create_simple_edge_list(self):
        """Test creating a simple edge list."""
        u = np.array([0, 1, 2], dtype=np.int64)
        v = np.array([1, 2, 0], dtype=np.int64)
        
        edges = EdgeList(u=u, v=v, directed=False)
        
        assert len(edges.u) == 3
        assert len(edges.v) == 3
        assert edges.w is None
        assert edges.directed is False
        assert edges.n_nodes == 3  # auto-calculated
    
    def test_create_directed_edge_list(self):
        """Test creating a directed edge list."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 0], dtype=np.int64)
        
        edges = EdgeList(u=u, v=v, directed=True, n_nodes=2)
        
        assert edges.directed is True
        assert edges.n_nodes == 2
    
    def test_create_weighted_edge_list(self):
        """Test creating a weighted edge list."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 0], dtype=np.int64)
        w = np.array([0.5, 1.5], dtype=np.float64)
        
        edges = EdgeList(u=u, v=v, w=w, directed=False)
        
        assert edges.w is not None
        assert np.array_equal(edges.w, w)
    
    def test_auto_calculate_n_nodes(self):
        """Test that n_nodes is auto-calculated if not provided."""
        u = np.array([0, 1, 2, 3], dtype=np.int64)
        v = np.array([1, 2, 3, 0], dtype=np.int64)
        
        edges = EdgeList(u=u, v=v)
        
        assert edges.n_nodes == 4  # max index + 1
    
    def test_validation_u_v_length_mismatch(self):
        """Test that mismatched u/v lengths raise ValueError."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1], dtype=np.int64)  # Different length
        
        with pytest.raises(ValueError, match="u and v must have same length"):
            EdgeList(u=u, v=v)
    
    def test_validation_w_length_mismatch(self):
        """Test that mismatched weight length raises ValueError."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 0], dtype=np.int64)
        w = np.array([0.5], dtype=np.float64)  # Wrong length
        
        with pytest.raises(ValueError, match="w must have same length"):
            EdgeList(u=u, v=v, w=w)
    
    def test_n_nodes_explicit(self):
        """Test that explicit n_nodes overrides auto-calculation."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 0], dtype=np.int64)
        
        edges = EdgeList(u=u, v=v, n_nodes=10)
        
        assert edges.n_nodes == 10


class TestGraphData:
    """Tests for GraphData container."""
    
    def test_create_graph_data(self):
        """Test creating GraphData from EdgeList."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 0], dtype=np.int64)
        edges = EdgeList(u=u, v=v)
        
        graph_data = GraphData(edges=edges)
        
        assert graph_data.edges == edges

