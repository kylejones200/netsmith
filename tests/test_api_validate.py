"""
Tests for API validation functions.
"""

import pytest
import numpy as np
from netsmith.api.validate import validate_edges


class TestValidateEdges:
    """Tests for validate_edges function."""
    
    def test_validate_valid_edges(self):
        """Test validation of valid edge list."""
        u = np.array([0, 1, 2], dtype=np.int64)
        v = np.array([1, 2, 0], dtype=np.int64)
        
        # Should not raise
        validate_edges(u, v)
    
    def test_validate_length_mismatch(self):
        """Test that mismatched lengths raise ValueError."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1], dtype=np.int64)  # Different length
        
        with pytest.raises(ValueError, match="same length"):
            validate_edges(u, v)
    
    def test_validate_negative_nodes(self):
        """Test that negative node indices raise ValueError."""
        u = np.array([0, -1], dtype=np.int64)
        v = np.array([1, 2], dtype=np.int64)
        
        with pytest.raises(ValueError, match="non-negative"):
            validate_edges(u, v)
        
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, -2], dtype=np.int64)
        
        with pytest.raises(ValueError, match="non-negative"):
            validate_edges(u, v)
    
    def test_validate_weights_length(self):
        """Test that weight length must match edges."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 2], dtype=np.int64)
        w = np.array([0.5], dtype=np.float64)  # Wrong length
        
        with pytest.raises(ValueError, match="same length"):
            validate_edges(u, v, w=w)
    
    def test_validate_weights_finite(self):
        """Test that weights must be finite."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 2], dtype=np.int64)
        w = np.array([0.5, np.inf], dtype=np.float64)
        
        with pytest.raises(ValueError, match="finite"):
            validate_edges(u, v, w=w)
        
        w = np.array([0.5, np.nan], dtype=np.float64)
        
        with pytest.raises(ValueError, match="finite"):
            validate_edges(u, v, w=w)
    
    def test_validate_n_nodes(self):
        """Test that node indices must be less than n_nodes."""
        u = np.array([0, 1, 5], dtype=np.int64)  # 5 >= 3
        v = np.array([1, 2, 0], dtype=np.int64)
        
        with pytest.raises(ValueError, match="n_nodes"):
            validate_edges(u, v, n_nodes=3)
    
    def test_validate_with_weights(self):
        """Test validation with valid weights."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 2], dtype=np.int64)
        w = np.array([0.5, 1.5], dtype=np.float64)
        
        # Should not raise
        validate_edges(u, v, w=w)
    
    def test_validate_with_n_nodes(self):
        """Test validation with explicit n_nodes."""
        u = np.array([0, 1], dtype=np.int64)
        v = np.array([1, 2], dtype=np.int64)
        
        # Should not raise
        validate_edges(u, v, n_nodes=3)
        
        # Should raise if nodes out of range
        with pytest.raises(ValueError):
            validate_edges(u, v, n_nodes=2)  # Node 2 is out of range

