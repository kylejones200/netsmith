"""
Rust backend: Accelerated kernels.
"""

try:
    import netsmith_rs
    
    # Degree functions
    def degree_rust(edges):
        """Compute degree sequence using Rust backend."""
        from ..contracts import EdgeList
        import numpy as np
        
        # Convert EdgeList to format expected by Rust
        u = edges.u
        v = edges.v
        n = edges.n_nodes
        
        # Create edge array [m, 2]
        edge_array = np.column_stack([u, v]).astype(np.uintp)
        
        degrees = netsmith_rs.degree_rust(n, edge_array, edges.directed)
        return degrees
    
    def strength_rust(edges):
        """Compute strength sequence using Rust backend."""
        from ..contracts import EdgeList
        import numpy as np
        
        u = edges.u
        v = edges.v
        w = edges.w
        n = edges.n_nodes
        
        if w is None:
            # Fall back to degree if unweighted
            return degree_rust(edges).astype(np.float64)
        
        edge_array = np.column_stack([u, v]).astype(np.uintp)
        strengths = netsmith_rs.strength_rust(n, edge_array, w, edges.directed)
        return strengths
    
    def clustering_rust(edges):
        """Compute local clustering coefficients using Rust backend."""
        from ..contracts import EdgeList
        import numpy as np
        
        u = edges.u
        v = edges.v
        n = edges.n_nodes
        
        edge_array = np.column_stack([u, v]).astype(np.uintp)
        clustering = netsmith_rs.clustering_local_rust(n, edge_array)
        return clustering
    
    def mean_shortest_path_rust(edges):
        """Compute mean shortest path using Rust backend."""
        from ..contracts import EdgeList
        import numpy as np
        
        u = edges.u
        v = edges.v
        n = edges.n_nodes
        
        edge_array = np.column_stack([u, v]).astype(np.uintp)
        msp = netsmith_rs.mean_shortest_path_rust(n, edge_array)
        return msp
    
    def shortest_paths_rust(edges, source, directed):
        """Compute shortest paths from source using Rust backend."""
        from ..contracts import EdgeList
        import numpy as np
        
        u = edges.u
        v = edges.v
        n = edges.n_nodes
        
        edge_array = np.column_stack([u, v]).astype(np.uintp)
        dist = netsmith_rs.shortest_paths_rust(n, edge_array, source, directed)
        return dist
    
    def components_rust(edges):
        """Compute connected components using Rust backend."""
        from ..contracts import EdgeList
        import numpy as np
        
        u = edges.u
        v = edges.v
        n = edges.n_nodes
        
        edge_array = np.column_stack([u, v]).astype(np.uintp)
        n_components, labels = netsmith_rs.connected_components_rust(n, edge_array)
        return labels
    
    # Backend is available
    _RUST_AVAILABLE = True
    
except ImportError:
    # Rust backend not available
    _RUST_AVAILABLE = False
    
    def degree_rust(edges):
        raise ImportError("Rust backend not available")
    
    def strength_rust(edges):
        raise ImportError("Rust backend not available")
    
    def clustering_rust(edges):
        raise ImportError("Rust backend not available")
    
    def mean_shortest_path_rust(edges):
        raise ImportError("Rust backend not available")
    
    def components_rust(edges):
        raise ImportError("Rust backend not available")
    
    def shortest_paths_rust(edges, source, directed):
        raise ImportError("Rust backend not available")

__all__ = [
    "degree_rust",
    "strength_rust",
    "clustering_rust",
    "mean_shortest_path_rust",
    "components_rust",
    "shortest_paths_rust",
    "_RUST_AVAILABLE",
]
