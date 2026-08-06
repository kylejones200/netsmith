"""
Rust backend: Accelerated kernels.
"""

try:
    import netsmith_rs

    def _check_non_negative_nodes(edges):
        """Guard the unsigned cast: negative node ids would wrap silently."""
        import numpy as np

        for name, arr in (("u", edges.u), ("v", edges.v)):
            arr = np.asarray(arr)
            if arr.size and arr.min() < 0:
                raise ValueError(f"edge endpoint array {name} contains negative node ids")

    # Degree functions
    def degree_rust(edges):
        """Compute degree sequence using Rust backend."""
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

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
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

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
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        u = edges.u
        v = edges.v
        n = edges.n_nodes

        edge_array = np.column_stack([u, v]).astype(np.uintp)
        clustering = netsmith_rs.clustering_local_rust(n, edge_array)
        return clustering

    def mean_shortest_path_rust(edges):
        """Compute mean shortest path using Rust backend."""
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        u = edges.u
        v = edges.v
        n = edges.n_nodes

        edge_array = np.column_stack([u, v]).astype(np.uintp)
        msp = netsmith_rs.mean_shortest_path_rust(n, edge_array)
        return msp

    def shortest_paths_rust(edges, source, directed):
        """Compute shortest paths from source using Rust backend."""
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        u = edges.u
        v = edges.v
        n = edges.n_nodes

        edge_array = np.column_stack([u, v]).astype(np.uintp)
        dist = netsmith_rs.shortest_paths_rust(n, edge_array, source, directed)
        return dist

    def components_rust(edges):
        """Compute connected components using Rust backend."""
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        u = edges.u
        v = edges.v
        n = edges.n_nodes

        edge_array = np.column_stack([u, v]).astype(np.uintp)
        n_components, labels = netsmith_rs.connected_components_rust(n, edge_array)
        return labels

    def louvain_rust(edges, resolution=1.0, seed=None, max_levels=20):
        """Detect communities with the Louvain method using the Rust backend.

        Returns a dict with "communities", "modularity", "n_communities" and
        "n_levels". The graph is treated as undirected; duplicate and
        reciprocal edges are summed.
        """
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        n = edges.n_nodes
        _check_non_negative_nodes(edges)
        if seed is not None and int(seed) < 0:
            raise ValueError("seed must be non-negative")
        edge_array = np.column_stack([edges.u, edges.v]).astype(np.uintp)
        weights = None if edges.w is None else np.ascontiguousarray(edges.w, dtype=np.float64)

        labels, modularity, n_communities, n_levels = netsmith_rs.louvain_rust(
            n, edge_array, weights, float(resolution), seed, int(max_levels)
        )
        return {
            "communities": np.asarray(labels, dtype=np.int64),
            "modularity": float(modularity),
            "n_communities": int(n_communities),
            "n_levels": int(n_levels),
        }

    def modularity_rust(edges, labels, resolution=1.0):
        """Compute the modularity of a partition using the Rust backend."""
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        n = edges.n_nodes
        _check_non_negative_nodes(edges)
        label_array = np.ascontiguousarray(labels).ravel()
        if label_array.size and label_array.min() < 0:
            raise ValueError("community ids must be non-negative")
        edge_array = np.column_stack([edges.u, edges.v]).astype(np.uintp)
        label_array = label_array.astype(np.uintp)
        weights = None if edges.w is None else np.ascontiguousarray(edges.w, dtype=np.float64)

        return float(
            netsmith_rs.modularity_rust(n, edge_array, label_array, weights, float(resolution))
        )

    def communities_rust(edges, method="louvain"):
        """Compute community assignments using the Rust backend."""
        if method != "louvain":
            raise ValueError(
                f"unsupported community detection method: {method!r} (expected 'louvain')"
            )
        return louvain_rust(edges)["communities"]

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

    def louvain_rust(edges, resolution=1.0, seed=None, max_levels=20):
        raise ImportError("Rust backend not available")

    def modularity_rust(edges, labels, resolution=1.0):
        raise ImportError("Rust backend not available")

    def communities_rust(edges, method="louvain"):
        raise ImportError("Rust backend not available")


__all__ = [
    "degree_rust",
    "strength_rust",
    "clustering_rust",
    "mean_shortest_path_rust",
    "components_rust",
    "shortest_paths_rust",
    "louvain_rust",
    "modularity_rust",
    "communities_rust",
    "_RUST_AVAILABLE",
]
