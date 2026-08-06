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

    def _to_int64_distances(raw):
        """Map the kernel's usize::MAX onto the shared UNREACHABLE sentinel.

        Without this the two backends disagree on what "unreachable" is, and
        `distances == UNREACHABLE` silently misses on one of them.
        """
        import numpy as np

        from ..contracts import UNREACHABLE

        raw = np.asarray(raw)
        distances = np.where(raw == np.iinfo(np.uintp).max, UNREACHABLE, raw)
        return distances.astype(np.int64)

    def shortest_paths_rust(edges, source, directed):
        """Compute shortest paths from source using Rust backend."""
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        u = edges.u
        v = edges.v
        n = edges.n_nodes

        edge_array = np.column_stack([u, v]).astype(np.uintp)
        dist = netsmith_rs.shortest_paths_rust(n, edge_array, source, directed)
        return _to_int64_distances(dist)

    def shortest_paths_multi_rust(edges, sources):
        """Compute hop distances from several sources using the Rust backend.

        Builds the adjacency list once and sweeps the sources in parallel.
        """
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        n = edges.n_nodes
        _check_non_negative_nodes(edges)
        sources = np.asarray(sources, dtype=np.int64).ravel()
        if sources.size and sources.min() < 0:
            raise ValueError("source node ids must be non-negative")

        edge_array = np.column_stack([edges.u, edges.v]).astype(np.uintp)
        distances = netsmith_rs.shortest_paths_multi_rust(
            n, edge_array, sources.astype(np.uintp), bool(edges.directed)
        )
        return _to_int64_distances(distances)

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

    def betweenness_rust(edges, normalized=True, weight=None):
        """Compute betweenness centrality using the Rust backend.

        Self-loops are ignored and parallel edges collapse to the lightest one.
        Weights, when used, are read as shortest-path distances and must be
        strictly positive.
        """
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        n = edges.n_nodes
        _check_non_negative_nodes(edges)
        weighted = (edges.w is not None) if weight is None else bool(weight)
        if weighted and edges.w is None:
            raise ValueError("weight=True requires an edge list with weights")

        edge_array = np.column_stack([edges.u, edges.v]).astype(np.uintp)
        weights = np.ascontiguousarray(edges.w, dtype=np.float64) if weighted else None

        scores = netsmith_rs.betweenness_rust(
            n, edge_array, weights, bool(edges.directed), bool(normalized)
        )
        return np.asarray(scores, dtype=np.float64)

    def core_numbers_rust(edges):
        """Compute k-core numbers using the Rust backend."""
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        _check_non_negative_nodes(edges)
        edge_array = np.column_stack([edges.u, edges.v]).astype(np.uintp)
        cores = netsmith_rs.core_numbers_rust(edges.n_nodes, edge_array)
        return np.asarray(cores, dtype=np.int64)

    def label_propagation_rust(edges, seed=None, max_iter=100):
        """Detect communities by label propagation using the Rust backend."""
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        _check_non_negative_nodes(edges)
        edge_array = np.column_stack([edges.u, edges.v]).astype(np.uintp)
        weights = None if edges.w is None else np.ascontiguousarray(edges.w, dtype=np.float64)
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**63))

        labels = netsmith_rs.label_propagation_rust(
            edges.n_nodes, edge_array, weights, int(seed) % (2**64), int(max_iter)
        )
        labels = np.asarray(labels, dtype=np.int64)
        n_communities = int(labels.max()) + 1 if labels.size else 0
        return {"communities": labels, "n_communities": n_communities}

    def configuration_model_rust(degrees, seed):
        """Sample a simple graph with the given degree sequence (Rust backend)."""
        import numpy as np

        degrees = np.ascontiguousarray(degrees, dtype=np.uintp)
        edges, discarded = netsmith_rs.configuration_model_rust(degrees, int(seed) % (2**64))
        return np.asarray(edges, dtype=np.int64), int(discarded)

    def erdos_renyi_rust(n, m, seed):
        """Sample a random simple graph with n nodes and m edges (Rust backend)."""
        import numpy as np

        edges = netsmith_rs.erdos_renyi_rust(int(n), int(m), int(seed) % (2**64))
        return np.asarray(edges, dtype=np.int64)

    def rewire_degree_preserving_rust(edges, n_samples, target_swaps, max_attempts, seed):
        """Generate degree-preserving null models using the Rust backend.

        Returns (edges, swaps, attempts): an array of rewired edge lists plus
        what each sample actually achieved, so a sample that could not be
        randomized is visible rather than silently returned as-is.
        """
        import numpy as np

        from ..contracts import EdgeList  # noqa: F401

        n = edges.n_nodes
        _check_non_negative_nodes(edges)
        edge_array = np.column_stack([edges.u, edges.v]).astype(np.uintp)

        rewired, swaps, attempts = netsmith_rs.rewire_degree_preserving_rust(
            n, edge_array, int(n_samples), int(target_swaps), int(max_attempts), int(seed)
        )
        return np.asarray(rewired, dtype=np.int64), list(swaps), list(attempts)

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

    def shortest_paths_multi_rust(edges, sources):
        raise ImportError("Rust backend not available")

    def betweenness_rust(edges, normalized=True, weight=None):
        raise ImportError("Rust backend not available")

    def core_numbers_rust(edges):
        raise ImportError("Rust backend not available")

    def label_propagation_rust(edges, seed=None, max_iter=100):
        raise ImportError("Rust backend not available")

    def configuration_model_rust(degrees, seed):
        raise ImportError("Rust backend not available")

    def erdos_renyi_rust(n, m, seed):
        raise ImportError("Rust backend not available")

    def rewire_degree_preserving_rust(edges, n_samples, target_swaps, max_attempts, seed):
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
    "shortest_paths_multi_rust",
    "betweenness_rust",
    "core_numbers_rust",
    "label_propagation_rust",
    "configuration_model_rust",
    "erdos_renyi_rust",
    "rewire_degree_preserving_rust",
    "louvain_rust",
    "modularity_rust",
    "communities_rust",
    "_RUST_AVAILABLE",
]
