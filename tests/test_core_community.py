"""
Tests for community detection (Louvain) across the Python and Rust backends.

Every behavioural test runs against both backends so the pure-Python fallback
and the Rust kernel stay in step.
"""

import numpy as np
import pytest

from netsmith.core.community import louvain_hooks, modularity
from netsmith.core.graph import Graph
from netsmith.engine.contracts import EdgeList
from netsmith.engine.dispatch import compute_communities
from netsmith.engine.python import louvain_python, modularity_python
from netsmith.engine.rust import _RUST_AVAILABLE, louvain_rust, modularity_rust

BACKENDS = ["python"] + (["rust"] if _RUST_AVAILABLE else [])

requires_rust = pytest.mark.skipif(not _RUST_AVAILABLE, reason="netsmith_rs not built")


def run_louvain(backend, edges, **kwargs):
    """Run Louvain on the named backend."""
    if backend == "rust":
        return louvain_rust(edges, **kwargs)
    return louvain_python(edges, **kwargs)


def run_modularity(backend, edges, labels, **kwargs):
    """Compute modularity on the named backend."""
    if backend == "rust":
        return modularity_rust(edges, labels, **kwargs)
    return modularity_python(edges, labels, **kwargs)


def edge_list(edges, n_nodes, weights=None, directed=False):
    """Build an EdgeList from a list of (u, v) tuples."""
    u = np.array([e[0] for e in edges], dtype=np.int64)
    v = np.array([e[1] for e in edges], dtype=np.int64)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    return EdgeList(u=u, v=v, w=w, directed=directed, n_nodes=n_nodes)


def two_triangles():
    """Two disjoint triangles joined by a single bridge edge."""
    return edge_list([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)], n_nodes=6)


def ring_of_cliques(n_cliques=4, size=6):
    """Cliques of `size` nodes joined in a ring by single edges."""
    edges = []
    for c in range(n_cliques):
        base = c * size
        for i in range(size):
            for j in range(i + 1, size):
                edges.append((base + i, base + j))
        edges.append((base, ((c + 1) % n_cliques) * size))
    return edge_list(edges, n_nodes=n_cliques * size)


@pytest.mark.parametrize("backend", BACKENDS)
class TestLouvain:
    """Behaviour shared by every Louvain backend."""

    def test_finds_two_triangles(self, backend):
        """Two triangles joined by a bridge split into two communities."""
        edges = two_triangles()
        result = run_louvain(backend, edges, seed=42)

        labels = result["communities"]
        assert result["n_communities"] == 2
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]
        # m = 7, each community has 3 internal edges and total degree 7.
        assert result["modularity"] == pytest.approx(2 * (3 / 7 - 0.25))

    def test_finds_ring_of_cliques(self, backend):
        """Each clique in a ring of cliques becomes one community."""
        edges = ring_of_cliques()
        result = run_louvain(backend, edges, seed=7)

        labels = result["communities"]
        assert result["n_communities"] == 4
        for c in range(4):
            base = c * 6
            assert len(set(labels[base : base + 6].tolist())) == 1
        assert result["modularity"] > 0.6

    def test_labels_are_consecutive_from_zero(self, backend):
        """Community ids form a dense range 0..n_communities-1."""
        result = run_louvain(backend, ring_of_cliques(), seed=3)

        labels = result["communities"]
        assert labels.dtype == np.int64
        assert set(labels.tolist()) == set(range(result["n_communities"]))

    def test_reported_modularity_matches_recomputation(self, backend):
        """The reported modularity equals a fresh scoring of the partition."""
        edges = ring_of_cliques(5, 5)
        result = run_louvain(backend, edges, seed=11)

        recomputed = run_modularity(backend, edges, result["communities"])
        assert result["modularity"] == pytest.approx(recomputed)

    def test_same_seed_is_reproducible(self, backend):
        """A given backend and seed always produce the same partition."""
        edges = ring_of_cliques(4, 5)
        first = run_louvain(backend, edges, seed=99)
        second = run_louvain(backend, edges, seed=99)

        np.testing.assert_array_equal(first["communities"], second["communities"])
        assert first["modularity"] == pytest.approx(second["modularity"])

    def test_weights_drive_the_partition(self, backend):
        """Heavy edges hold their endpoints together, light edges do not."""
        # Square 0-1-2-3-0 with heavy weights on 0-1 and 2-3.
        edges = edge_list(
            [(0, 1), (1, 2), (2, 3), (3, 0)], n_nodes=4, weights=[10.0, 0.1, 10.0, 0.1]
        )
        result = run_louvain(backend, edges, seed=1)

        labels = result["communities"]
        assert result["n_communities"] == 2
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]

    def test_resolution_controls_community_size(self, backend):
        """Low resolution merges communities, high resolution splits them."""
        edges = ring_of_cliques()
        coarse = run_louvain(backend, edges, resolution=0.05, seed=5)
        natural = run_louvain(backend, edges, resolution=1.0, seed=5)
        fine = run_louvain(backend, edges, resolution=8.0, seed=5)

        assert coarse["n_communities"] < natural["n_communities"]
        assert natural["n_communities"] == 4
        assert fine["n_communities"] > natural["n_communities"]

    def test_edgeless_graph_gives_singleton_communities(self, backend):
        """With no edges every node is its own community and Q is 0."""
        edges = EdgeList(u=np.array([], dtype=np.int64), v=np.array([], dtype=np.int64), n_nodes=4)
        result = run_louvain(backend, edges)

        assert result["n_communities"] == 4
        assert result["modularity"] == pytest.approx(0.0)
        assert len(result["communities"]) == 4

    def test_zero_weights_behave_like_no_edges(self, backend):
        """All-zero weights carry no structure, so no community forms."""
        edges = edge_list([(0, 1), (1, 2)], n_nodes=3, weights=[0.0, 0.0])
        result = run_louvain(backend, edges)

        assert result["n_communities"] == 3
        assert result["modularity"] == pytest.approx(0.0)

    def test_self_loops_are_handled(self, backend):
        """Self-loops contribute internal weight without breaking the run."""
        edges = edge_list([(0, 0), (0, 1), (1, 1), (2, 3), (3, 3)], n_nodes=4)
        result = run_louvain(backend, edges, seed=2)

        labels = result["communities"]
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_disconnected_components_never_merge(self, backend):
        """Nodes in different components land in different communities."""
        edges = edge_list([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], n_nodes=6)
        result = run_louvain(backend, edges, seed=4)

        labels = result["communities"]
        assert result["n_communities"] == 2
        assert labels[0] != labels[3]

    def test_parallel_edges_are_summed(self, backend):
        """Splitting an edge into parallel halves leaves modularity unchanged."""
        split = edge_list([(0, 1), (1, 0), (1, 2)], n_nodes=3, weights=[0.5, 0.5, 1.0])
        merged = edge_list([(0, 1), (1, 2)], n_nodes=3, weights=[1.0, 1.0])
        labels = np.array([0, 0, 1], dtype=np.int64)

        assert run_modularity(backend, split, labels) == pytest.approx(
            run_modularity(backend, merged, labels)
        )

    def test_modularity_matches_hand_computation(self, backend):
        """Modularity on a 3-node path matches the closed-form value."""
        edges = edge_list([(0, 1), (1, 2)], n_nodes=3)

        assert run_modularity(backend, edges, np.array([0, 0, 0])) == pytest.approx(0.0)
        assert run_modularity(backend, edges, np.array([0, 0, 1])) == pytest.approx(-0.125)

    def test_modularity_rejects_wrong_label_length(self, backend):
        """A partition must cover exactly the nodes in the graph."""
        edges = edge_list([(0, 1), (1, 2)], n_nodes=3)

        with pytest.raises(ValueError):
            run_modularity(backend, edges, np.array([0, 0]))

    def test_negative_weights_are_rejected(self, backend):
        """Modularity is undefined for negative weights, so refuse them."""
        edges = edge_list([(0, 1), (1, 2)], n_nodes=3, weights=[1.0, -1.0])

        with pytest.raises(ValueError):
            run_louvain(backend, edges)

    def test_invalid_resolution_is_rejected(self, backend):
        """Resolution must be a positive finite number."""
        edges = two_triangles()

        with pytest.raises(ValueError):
            run_louvain(backend, edges, resolution=0.0)
        with pytest.raises(ValueError):
            run_louvain(backend, edges, resolution=-1.0)


@requires_rust
class TestBackendAgreement:
    """The Rust kernel and the Python fallback must agree where it matters."""

    @pytest.mark.parametrize(
        "edges",
        [two_triangles(), ring_of_cliques(4, 6), ring_of_cliques(5, 4)],
        ids=["two_triangles", "ring_4x6", "ring_5x4"],
    )
    def test_modularity_agrees_on_the_same_partition(self, edges):
        """Both backends score an identical partition identically."""
        labels = louvain_python(edges, seed=17)["communities"]

        assert modularity_rust(edges, labels) == pytest.approx(
            modularity_python(edges, labels), abs=1e-12
        )

    def test_both_backends_recover_planted_communities(self):
        """Both backends find the four planted cliques."""
        edges = ring_of_cliques(4, 6)

        assert louvain_rust(edges, seed=5)["n_communities"] == 4
        assert louvain_python(edges, seed=5)["n_communities"] == 4


class TestDispatchAndCoreAPI:
    """The public entry points route to a real implementation."""

    def test_compute_communities_returns_real_partition(self):
        """dispatch.compute_communities no longer returns a zeros placeholder."""
        labels = compute_communities(ring_of_cliques(), method="louvain")

        assert labels.dtype == np.int64
        assert len(set(labels.tolist())) == 4

    def test_compute_communities_rejects_unknown_method(self):
        """An unimplemented method is an error, not a silent fallback."""
        with pytest.raises(ValueError):
            compute_communities(two_triangles(), method="infomap")

    def test_louvain_hooks_on_graph(self):
        """louvain_hooks works from the Graph type."""
        graph = Graph(
            edges=[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)],
            n_nodes=6,
            directed=False,
        )
        result = louvain_hooks(graph, seed=42)

        assert result["n_communities"] == 2
        assert result["modularity"] == pytest.approx(2 * (3 / 7 - 0.25))

    def test_louvain_hooks_weighted_graph(self):
        """Weighted Graph edges reach the kernel."""
        graph = Graph(
            edges=[(0, 1, 10.0), (1, 2, 0.1), (2, 3, 10.0), (3, 0, 0.1)],
            n_nodes=4,
            directed=False,
            weighted=True,
        )
        result = louvain_hooks(graph, seed=1)

        labels = result["communities"]
        assert result["n_communities"] == 2
        assert labels[0] == labels[1]

    def test_modularity_on_graph_matches_engine(self):
        """core.modularity agrees with the engine-level function."""
        graph = Graph(edges=[(0, 1), (1, 2)], n_nodes=3, directed=False)

        assert modularity(graph, np.array([0, 0, 1])) == pytest.approx(-0.125)

    def test_unknown_backend_is_rejected(self):
        """A typo in the backend name fails loudly."""
        graph = Graph(edges=[(0, 1)], n_nodes=2, directed=False)

        with pytest.raises(ValueError):
            louvain_hooks(graph, backend="julia")


@pytest.mark.skipif(
    pytest.importorskip("networkx", reason="networkx not installed") is None,
    reason="networkx not installed",
)
class TestAgainstNetworkX:
    """Our modularity must match NetworkX's definition exactly."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_modularity_matches_networkx(self, backend):
        """Same partition, same score as networkx.community.modularity."""
        import networkx as nx

        nx_graph = nx.karate_club_graph()
        edges = edge_list(list(nx_graph.edges()), n_nodes=nx_graph.number_of_nodes())
        labels = run_louvain(backend, edges, seed=42)["communities"]

        groups = {}
        for node, comm in enumerate(labels):
            groups.setdefault(int(comm), set()).add(node)
        expected = nx.community.modularity(nx_graph, list(groups.values()), weight=None)

        assert run_modularity(backend, edges, labels) == pytest.approx(expected)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_quality_is_competitive_with_networkx(self, backend):
        """Best-of-several-seeds reaches the known karate optimum (Q = 0.4198)."""
        import networkx as nx

        nx_graph = nx.karate_club_graph()
        edges = edge_list(list(nx_graph.edges()), n_nodes=nx_graph.number_of_nodes())

        best = max(run_louvain(backend, edges, seed=s)["modularity"] for s in range(10))
        assert best == pytest.approx(0.4198, abs=1e-3)
