"""
Tests for multi-source shortest paths.

The point of the multi-source entry point is that it builds the adjacency list
once, so these check it agrees with the per-source path it is meant to replace,
on both backends.
"""

import numpy as np
import pytest

from netsmith.api import UNREACHABLE, shortest_paths, shortest_paths_multi
from netsmith.core.graph import Graph
from netsmith.engine.contracts import EdgeList
from netsmith.engine.dispatch import compute_shortest_paths_multi
from netsmith.engine.python import shortest_paths_multi_python
from netsmith.engine.rust import _RUST_AVAILABLE, shortest_paths_multi_rust
from netsmith.exceptions import ValidationError

BACKENDS = ["python"] + (["rust"] if _RUST_AVAILABLE else [])

requires_rust = pytest.mark.skipif(not _RUST_AVAILABLE, reason="netsmith_rs not built")


def run(backend, edges, sources):
    """Compute multi-source distances on the named backend."""
    if backend == "rust":
        return shortest_paths_multi_rust(edges, sources)
    return shortest_paths_multi_python(edges, sources)


def edge_list(edges, n_nodes, directed=False):
    """Build an EdgeList from a list of (u, v) tuples."""
    u = np.array([e[0] for e in edges], dtype=np.int64)
    v = np.array([e[1] for e in edges], dtype=np.int64)
    return EdgeList(u=u, v=v, directed=directed, n_nodes=n_nodes)


def two_triangles():
    """Two triangles with no edge between them, plus an isolated node."""
    return edge_list([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], n_nodes=7)


@pytest.mark.parametrize("backend", BACKENDS)
class TestMultiSource:
    """Behaviour shared by every backend."""

    def test_rows_match_single_source_calls(self, backend):
        """The whole point: same answers, less setup."""
        edges = two_triangles()
        sources = [0, 3, 5, 1]

        many = run(backend, edges, sources)

        assert many.shape == (len(sources), edges.n_nodes)
        for row, source in enumerate(sources):
            one = shortest_paths(
                Graph(
                    edges=[(int(u), int(v)) for u, v in zip(edges.u, edges.v)],
                    n_nodes=edges.n_nodes,
                ),
                source=source,
                backend=backend,
            )
            np.testing.assert_array_equal(many[row], one)

    def test_hop_counts_along_a_path(self, backend):
        edges = edge_list([(0, 1), (1, 2), (2, 3)], n_nodes=4)

        distances = run(backend, edges, [0])

        np.testing.assert_array_equal(distances[0], [0, 1, 2, 3])

    def test_unreachable_nodes_use_the_shared_sentinel(self, backend):
        """Both backends must mark unreachable the same way.

        They did not: the Rust kernel returned usize::MAX as uint64 while the
        Python one returned int64 max, so `== UNREACHABLE` missed on one.
        """
        edges = two_triangles()

        distances = run(backend, edges, [0])

        assert distances.dtype == np.int64
        assert distances[0][6] == UNREACHABLE
        assert distances[0][3] == UNREACHABLE

    def test_direction_is_respected(self, backend):
        forward = edge_list([(0, 1), (1, 2)], n_nodes=3, directed=True)

        distances = run(backend, forward, [0, 2])

        assert distances[0][2] == 2
        assert distances[1][0] == UNREACHABLE

    def test_no_sources_yields_no_rows(self, backend):
        edges = two_triangles()

        assert run(backend, edges, []).shape == (0, edges.n_nodes)

    def test_repeated_sources_are_allowed(self, backend):
        """Asking twice is not an error; it just costs a second sweep."""
        edges = two_triangles()

        distances = run(backend, edges, [1, 1])

        np.testing.assert_array_equal(distances[0], distances[1])

    def test_source_outside_the_graph_is_an_error(self, backend):
        """An all-unreachable row cannot say "there is no such node"."""
        edges = two_triangles()

        with pytest.raises(ValueError):
            run(backend, edges, [0, 99])

    def test_out_of_range_edges_are_rejected(self, backend):
        edges = edge_list([(0, 1), (1, 9)], n_nodes=3)

        with pytest.raises(ValueError):
            run(backend, edges, [0])


@requires_rust
class TestBackendAgreement:
    """The parallel Rust sweep and the serial Python one must agree."""

    @pytest.mark.parametrize("directed", [False, True])
    def test_backends_agree_on_a_random_graph(self, directed):
        nx = pytest.importorskip("networkx")

        graph = nx.gnp_random_graph(80, 0.05, seed=4, directed=directed)
        pairs = list(graph.edges())
        edges = edge_list(pairs, n_nodes=80, directed=directed)
        sources = list(range(0, 80, 7))

        np.testing.assert_array_equal(
            shortest_paths_multi_rust(edges, sources),
            shortest_paths_multi_python(edges, sources),
        )


class TestPublicAPI:
    """The public entry point behaves like the engine one."""

    def test_api_returns_one_row_per_source(self):
        graph = Graph(edges=[(0, 1), (1, 2), (2, 3)], n_nodes=4)

        distances = shortest_paths_multi(graph, [0, 3])

        assert distances.shape == (2, 4)
        np.testing.assert_array_equal(distances[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(distances[1], [3, 2, 1, 0])

    def test_api_rejects_an_out_of_range_source(self):
        graph = Graph(edges=[(0, 1)], n_nodes=2)

        with pytest.raises(ValidationError, match="out of range"):
            shortest_paths_multi(graph, [0, 5])

    def test_dispatch_backends_agree(self):
        edges = edge_list([(0, 1), (1, 2), (2, 3)], n_nodes=4)

        np.testing.assert_array_equal(
            compute_shortest_paths_multi(edges, [0, 2], backend="python"),
            compute_shortest_paths_multi(edges, [0, 2], backend="auto"),
        )

    def test_matches_networkx_hop_counts(self):
        nx = pytest.importorskip("networkx")

        source_graph = nx.karate_club_graph()
        graph = Graph(edges=list(source_graph.edges()), n_nodes=source_graph.number_of_nodes())
        sources = [0, 5, 33]

        distances = shortest_paths_multi(graph, sources)

        for row, source in enumerate(sources):
            expected = nx.single_source_shortest_path_length(source_graph, source)
            for node, hops in expected.items():
                assert distances[row][node] == hops
