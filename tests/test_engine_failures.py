"""
Tests for the failure policy: nothing is skipped, defaulted, or substituted.

Each of these covers a case that used to return a plausible number for a graph
or a backend the caller never asked about.
"""

import numpy as np
import pytest

from netsmith.core.graph import Graph
from netsmith.engine.contracts import EdgeList
from netsmith.engine.dispatch import (
    compute_betweenness,
    compute_communities,
    compute_degree,
    compute_pagerank,
    compute_shortest_paths,
)
from netsmith.engine.rust import _RUST_AVAILABLE
from netsmith.exceptions import BackendError, ValidationError

requires_rust = pytest.mark.skipif(not _RUST_AVAILABLE, reason="netsmith_rs not built")

BACKENDS = ["python"] + (["rust"] if _RUST_AVAILABLE else [])


def edge_list(edges, n_nodes, weights=None, directed=False):
    """Build an EdgeList from a list of (u, v) tuples."""
    u = np.array([e[0] for e in edges], dtype=np.int64)
    v = np.array([e[1] for e in edges], dtype=np.int64)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    return EdgeList(u=u, v=v, w=w, directed=directed, n_nodes=n_nodes)


class TestOutOfRangeEdgesAreRejected:
    """An edge naming a node that does not exist is an error, not a dropped edge."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_degree(self, backend):
        edges = edge_list([(0, 1), (1, 9)], n_nodes=3)

        with pytest.raises(ValueError, match="9"):
            compute_degree(edges, backend=backend)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_communities(self, backend):
        edges = edge_list([(0, 1), (1, 9)], n_nodes=3)

        with pytest.raises(ValueError):
            compute_communities(edges, backend=backend)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_betweenness(self, backend):
        edges = edge_list([(0, 1), (1, 9)], n_nodes=3)

        with pytest.raises(ValueError):
            compute_betweenness(edges, backend=backend)

    @requires_rust
    def test_the_error_names_the_offending_edge(self):
        """The message has to be actionable, not just 'invalid input'."""
        import netsmith_rs

        with pytest.raises(ValueError) as excinfo:
            netsmith_rs.degree_rust(3, np.array([[0, 1], [1, 9]], dtype=np.uintp), False)

        message = str(excinfo.value)
        assert "edge 1" in message
        assert "(1, 9)" in message
        assert "[0, 3)" in message


class TestBackendIsNeverSubstituted:
    """backend="rust" must not quietly return a Python result."""

    @requires_rust
    def test_missing_rust_kernel_raises_instead_of_falling_back(self):
        """There is no Rust PageRank; asking for one is an error."""
        edges = edge_list([(0, 1), (1, 2)], n_nodes=3)

        with pytest.raises(BackendError, match="pagerank_rust"):
            compute_pagerank(edges, backend="rust")

    @requires_rust
    def test_auto_still_falls_back_for_a_missing_kernel(self):
        """ "auto" is allowed to choose; that is what it means."""
        edges = edge_list([(0, 1), (1, 2)], n_nodes=3)

        scores = compute_pagerank(edges, backend="auto")

        assert scores.sum() == pytest.approx(1.0)

    def test_unknown_backend_name_is_rejected(self):
        edges = edge_list([(0, 1)], n_nodes=2)

        with pytest.raises(ValueError, match="unknown backend"):
            compute_degree(edges, backend="julia")


class TestWeightsAreNotSilentlyIgnored:
    """A weight argument either changes the answer or raises."""

    def test_weighted_shortest_paths_raise_rather_than_return_hops(self):
        # 0->2 direct is 100 long, 0->1->2 is 2. Hop counts would say the
        # direct edge is nearer, which is not what weight= was asking for.
        edges = edge_list([(0, 1), (1, 2), (0, 2)], n_nodes=3, weights=[1.0, 1.0, 100.0])

        with pytest.raises(NotImplementedError, match="hop counts"):
            compute_shortest_paths(edges, source=0, weight="weight")

    def test_unweighted_shortest_paths_still_work(self):
        edges = edge_list([(0, 1), (1, 2)], n_nodes=3)

        assert list(compute_shortest_paths(edges, source=0)[:3]) == [0, 1, 2]

    def test_short_weight_arrays_are_rejected_not_padded(self):
        """A weights array that does not cover every edge is an error.

        EdgeList catches this at construction, the earliest possible point.
        """
        with pytest.raises(ValueError):
            EdgeList(
                u=np.array([0, 1], dtype=np.int64),
                v=np.array([1, 2], dtype=np.int64),
                w=np.array([1.0]),
                n_nodes=3,
            )


class TestMalformedGraphsAreRejected:
    """A graph that cannot be read is an error, not an empty graph."""

    def test_construction_rejects_a_weighted_graph_without_weights(self):
        """The constructor is the first line of defence."""
        with pytest.raises(ValidationError, match="weighted"):
            Graph(edges=[(0, 1), (1, 2)], n_nodes=3, directed=False, weighted=True)

    def test_edges_mutated_after_construction_still_raise(self):
        """Reading a broken edge list used to warn and return an empty graph,
        turning every downstream metric into a confident wrong answer."""
        graph = Graph(edges=[(0, 1, 1.0), (1, 2, 1.0)], n_nodes=3, directed=False, weighted=True)
        graph.edges = [(0, 1), (1, 2)]  # weights dropped after validation

        with pytest.raises(ValidationError, match="malformed"):
            graph.edges_coo()

    def test_metrics_on_a_malformed_graph_do_not_return_zeros(self):
        """The failure surfaces through anything that reads the edges."""
        graph = Graph(edges=[(0, 1, 1.0), (1, 2, 1.0)], n_nodes=3, directed=False, weighted=True)
        graph.edges = [(0, 1), (1, 2)]

        with pytest.raises(ValidationError):
            graph.to_edge_list()
