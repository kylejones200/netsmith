"""
Tests for betweenness centrality across the Python and Rust backends.

Every behavioural test runs against both backends, so the parallel Rust kernel
and the serial Python fallback cannot drift apart.
"""

import numpy as np
import pytest

from netsmith.api.metrics import betweenness as api_betweenness
from netsmith.core.graph import Graph
from netsmith.core.metrics import centrality
from netsmith.engine.contracts import EdgeList
from netsmith.engine.dispatch import compute_betweenness
from netsmith.engine.python import betweenness_python
from netsmith.engine.rust import _RUST_AVAILABLE, betweenness_rust

BACKENDS = ["python"] + (["rust"] if _RUST_AVAILABLE else [])

requires_rust = pytest.mark.skipif(not _RUST_AVAILABLE, reason="netsmith_rs not built")


def run(backend, edges, **kwargs):
    """Compute betweenness on the named backend."""
    if backend == "rust":
        return betweenness_rust(edges, **kwargs)
    return betweenness_python(edges, **kwargs)


def edge_list(edges, n_nodes, weights=None, directed=False):
    """Build an EdgeList from a list of (u, v) tuples."""
    u = np.array([e[0] for e in edges], dtype=np.int64)
    v = np.array([e[1] for e in edges], dtype=np.int64)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    return EdgeList(u=u, v=v, w=w, directed=directed, n_nodes=n_nodes)


def path(n, directed=False):
    """Path 0-1-2-...-(n-1)."""
    return edge_list([(i, i + 1) for i in range(n - 1)], n_nodes=n, directed=directed)


def star(leaves):
    """Star with node 0 at the centre."""
    return edge_list([(0, i) for i in range(1, leaves + 1)], n_nodes=leaves + 1)


@pytest.mark.parametrize("backend", BACKENDS)
class TestBetweenness:
    """Behaviour shared by every backend."""

    def test_star_centre_takes_everything(self, backend):
        """Every path between two leaves runs through the centre."""
        scores = run(backend, star(4))

        assert scores[0] == pytest.approx(1.0)
        np.testing.assert_allclose(scores[1:], 0.0)

    def test_path_middle_beats_the_ends(self, backend):
        """On a 5-path the centre brokers the most pairs; the ends broker none."""
        scores = run(backend, path(5), normalized=False)

        # Node 2 separates {0,1} from {3,4}: 2 * 2 = 4 pairs.
        np.testing.assert_allclose(scores, [0.0, 3.0, 4.0, 3.0, 0.0])

    def test_ties_split_credit(self, backend):
        """Two equally short routes each carry half of the pair."""
        scores = run(backend, edge_list([(0, 1), (1, 2), (2, 3), (3, 0)], n_nodes=4))

        np.testing.assert_allclose(scores, scores[0])
        assert scores[0] > 0

    def test_complete_graph_has_no_brokers(self, backend):
        """When everyone is adjacent, nobody sits between anyone."""
        pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        scores = run(backend, edge_list(pairs, n_nodes=5))

        np.testing.assert_allclose(scores, 0.0, atol=1e-12)

    def test_disconnected_components_count_separately(self, backend):
        """Unreachable pairs contribute nothing rather than blowing up."""
        edges = edge_list([(0, 1), (1, 2), (3, 4), (4, 5)], n_nodes=6)
        scores = run(backend, edges, normalized=False)

        assert scores[1] == pytest.approx(1.0)
        assert scores[4] == pytest.approx(1.0)
        assert scores[0] == pytest.approx(0.0)

    def test_isolated_node_scores_zero(self, backend):
        """A node with no edges brokers nothing."""
        scores = run(backend, edge_list([(0, 1), (1, 2)], n_nodes=4))

        assert scores[3] == pytest.approx(0.0)

    def test_weights_are_distances(self, backend):
        """A heavy direct edge can make the detour the shortest route."""
        # Triangle where 0-2 is long: unweighted 0->2 is one hop, so node 1
        # brokers nothing; weighted, the route through 1 is shorter.
        edges = [(0, 1), (1, 2), (0, 2)]
        unweighted = run(backend, edge_list(edges, n_nodes=3), normalized=False)
        weighted = run(
            backend,
            edge_list(edges, n_nodes=3, weights=[1.0, 1.0, 10.0]),
            normalized=False,
        )

        assert unweighted[1] == pytest.approx(0.0)
        assert weighted[1] == pytest.approx(1.0)

    def test_weights_can_be_ignored_explicitly(self, backend):
        """weight=False scores a weighted graph by hop count."""
        edges = edge_list([(0, 1), (1, 2), (0, 2)], n_nodes=3, weights=[1.0, 1.0, 10.0])

        assert run(backend, edges, weight=False, normalized=False)[1] == pytest.approx(0.0)

    def test_direction_is_respected(self, backend):
        """A directed chain brokers only the pairs its arrows allow."""
        forward = edge_list([(0, 1), (1, 2)], n_nodes=3, directed=True)
        scores = run(backend, forward, normalized=False)

        assert scores[1] == pytest.approx(1.0)

        # Reversing one arrow leaves no path from 0 to 2 at all.
        broken = edge_list([(0, 1), (2, 1)], n_nodes=3, directed=True)
        assert run(backend, broken, normalized=False)[1] == pytest.approx(0.0)

    def test_normalized_scores_stay_within_unit_range(self, backend):
        """Normalization puts every score in [0, 1]."""
        scores = run(backend, path(7))

        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_self_loops_and_parallel_edges_are_ignored(self, backend):
        """Neither can lie on a shortest path, so neither changes the scores."""
        clean = path(5)
        messy = edge_list(
            [(i, i + 1) for i in range(4)] + [(2, 2), (1, 2)],
            n_nodes=5,
        )

        np.testing.assert_allclose(run(backend, clean), run(backend, messy))

    def test_tiny_graphs_are_handled(self, backend):
        """No node count is too small to score."""
        empty = EdgeList(u=np.array([], dtype=np.int64), v=np.array([], dtype=np.int64), n_nodes=0)
        assert len(run(backend, empty)) == 0

        two = run(backend, edge_list([(0, 1)], n_nodes=2))
        np.testing.assert_allclose(two, 0.0)

    def test_non_positive_weights_are_rejected(self, backend):
        """Zero and negative distances break shortest paths, so refuse them."""
        with pytest.raises(ValueError):
            run(backend, edge_list([(0, 1), (1, 2)], n_nodes=3, weights=[1.0, 0.0]))
        with pytest.raises(ValueError):
            run(backend, edge_list([(0, 1), (1, 2)], n_nodes=3, weights=[1.0, -2.0]))

    def test_weight_true_without_weights_is_rejected(self, backend):
        """Asking for weighted scores on an unweighted graph is an error."""
        with pytest.raises(ValueError):
            run(backend, path(4), weight=True)


@requires_rust
class TestBackendAgreement:
    """The parallel kernel and the serial fallback must agree."""

    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_backends_agree_on_random_graphs(self, directed, weighted):
        """Same scores from both backends on a graph big enough to be split."""
        nx = pytest.importorskip("networkx")

        graph = nx.gnp_random_graph(60, 0.1, seed=5, directed=directed)
        pairs = list(graph.edges())
        rng = np.random.default_rng(1)
        weights = rng.uniform(0.5, 5.0, size=len(pairs)) if weighted else None
        edges = edge_list(pairs, n_nodes=60, weights=weights, directed=directed)

        np.testing.assert_allclose(betweenness_rust(edges), betweenness_python(edges), atol=1e-9)


class TestAgainstNetworkX:
    """Match the reference implementation."""

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("normalized", [True, False])
    def test_matches_networkx_undirected(self, backend, normalized):
        """Karate club, the standard betweenness benchmark."""
        nx = pytest.importorskip("networkx")

        graph = nx.karate_club_graph()
        edges = edge_list(list(graph.edges()), n_nodes=graph.number_of_nodes())

        ours = run(backend, edges, normalized=normalized)
        theirs = nx.betweenness_centrality(graph, normalized=normalized, weight=None)

        np.testing.assert_allclose(ours, [theirs[i] for i in range(len(ours))], atol=1e-9)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_matches_networkx_directed(self, backend):
        """Direction handling matches, including the normalization branch."""
        nx = pytest.importorskip("networkx")

        graph = nx.gnp_random_graph(40, 0.1, seed=3, directed=True)
        edges = edge_list(list(graph.edges()), n_nodes=40, directed=True)

        ours = run(backend, edges)
        theirs = nx.betweenness_centrality(graph, normalized=True, weight=None)

        np.testing.assert_allclose(ours, [theirs[i] for i in range(40)], atol=1e-9)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_matches_networkx_weighted(self, backend):
        """Weighted betweenness agrees, ties included."""
        nx = pytest.importorskip("networkx")

        graph = nx.gnp_random_graph(40, 0.12, seed=8)
        rng = np.random.default_rng(2)
        for _, _, data in graph.edges(data=True):
            data["weight"] = float(rng.uniform(0.5, 5.0))

        pairs = list(graph.edges())
        edges = edge_list(pairs, n_nodes=40, weights=[graph[u][v]["weight"] for u, v in pairs])

        ours = run(backend, edges)
        theirs = nx.betweenness_centrality(graph, normalized=True, weight="weight")

        np.testing.assert_allclose(ours, [theirs[i] for i in range(40)], atol=1e-9)


class TestPublicAPI:
    """The public entry points reach the kernel."""

    def test_centrality_method_betweenness(self):
        """core.metrics.centrality no longer raises NotImplementedError."""
        graph = Graph(edges=[(0, 1), (1, 2), (2, 3), (3, 4)], n_nodes=5, directed=False)

        scores = centrality(graph, method="betweenness")

        np.testing.assert_allclose(scores, [0.0, 0.5, 2 / 3, 0.5, 0.0])

    def test_api_betweenness_single_node(self):
        """Asking for one node returns a float."""
        graph = Graph(edges=[(0, 1), (1, 2), (2, 3), (3, 4)], n_nodes=5, directed=False)

        assert api_betweenness(graph, node=2) == pytest.approx(2 / 3)

    def test_api_betweenness_rejects_bad_node(self):
        """An out-of-range node is a validation error."""
        from netsmith.exceptions import ValidationError

        graph = Graph(edges=[(0, 1)], n_nodes=2, directed=False)

        with pytest.raises(ValidationError):
            api_betweenness(graph, node=7)

    def test_weighted_graph_uses_its_weights(self):
        """A weighted Graph reaches the kernel with its weights."""
        graph = Graph(
            edges=[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 10.0)],
            n_nodes=3,
            directed=False,
            weighted=True,
        )

        assert centrality(graph, method="betweenness", normalized=False)[1] == pytest.approx(1.0)

    def test_dispatch_backends_agree(self):
        """compute_betweenness returns the same thing whichever backend runs."""
        edges = path(6)

        np.testing.assert_allclose(
            compute_betweenness(edges, backend="python"),
            compute_betweenness(edges, backend="auto"),
            atol=1e-12,
        )

    def test_unknown_centrality_method_still_raises(self):
        """Unimplemented methods stay unimplemented rather than guessing."""
        graph = Graph(edges=[(0, 1)], n_nodes=2, directed=False)

        with pytest.raises(NotImplementedError):
            centrality(graph, method="eigenvector")
