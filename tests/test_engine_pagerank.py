"""
Tests for the PageRank kernel.

The invariant that matters is that PageRank is a probability distribution:
every node holds some score and the scores sum to 1. The previous
implementation leaked mass on undirected edges and dangling nodes, which these
tests pin down.
"""

import numpy as np
import pytest

from netsmith.engine.contracts import EdgeList
from netsmith.engine.python import pagerank_python


def edge_list(edges, n_nodes, weights=None, directed=False):
    """Build an EdgeList from a list of (u, v) tuples."""
    u = np.array([e[0] for e in edges], dtype=np.int64)
    v = np.array([e[1] for e in edges], dtype=np.int64)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    return EdgeList(u=u, v=v, w=w, directed=directed, n_nodes=n_nodes)


def path_graph(n, directed=False):
    """A simple path 0-1-2-...-(n-1)."""
    return edge_list([(i, i + 1) for i in range(n - 1)], n_nodes=n, directed=directed)


class TestPageRankIsADistribution:
    """PageRank must always sum to 1 and stay positive."""

    @pytest.mark.parametrize("directed", [False, True])
    def test_path_graph(self, directed):
        """A path sums to 1 in both orientations."""
        pr = pagerank_python(path_graph(6, directed=directed))

        assert pr.sum() == pytest.approx(1.0)
        assert (pr > 0).all()

    def test_dangling_nodes(self):
        """A directed graph whose sinks have no out-edges keeps its mass."""
        # 0 -> 1 -> 2, and 3 -> 2. Nodes 2 and 3 dangle in different ways.
        edges = edge_list([(0, 1), (1, 2), (3, 2)], n_nodes=4, directed=True)
        pr = pagerank_python(edges)

        assert pr.sum() == pytest.approx(1.0)
        assert (pr > 0).all()

    def test_isolated_nodes(self):
        """Nodes with no edges at all still receive teleport mass."""
        edges = EdgeList(
            u=np.array([0], dtype=np.int64), v=np.array([1], dtype=np.int64), n_nodes=5
        )
        pr = pagerank_python(edges)

        assert pr.sum() == pytest.approx(1.0)
        assert (pr[2:] > 0).all()

    def test_edgeless_graph_is_uniform(self):
        """With no edges every node holds the same score."""
        edges = EdgeList(u=np.array([], dtype=np.int64), v=np.array([], dtype=np.int64), n_nodes=4)
        pr = pagerank_python(edges)

        assert pr.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(pr, 0.25)

    def test_self_loops(self):
        """Self-loops are ordinary links and do not break normalization."""
        edges = edge_list([(0, 0), (0, 1), (1, 2), (2, 2)], n_nodes=3)
        pr = pagerank_python(edges)

        assert pr.sum() == pytest.approx(1.0)

    def test_empty_graph(self):
        """A graph with no nodes returns an empty vector."""
        edges = EdgeList(u=np.array([], dtype=np.int64), v=np.array([], dtype=np.int64), n_nodes=0)
        assert len(pagerank_python(edges)) == 0


class TestPageRankValues:
    """Scores must reflect the structure of the graph."""

    def test_symmetric_graph_is_uniform(self):
        """Every node of a cycle is equivalent, so all scores are equal."""
        edges = edge_list([(0, 1), (1, 2), (2, 3), (3, 0)], n_nodes=4)
        pr = pagerank_python(edges)

        np.testing.assert_allclose(pr, 0.25, atol=1e-9)

    def test_hub_outranks_leaves(self):
        """The centre of a star holds the most score."""
        edges = edge_list([(0, 1), (0, 2), (0, 3), (0, 4)], n_nodes=5)
        pr = pagerank_python(edges)

        assert pr[0] > pr[1:].max()
        np.testing.assert_allclose(pr[1:], pr[1], atol=1e-9)

    def test_weights_shift_score(self):
        """A heavier link carries more score to its target."""
        edges = edge_list([(0, 1), (0, 2)], n_nodes=3, weights=[10.0, 1.0], directed=True)
        pr = pagerank_python(edges)

        assert pr[1] > pr[2]
        assert pr.sum() == pytest.approx(1.0)

    def test_alpha_zero_is_uniform_teleport(self):
        """With no damping the walk is pure teleportation."""
        pr = pagerank_python(path_graph(5), alpha=0.0)

        np.testing.assert_allclose(pr, 0.2)

    def test_every_node_gets_at_least_teleport_mass(self):
        """No node can score below (1 - alpha) / n."""
        alpha = 0.85
        pr = pagerank_python(path_graph(10), alpha=alpha)

        assert (pr >= (1 - alpha) / 10 - 1e-12).all()


class TestPageRankValidation:
    """Bad input fails loudly."""

    def test_negative_weights_are_rejected(self):
        edges = edge_list([(0, 1), (1, 2)], n_nodes=3, weights=[1.0, -1.0])

        with pytest.raises(ValueError):
            pagerank_python(edges)

    @pytest.mark.parametrize("alpha", [-0.1, 1.5])
    def test_alpha_out_of_range_is_rejected(self, alpha):
        with pytest.raises(ValueError):
            pagerank_python(path_graph(4), alpha=alpha)

    def test_non_convergence_warns(self):
        """Stopping at max_iter is reported, not returned silently."""
        with pytest.warns(RuntimeWarning, match="did not converge"):
            pagerank_python(path_graph(20), tol=0.0, max_iter=2)


class TestAgainstNetworkX:
    """Match the reference implementation."""

    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx(self, directed):
        """Same scores as networkx.pagerank on a random graph."""
        nx = pytest.importorskip("networkx")
        pytest.importorskip("scipy", reason="networkx.pagerank needs scipy")

        graph = nx.gnp_random_graph(30, 0.15, seed=11, directed=directed)
        edges = edge_list(list(graph.edges()), n_nodes=30, directed=directed)

        ours = pagerank_python(edges, tol=1e-12, max_iter=500)
        theirs = nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=500)

        np.testing.assert_allclose(ours, [theirs[i] for i in range(30)], atol=1e-8)

    def test_matches_networkx_weighted(self):
        """Weighted graphs agree too."""
        nx = pytest.importorskip("networkx")
        pytest.importorskip("scipy", reason="networkx.pagerank needs scipy")

        graph = nx.gnp_random_graph(20, 0.2, seed=3)
        rng = np.random.default_rng(0)
        for _, _, data in graph.edges(data=True):
            data["weight"] = float(rng.uniform(0.5, 5.0))

        pairs = list(graph.edges())
        edges = edge_list(
            pairs,
            n_nodes=20,
            weights=[graph[u][v]["weight"] for u, v in pairs],
        )

        ours = pagerank_python(edges, tol=1e-12, max_iter=500)
        theirs = nx.pagerank(graph, weight="weight", tol=1e-12, max_iter=500)

        np.testing.assert_allclose(ours, [theirs[i] for i in range(20)], atol=1e-8)
