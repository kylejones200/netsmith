"""
The library must not need NetworkX.

NetworkX stays in the dev dependencies because it is a useful independent
reference to test against, and `Graph.as_networkx()` exists for interop. What
must not happen is a public function reaching for it to compute something.
"""

import builtins
import importlib
import sys

import numpy as np
import pytest

from netsmith.core.community import label_propagation_hooks, louvain_hooks, modularity
from netsmith.core.graph import Graph
from netsmith.core.metrics import clustering, k_core
from netsmith.core.nulls import null_models
from netsmith.core.paths import shortest_paths


@pytest.fixture
def without_networkx(monkeypatch):
    """Make any `import networkx` fail, as on an install that lacks it."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "networkx" or name.startswith("networkx."):
            raise ImportError("networkx is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    for module in [m for m in sys.modules if m.startswith("networkx")]:
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def graph():
    """Two triangles joined by a bridge, with a pendant node."""
    return Graph(
        edges=[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3), (5, 6)],
        n_nodes=7,
        directed=False,
    )


def test_the_guard_actually_bites(without_networkx):
    """If this fails the other tests here prove nothing."""
    with pytest.raises(ImportError):
        importlib.import_module("networkx")


class TestEverythingWorksWithoutNetworkX:
    """Each of these used to route through NetworkX."""

    def test_k_core(self, graph, without_networkx):
        cores = k_core(graph)
        assert list(cores) == [2, 2, 2, 2, 2, 2, 1]

    def test_louvain(self, graph, without_networkx):
        result = louvain_hooks(graph, seed=42)
        assert result["n_communities"] >= 2
        assert result["modularity"] > 0

    def test_modularity(self, graph, without_networkx):
        score = modularity(graph, np.array([0, 0, 0, 1, 1, 1, 1]))
        assert -0.5 <= score <= 1.0

    def test_label_propagation(self, graph, without_networkx):
        result = label_propagation_hooks(graph, seed=7)
        assert len(result["communities"]) == graph.n_nodes
        assert result["n_communities"] >= 1

    def test_clustering(self, graph, without_networkx):
        assert len(clustering(graph)) == graph.n_nodes

    def test_shortest_paths(self, graph, without_networkx):
        assert shortest_paths(graph, source=0)[1] == 1

    @pytest.mark.parametrize("method", ["configuration", "erdos_renyi", "degree_preserving"])
    def test_null_models(self, method, without_networkx):
        source = Graph(
            edges=[(i, (i + 1) % 30) for i in range(30)] + [(i, (i + 3) % 30) for i in range(30)],
            n_nodes=30,
            directed=False,
        )

        result = null_models(source, method=method, n_samples=3, seed=1)

        assert result["n_samples"] == 3
        assert all(g.n_nodes == 30 for g in result["graphs"])

    def test_betweenness(self, graph, without_networkx):
        from netsmith.api import betweenness

        scores = betweenness(graph)
        assert len(scores) == graph.n_nodes
        assert (scores >= 0).all()

    def test_pagerank(self, graph, without_networkx):
        from netsmith.api import pagerank

        assert pagerank(graph.to_edge_list()).sum() == pytest.approx(1.0)


class TestInteropStillExists:
    """as_networkx() is interop, not a dependency: it fails clearly."""

    def test_as_networkx_raises_a_useful_message(self, graph, without_networkx):
        with pytest.raises(ImportError, match="NetworkX is required"):
            graph.as_networkx()

    def test_as_networkx_works_when_installed(self, graph):
        pytest.importorskip("networkx")

        nx_graph = graph.as_networkx()

        assert nx_graph.number_of_nodes() == graph.n_nodes
        assert nx_graph.number_of_edges() == graph.n_edges
