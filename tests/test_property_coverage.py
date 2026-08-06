"""
Property and differential tests for the corners the example-based suite missed.

The bugs this codebase has actually shipped were not caught by hand-picked
examples — they were caught by checking invariants and by comparing against an
independent implementation on randomized input. These do that for the parts
that had neither: assortativity, the Gini coefficient, the Three-Es scores and
silo detection.
"""

import numpy as np
import pytest

from netsmith.core.graph import Graph
from netsmith.core.metrics import assortativity
from netsmith.core.paths import walk_metrics
from netsmith.core.stats import distributions
from netsmith.ona import Communication, detect_silos, gini_coefficient
from netsmith.ona.three_es import (
    energy_score,
    engagement_score,
    exploration_score,
    overall_score,
    score_team,
)


def communication(sender, receiver, minutes=30.0, kind="email", cross_team=False):
    """Build a Communication with the fields these scores read."""
    return Communication(
        sender_id=sender,
        receiver_id=receiver,
        duration_minutes=minutes,
        comm_type=kind,
        is_cross_team=cross_team,
    )


class TestAssortativity:
    """Correlation of an attribute across edges."""

    @pytest.mark.parametrize("seed", range(6))
    def test_matches_networkx_on_random_graphs(self, seed):
        nx = pytest.importorskip("networkx")

        source = nx.gnp_random_graph(60, 0.12, seed=seed)
        graph = Graph(edges=list(source.edges()), n_nodes=60)

        assert assortativity(graph) == pytest.approx(
            nx.degree_assortativity_coefficient(source), abs=1e-9
        )

    @pytest.mark.parametrize("seed", range(4))
    def test_matches_networkx_on_directed_graphs(self, seed):
        """Directed degree assortativity correlates out-degree with in-degree."""
        nx = pytest.importorskip("networkx")

        source = nx.gnp_random_graph(50, 0.1, seed=seed, directed=True)
        graph = Graph(edges=list(source.edges()), n_nodes=50, directed=True)

        assert assortativity(graph) == pytest.approx(
            nx.degree_assortativity_coefficient(source), abs=1e-9
        )

    def test_independent_of_how_each_edge_is_stored(self):
        """(u, v) and (v, u) are the same undirected edge.

        Correlating only the stored orientation made the answer depend on which
        way round each edge happened to be written down.
        """
        nx = pytest.importorskip("networkx")

        edges = list(nx.karate_club_graph().edges())
        rng = np.random.default_rng(0)
        flipped = [(v, u) if rng.integers(2) else (u, v) for u, v in edges]

        as_stored = assortativity(Graph(edges=edges, n_nodes=34))
        reversed_edges = assortativity(Graph(edges=flipped, n_nodes=34))

        assert as_stored == pytest.approx(reversed_edges, abs=1e-12)

    def test_a_perfectly_assortative_graph_scores_one(self):
        """Two cliques of different size: every edge joins equal degrees."""
        triangle = [(0, 1), (1, 2), (0, 2)]
        pair = [(3, 4)]

        assert assortativity(Graph(edges=triangle + pair, n_nodes=5)) == pytest.approx(1.0)

    def test_a_star_is_perfectly_disassortative(self):
        """Every edge joins the hub to a leaf, so the correlation is -1."""
        star = [(0, i) for i in range(1, 6)]

        assert assortativity(Graph(edges=star, n_nodes=6)) == pytest.approx(-1.0)

    def test_custom_attributes_are_used(self):
        """An attribute that agrees across every edge correlates perfectly."""
        edges = [(0, 1), (2, 3), (4, 5)]
        graph = Graph(edges=edges, n_nodes=6)
        matching = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])

        assert assortativity(graph, attribute=matching) == pytest.approx(1.0)

    def test_a_constant_attribute_has_no_correlation_to_report(self):
        """Zero variance means undefined, not zero."""
        graph = Graph(edges=[(0, 1), (1, 2)], n_nodes=3)

        assert np.isnan(assortativity(graph, attribute=np.ones(3)))

    def test_an_edgeless_graph_is_zero(self):
        assert assortativity(Graph(edges=[], n_nodes=4)) == 0.0


class TestGiniCoefficient:
    """Inequality of a distribution."""

    def reference_gini(self, values):
        """Textbook definition: mean absolute difference over twice the mean."""
        values = np.asarray(values, dtype=float)
        n = values.size
        differences = np.abs(values[:, None] - values[None, :]).sum()
        return differences / (2 * n * n * values.mean())

    @pytest.mark.parametrize("seed", range(8))
    def test_matches_the_textbook_definition(self, seed):
        """The implementation uses a sorted-cumulative form; check the algebra."""
        rng = np.random.default_rng(seed)
        values = rng.exponential(scale=3.0, size=int(rng.integers(2, 60)))

        assert gini_coefficient(values) == pytest.approx(self.reference_gini(values), abs=1e-9)

    def test_perfect_equality_is_zero(self):
        assert gini_coefficient([7.0] * 10) == pytest.approx(0.0)

    def test_one_taker_approaches_one(self):
        """With n people and one holding everything, Gini is (n-1)/n."""
        values = [0.0] * 9 + [100.0]

        assert gini_coefficient(values) == pytest.approx(0.9)

    def test_invariant_under_scaling(self):
        """Inequality is about proportions, not units."""
        values = [1.0, 2.0, 7.0, 11.0]

        assert gini_coefficient(values) == pytest.approx(
            gini_coefficient([v * 1000 for v in values])
        )

    def test_invariant_under_ordering(self):
        rng = np.random.default_rng(1)
        values = rng.exponential(size=30)

        assert gini_coefficient(values) == pytest.approx(gini_coefficient(rng.permutation(values)))

    def test_stays_within_bounds(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            values = rng.exponential(scale=5.0, size=int(rng.integers(1, 40)))
            assert 0.0 <= gini_coefficient(values) <= 1.0

    def test_empty_and_all_zero_are_zero(self):
        assert gini_coefficient([]) == 0.0
        assert gini_coefficient([0.0, 0.0, 0.0]) == 0.0


class TestThreeEsScores:
    """The Three-Es scores are bespoke formulas, so the checks are invariants."""

    @pytest.fixture
    def team(self):
        return ["alice", "bob", "carol"]

    @pytest.fixture
    def comms(self):
        return [
            communication("alice", "bob", 30.0, "face-to-face"),
            communication("bob", "alice", 20.0, "email"),
            communication("carol", "alice", 45.0, "face-to-face", cross_team=True),
            communication("alice", "carol", 15.0, "email"),
        ]

    @pytest.mark.parametrize("score", [energy_score, engagement_score, exploration_score])
    def test_scores_stay_within_zero_and_one_hundred(self, score, team):
        """Randomized traffic must never escape the documented range."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            traffic = [
                communication(
                    str(rng.choice(team)),
                    str(rng.choice(team)),
                    float(rng.exponential(60)),
                    str(rng.choice(["email", "face-to-face", "chat"])),
                    bool(rng.integers(2)),
                )
                for _ in range(int(rng.integers(0, 40)))
            ]
            value, detail = score(traffic, team)
            assert 0.0 <= value <= 100.0
            assert isinstance(detail, dict)

    @pytest.mark.parametrize("score", [energy_score, engagement_score, exploration_score])
    def test_order_of_communications_does_not_matter(self, score, team, comms):
        """These are aggregates; shuffling the log must not move them."""
        rng = np.random.default_rng(3)
        shuffled = list(rng.permutation(np.array(comms, dtype=object)))

        assert score(comms, team)[0] == score(shuffled, team)[0]

    @pytest.mark.parametrize("score", [energy_score, engagement_score, exploration_score])
    def test_empty_inputs_score_zero_with_a_full_detail_dict(self, score, team):
        """An early return must not change the shape of the result."""
        empty_value, empty_detail = score([], team)
        assert empty_value == 0.0

        populated_detail = score([communication("alice", "bob")], team)[1]
        assert set(empty_detail) == set(populated_detail), (
            "the detail dict must have the same keys either way; a caller "
            "reading detail['freq_normalized'] should not get a KeyError just "
            "because the team had a quiet month"
        )

    def test_energy_rises_with_face_to_face_contact(self, team):
        """Face-to-face carries half the energy score."""
        emails = [communication("alice", "bob", 30.0, "email") for _ in range(6)]
        in_person = [communication("alice", "bob", 30.0, "face-to-face") for _ in range(6)]

        assert energy_score(in_person, team)[0] > energy_score(emails, team)[0]

    def test_engagement_rises_when_participation_is_even(self, team):
        """Balance is measured by Gini, so a monopoly must score lower."""
        monopoly = [communication("alice", "bob") for _ in range(9)]
        even = [communication(sender, "bob") for sender in ["alice", "bob", "carol"] * 3]

        assert engagement_score(even, team)[0] > engagement_score(monopoly, team)[0]

    def test_exploration_rises_with_cross_team_contact(self, team):
        """Nobody reaching outside the team is the floor."""
        inside = [communication("alice", "bob", cross_team=False) for _ in range(5)]
        outside = [communication("alice", "bob", cross_team=True) for _ in range(5)]

        assert exploration_score(inside, team)[0] == 0.0
        assert exploration_score(outside, team)[0] > 0.0

    def test_overall_is_the_weighted_mean_of_the_three(self):
        assert overall_score(100.0, 100.0, 100.0) == pytest.approx(100.0)
        assert overall_score(0.0, 0.0, 0.0) == pytest.approx(0.0)
        assert overall_score(80.0, 60.0, 40.0) == pytest.approx(80 * 0.35 + 60 * 0.40 + 40 * 0.25)

    def test_overall_weights_are_a_partition(self):
        """Weights that do not sum to 1 would silently rescale every score."""
        import inspect

        from netsmith.ona.three_es import overall_score as scorer

        default = inspect.signature(scorer).parameters["weights"].default
        assert sum(default) == pytest.approx(1.0)

    def test_score_team_agrees_with_the_individual_scores(self, team, comms):
        """The convenience wrapper must not compute anything different."""
        result = score_team(comms, team)

        assert result.energy == energy_score(comms, team)[0]
        assert result.engagement == engagement_score(comms, team)[0]
        assert result.exploration == exploration_score(comms, team)[0]
        assert result.overall == overall_score(result.energy, result.engagement, result.exploration)


class TestSiloDetection:
    """Union-find over the actors sharing each topic."""

    def test_components_match_a_reference_traversal(self):
        """Cross-check the union-find against connected components."""
        nx = pytest.importorskip("networkx")

        rng = np.random.default_rng(4)
        actors = [f"a{i}" for i in range(40)]
        edges = [(actors[int(rng.integers(40))], actors[int(rng.integers(40))]) for _ in range(45)]
        clusters = {actor: ["topic"] for actor in actors}

        results = detect_silos(edges, clusters, min_component_size=1, min_components=1)

        reference = nx.Graph()
        reference.add_nodes_from(actors)
        reference.add_edges_from(edges)
        expected = {frozenset(c) for c in nx.connected_components(reference)}

        assert {frozenset(c) for c in results[0].components} == expected

    def test_actor_order_does_not_change_the_result(self):
        """Union-find must not depend on the order actors are seen in."""
        rng = np.random.default_rng(5)
        edges = [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")]
        clusters = {name: ["topic"] for name in "abcdefgh"}

        first = detect_silos(edges, clusters)
        shuffled_clusters = {name: ["topic"] for name in rng.permutation(list("abcdefgh"))}
        second = detect_silos(
            list(rng.permutation(np.array(edges, dtype=object))), shuffled_clusters
        )

        assert first[0].component_count == second[0].component_count
        assert {frozenset(c) for c in first[0].components} == {
            frozenset(c) for c in second[0].components
        }

    def test_total_actors_equals_the_components_it_reports(self):
        """The summary numbers must agree with the components themselves."""
        edges = [("a", "b"), ("c", "d"), ("e", "f")]
        clusters = {name: ["topic"] for name in "abcdef"}

        result = detect_silos(edges, clusters)[0]

        assert result.component_count == len(result.components)
        assert result.total_actors == sum(len(c) for c in result.components)

    def test_a_connected_cluster_is_never_a_silo(self):
        """One component cannot be siloed from itself, at any size."""
        rng = np.random.default_rng(6)
        actors = [f"n{i}" for i in range(20)]
        # A path guarantees a single component.
        edges = [(actors[i], actors[i + 1]) for i in range(19)]
        rng.shuffle(edges)

        assert detect_silos(edges, {a: ["topic"] for a in actors}) == []


class TestUnimplementedStaysUnimplemented:
    """Planned features must announce themselves, not return empty results."""

    def test_walk_metrics_raises(self):
        with pytest.raises(NotImplementedError):
            walk_metrics(Graph(edges=[(0, 1)], n_nodes=2))

    def test_distributions_raises(self):
        with pytest.raises(NotImplementedError):
            distributions(np.arange(10.0))
