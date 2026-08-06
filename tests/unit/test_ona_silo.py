"""Tests for netsmith.ona.silo — union-find silo detection."""

from netsmith.ona import SiloResult, detect_silos

# ── Fixtures ──────────────────────────────────────────────────────────────────

# Group A: alice, bob, carol — all talk to each other
# Group B: dave, eve, frank — all talk to each other
# Neither group talks to the other → silo on any shared cluster

EDGES_SILOED = [
    ("alice", "bob"),
    ("bob", "carol"),
    ("alice", "carol"),  # group A internal
    ("dave", "eve"),
    ("eve", "frank"),
    ("dave", "frank"),  # group B internal
]

ACTOR_CLUSTERS_SILOED = {
    "alice": ["finance"],
    "bob": ["finance"],
    "carol": ["finance"],
    "dave": ["finance"],
    "eve": ["finance"],
    "frank": ["finance"],
}

EDGES_CONNECTED = EDGES_SILOED + [("carol", "dave")]  # bridge between groups


# ── Basic detection ───────────────────────────────────────────────────────────


def test_detects_silo_when_groups_disconnected():
    results = detect_silos(EDGES_SILOED, ACTOR_CLUSTERS_SILOED)
    assert len(results) == 1
    assert results[0].cluster_id == "finance"
    assert results[0].component_count == 2


def test_no_silo_when_groups_connected():
    results = detect_silos(EDGES_CONNECTED, ACTOR_CLUSTERS_SILOED)
    assert results == []


def test_silo_result_fields():
    names = {"finance": "Finance & Accounting"}
    results = detect_silos(EDGES_SILOED, ACTOR_CLUSTERS_SILOED, names)
    r = results[0]
    assert isinstance(r, SiloResult)
    assert r.cluster_name == "Finance & Accounting"
    assert r.total_actors == 6
    assert isinstance(r.components, list)
    assert all(isinstance(c, frozenset) for c in r.components)


# ── Severity ─────────────────────────────────────────────────────────────────


def test_severity_high_three_components():
    edges = [("a", "b"), ("c", "d"), ("e", "f")]  # three isolated pairs
    clusters = {n: ["topic"] for n in "abcdef"}
    results = detect_silos(edges, clusters, min_component_size=2, min_components=2)
    assert results[0].severity == "high"


def test_severity_medium_two_small_groups():
    # 2 components totalling 6 actors: neither high threshold applies
    # (>= 3 components, or >= 10 actors), so this is medium.
    results = detect_silos(EDGES_SILOED, ACTOR_CLUSTERS_SILOED)
    assert results[0].severity == "medium"


def test_severity_high_on_actor_count_alone():
    # Only 2 components, but 10 actors crosses the size threshold.
    pairs = [("a", "b"), ("c", "d"), ("b", "c")]  # one component of 4
    chain = [("e", "f"), ("f", "g"), ("g", "h"), ("h", "i"), ("i", "j")]  # one of 6
    clusters = {n: ["topic"] for n in "abcdefghij"}

    results = detect_silos(pairs + chain, clusters)

    assert results[0].component_count == 2
    assert results[0].total_actors == 10
    assert results[0].severity == "high"


def test_severity_low_for_two_single_actor_groups():
    # Components of one are only kept when min_component_size allows it;
    # two of them total fewer than 4 actors, which is the low band.
    clusters = {"alice": ["finance"], "bob": ["finance"]}

    results = detect_silos([], clusters, min_component_size=1)

    assert results[0].component_count == 2
    assert results[0].total_actors == 2
    assert results[0].severity == "low"


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_empty_edges_no_silos():
    clusters = {"alice": ["finance"], "bob": ["finance"]}
    # No edges → each actor is its own component of size 1 → filtered by min_component_size=2
    results = detect_silos([], clusters)
    assert results == []


def test_single_actor_per_cluster_no_silo():
    clusters = {"alice": ["finance"], "bob": ["ops"]}
    results = detect_silos([("alice", "bob")], clusters)
    assert results == []


def test_multiple_clusters_independent():
    edges = [("alice", "bob"), ("carol", "dave")]
    clusters = {
        "alice": ["finance"],
        "bob": ["finance"],
        "carol": ["legal"],
        "dave": ["legal"],
    }
    # No shared cluster between disconnected groups
    results = detect_silos(edges, clusters)
    assert results == []


def test_results_sorted_by_severity_desc():
    # Two siloed clusters of different severity. The medium one is listed
    # first in the input, so the sort has to actually reorder them.
    edges = [
        ("a", "b"),
        ("c", "d"),  # topic_medium: 2 components, 4 actors
        ("e", "f"),
        ("g", "h"),
        ("i", "j"),  # topic_high: 3 components, 6 actors
    ]
    clusters = {name: ["topic_medium"] for name in "abcd"}
    clusters.update({name: ["topic_high"] for name in "efghij"})

    results = detect_silos(edges, clusters)

    assert [r.cluster_id for r in results] == ["topic_high", "topic_medium"]
    assert [r.severity for r in results] == ["high", "medium"]
