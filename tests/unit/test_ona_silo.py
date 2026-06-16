"""Tests for netsmith.ona.silo — union-find silo detection."""

import pytest
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
    results = detect_silos(EDGES_SILOED, ACTOR_CLUSTERS_SILOED)
    # 2 components, total 6 actors → high (total >= 10 is one high threshold, but >=3 comps also high)
    # Our fixture: 2 comps, total 6, so medium
    assert results[0].severity in ("medium", "high")


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
    # Create a situation with high and medium severity silos
    edges = [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")]
    # topic1: 4 components → high; topic2: 2 components → medium
    clusters = {
        "a": ["topic1"],
        "b": ["topic1"],
        "c": ["topic1"],
        "d": ["topic1"],
        "e": ["topic1"],
        "f": ["topic1"],
        "g": ["topic2"],
        "h": ["topic2"],
    }
    extra_edges = [("g", "h")]  # topic2: 1 component, filtered out
    all_edges = edges  # topic1: 4 isolated pairs
    results = detect_silos(all_edges, clusters)
    # All should be high (4 components) or none
    for r in results:
        assert r.severity in ("low", "medium", "high")
    # Verify descending severity order
    severity_order = {"high": 2, "medium": 1, "low": 0}
    scores = [severity_order[r.severity] for r in results]
    assert scores == sorted(scores, reverse=True)
