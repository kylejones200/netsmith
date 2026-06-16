"""Tests for netsmith.ona.three_es — pure scoring functions."""

import pytest
from netsmith.ona import (
    Communication,
    ThreeEsResult,
    energy_score,
    engagement_score,
    exploration_score,
    gini_coefficient,
    overall_score,
    score_team,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

MEMBERS = ["alice", "bob", "carol", "dave"]


def _comms(pairs: list[tuple], *, cross: bool = False, comm_type: str = "email", dur: float = 10.0):
    return [
        Communication(
            sender_id=a,
            receiver_id=b,
            duration_minutes=dur,
            comm_type=comm_type,
            is_cross_team=cross,
        )
        for a, b in pairs
    ]


# ── gini_coefficient ──────────────────────────────────────────────────────────


def test_gini_perfect_equality():
    assert gini_coefficient([10, 10, 10, 10]) == pytest.approx(0.0, abs=1e-9)


def test_gini_total_inequality():
    # All value concentrated in one — approaches 1.0 as n grows
    g = gini_coefficient([0, 0, 0, 100])
    assert 0.7 < g <= 1.0


def test_gini_empty():
    assert gini_coefficient([]) == 0.0


def test_gini_all_zeros():
    assert gini_coefficient([0, 0, 0]) == 0.0


# ── energy_score ─────────────────────────────────────────────────────────────


def test_energy_no_comms():
    score, detail = energy_score([], MEMBERS, days=30)
    assert score == 0.0
    assert detail["total_comms"] == 0


def test_energy_face_to_face_boosts_score():
    email_comms = _comms([("alice", "bob")] * 10, comm_type="email")
    ftf_comms = _comms([("alice", "bob")] * 10, comm_type="face-to-face")
    e_email, _ = energy_score(email_comms, MEMBERS)
    e_ftf, _ = energy_score(ftf_comms, MEMBERS)
    assert e_ftf > e_email


def test_energy_bounded_0_100():
    # Even with massive volume the score caps at 100
    huge = _comms([("alice", "bob")] * 10_000, comm_type="face-to-face", dur=999.0)
    score, _ = energy_score(huge, MEMBERS, days=1)
    assert 0.0 <= score <= 100.0


def test_energy_no_members():
    comms = _comms([("alice", "bob")])
    score, _ = energy_score(comms, [])
    assert score == 0.0


# ── engagement_score ─────────────────────────────────────────────────────────


def test_engagement_all_active_balanced():
    # Everyone sends roughly the same amount
    comms = _comms(
        [
            ("alice", "bob"),
            ("bob", "carol"),
            ("carol", "dave"),
            ("dave", "alice"),
            ("alice", "carol"),
            ("bob", "dave"),
        ]
    )
    score, detail = engagement_score(comms, MEMBERS)
    assert score > 50.0
    assert detail["participation_rate"] == 1.0


def test_engagement_one_sender_low_score():
    # Only alice sends — low balance and participation
    comms = _comms([("alice", "bob")] * 20)
    score, detail = engagement_score(comms, MEMBERS)
    assert score < 50.0
    assert detail["participation_rate"] == pytest.approx(0.25)


def test_engagement_empty():
    score, _ = engagement_score([], MEMBERS)
    assert score == 0.0


def test_engagement_bounded():
    comms = _comms([(a, b) for a in MEMBERS for b in MEMBERS if a != b] * 5)
    score, _ = engagement_score(comms, MEMBERS)
    assert 0.0 <= score <= 100.0


# ── exploration_score ────────────────────────────────────────────────────────


def test_exploration_no_cross_team():
    comms = _comms([("alice", "bob")] * 10, cross=False)
    score, detail = exploration_score(comms, MEMBERS)
    assert score == 0.0
    assert detail["cross_team_count"] == 0


def test_exploration_all_cross_team():
    comms = _comms([("alice", "bob")] * 10, cross=True)
    score, detail = exploration_score(comms, MEMBERS)
    assert score > 0.0
    assert detail["exploration_ratio"] == 1.0


def test_exploration_bounded():
    comms = _comms([("alice", "bob")] * 1000, cross=True)
    score, _ = exploration_score(comms, MEMBERS)
    assert 0.0 <= score <= 100.0


# ── overall_score ─────────────────────────────────────────────────────────────


def test_overall_weighted_average():
    result = overall_score(100.0, 100.0, 100.0)
    assert result == 100.0


def test_overall_default_weights_sum_to_one():
    # Weights (0.35, 0.40, 0.25) should sum to 1.0
    assert pytest.approx(0.35 + 0.40 + 0.25) == 1.0


def test_overall_custom_weights():
    score = overall_score(50.0, 80.0, 0.0, weights=(0.0, 1.0, 0.0))
    assert score == 80.0


# ── score_team ────────────────────────────────────────────────────────────────


def test_score_team_returns_result():
    comms = _comms([("alice", "bob"), ("bob", "carol"), ("carol", "alice")], cross=True)
    result = score_team(comms, MEMBERS, days=7)
    assert isinstance(result, ThreeEsResult)
    assert 0.0 <= result.energy <= 100.0
    assert 0.0 <= result.engagement <= 100.0
    assert 0.0 <= result.exploration <= 100.0
    assert 0.0 <= result.overall <= 100.0


def test_score_team_detail_keys():
    comms = _comms([("alice", "bob")])
    result = score_team(comms, MEMBERS)
    assert set(result.detail.keys()) == {"energy", "engagement", "exploration"}


def test_score_team_empty_comms():
    result = score_team([], MEMBERS)
    assert result.energy == 0.0
    assert result.engagement == 0.0
    assert result.exploration == 0.0
    assert result.overall == 0.0
