"""
Three E's scoring: Energy, Engagement, Exploration.

Pure functions — no ORM, no I/O. Pass in simple dataclasses or dicts.

Reference: Cross, Borgatti & Parker (2002); Burt (2004).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "Communication",
    "ThreeEsResult",
    "energy_score",
    "engagement_score",
    "exploration_score",
    "overall_score",
    "score_team",
    "gini_coefficient",
]


@dataclass(frozen=True)
class Communication:
    """Single communication event between two actors."""

    sender_id: str | int
    receiver_id: str | int | None = None
    duration_minutes: float = 0.0
    comm_type: str = "email"  # email | meeting | face-to-face | chat
    is_cross_team: bool = False


@dataclass
class ThreeEsResult:
    """Full Three E's result for one team / time window."""

    energy: float
    engagement: float
    exploration: float
    overall: float
    detail: dict = field(default_factory=dict)


# ── Internals ─────────────────────────────────────────────────────────────────


def gini_coefficient(values: Sequence[float]) -> float:
    """Gini coefficient of a distribution. 0 = perfect equality, 1 = total inequality."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cumsum = np.cumsum(arr)
    return float((n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n)


# ── Public scoring functions ──────────────────────────────────────────────────


def energy_score(
    comms: Sequence[Communication],
    member_ids: Sequence[str | int],
    days: int = 30,
    *,
    freq_benchmark: float = 5.0,
    duration_benchmark: float = 120.0,
) -> tuple[float, dict]:
    """
    Energy = magnitude of communication (volume, duration, face-to-face quality).

    Benchmarks (Cross et al. 2002):
      - freq_benchmark: interactions per person per day considered "highly active"
      - duration_benchmark: minutes per person per day considered healthy

    Returns (score 0–100, detail dict).
    """
    n = len(member_ids)
    if n == 0 or not comms:
        # Same keys as the populated path: a caller reading detail["freq_normalized"]
        # should not hit a KeyError just because the team had a quiet month.
        return 0.0, {
            "total_comms": 0,
            "total_duration_min": 0.0,
            "face_to_face_ratio": 0.0,
            "freq_normalized": 0.0,
            "duration_normalized": 0.0,
        }

    durations = np.array([c.duration_minutes for c in comms], dtype=float)
    types = [c.comm_type for c in comms]

    total_comms = len(comms)
    total_duration = float(durations.sum())
    face_to_face = sum(1 for t in types if t == "face-to-face")

    days = max(days, 1)
    freq_norm = min((total_comms / (n * days)) / freq_benchmark, 1.0)
    dur_norm = min((total_duration / n / days) / duration_benchmark, 1.0)
    ftf_ratio = face_to_face / total_comms

    score = freq_norm * 20.0 + dur_norm * 30.0 + ftf_ratio * 50.0

    return round(float(np.clip(score, 0.0, 100.0)), 2), {
        "total_comms": total_comms,
        "total_duration_min": round(total_duration, 2),
        "face_to_face_ratio": round(ftf_ratio, 3),
        "freq_normalized": round(freq_norm, 3),
        "duration_normalized": round(dur_norm, 3),
    }


def engagement_score(
    comms: Sequence[Communication],
    member_ids: Sequence[str | int],
) -> tuple[float, dict]:
    """
    Engagement = balanced, two-way participation by all members.

    Uses Gini coefficient for balance (more robust than CV for skewed distributions).

    Returns (score 0–100, detail dict).
    """
    member_set = set(member_ids)
    n = len(member_set)
    if n == 0 or not comms:
        return 0.0, {
            "participation_rate": 0.0,
            "balance_score": 0.0,
            "gini": 0.0,
            "two_way_rate": 0.0,
        }

    sender_ids = [c.sender_id for c in comms]
    active = {s for s in sender_ids if s in member_set}
    participation = len(active) / n

    counts = {m: sum(1 for s in sender_ids if s == m) for m in member_set}
    gini = gini_coefficient(list(counts.values()))
    balance = 1.0 - gini  # lower inequality → higher balance

    # Two-way pairs: any pair (a, b) that communicated in both directions
    pair_counts: dict[tuple, int] = {}
    for c in comms:
        if c.receiver_id in member_set and c.sender_id in member_set:
            key = tuple(sorted([c.sender_id, c.receiver_id]))
            pair_counts[key] = pair_counts.get(key, 0) + 1
    two_way = sum(1 for v in pair_counts.values() if v >= 2)
    possible_pairs = n * (n - 1) / 2
    two_way_rate = two_way / possible_pairs if possible_pairs > 0 else 0.0

    score = participation * 40.0 + balance * 40.0 + two_way_rate * 20.0

    return round(float(np.clip(score, 0.0, 100.0)), 2), {
        "participation_rate": round(participation, 3),
        "balance_score": round(balance, 3),
        "gini": round(gini, 3),
        "two_way_rate": round(two_way_rate, 3),
    }


def exploration_score(
    comms: Sequence[Communication],
    member_ids: Sequence[str | int],
) -> tuple[float, dict]:
    """
    Exploration = cross-team communication breadth.

    Returns (score 0–100, detail dict).
    """
    member_set = set(member_ids)
    n = len(member_set)
    total = len(comms)
    if n == 0 or total == 0:
        return 0.0, {
            "cross_team_count": 0,
            "exploration_ratio": 0.0,
            "explorers": 0,
            "explorer_rate": 0.0,
        }

    cross = [c for c in comms if c.is_cross_team]
    cross_count = len(cross)
    ratio = cross_count / total
    explorers = {c.sender_id for c in cross if c.sender_id in member_set}
    explorer_rate = len(explorers) / n
    avg_cross = cross_count / n
    avg_norm = min(avg_cross / 5.0, 1.0)

    score = ratio * 40.0 + explorer_rate * 40.0 + avg_norm * 20.0

    return round(float(np.clip(score, 0.0, 100.0)), 2), {
        "cross_team_count": cross_count,
        "exploration_ratio": round(ratio, 3),
        "explorers": len(explorers),
        "explorer_rate": round(explorer_rate, 3),
    }


def overall_score(
    energy: float,
    engagement: float,
    exploration: float,
    *,
    weights: tuple[float, float, float] = (0.35, 0.40, 0.25),
) -> float:
    """Weighted overall score. Default weights per MIT research (Cross et al. 2002)."""
    we, wn, wx = weights
    return round(float(np.clip(energy * we + engagement * wn + exploration * wx, 0.0, 100.0)), 2)


def score_team(
    comms: Sequence[Communication],
    member_ids: Sequence[str | int],
    days: int = 30,
) -> ThreeEsResult:
    """Convenience wrapper: compute all three E's and overall in one call."""
    e_score, e_detail = energy_score(comms, member_ids, days)
    n_score, n_detail = engagement_score(comms, member_ids)
    x_score, x_detail = exploration_score(comms, member_ids)
    o_score = overall_score(e_score, n_score, x_score)

    return ThreeEsResult(
        energy=e_score,
        engagement=n_score,
        exploration=x_score,
        overall=o_score,
        detail={"energy": e_detail, "engagement": n_detail, "exploration": x_detail},
    )
