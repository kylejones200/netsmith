"""
netsmith.ona — Organizational Network Analysis

Three E's scoring (Cross, Borgatti & Parker 2002; Burt 2004) and silo detection.

    from netsmith.ona import score_team, detect_silos, Communication
"""

from .silo import SiloResult, detect_silos
from .three_es import (
    Communication,
    ThreeEsResult,
    energy_score,
    engagement_score,
    exploration_score,
    gini_coefficient,
    overall_score,
    score_team,
)

__all__ = [
    "Communication",
    "ThreeEsResult",
    "energy_score",
    "engagement_score",
    "exploration_score",
    "exploration_score",
    "gini_coefficient",
    "overall_score",
    "score_team",
    "SiloResult",
    "detect_silos",
]
