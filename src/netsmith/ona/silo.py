"""
Silo detection via union-find over actor pairs per topic cluster.

A silo exists when two or more components of ≥ min_size actors share a topic
cluster without any cross-component edges — they're talking about the same
things but not with each other.

Reference: Burt (2004) — structural holes and good ideas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

__all__ = ["SiloResult", "detect_silos"]


@dataclass(frozen=True)
class SiloResult:
    """One detected silo across two or more actor groups."""

    cluster_id: str | int
    cluster_name: str
    components: list[frozenset[str]]  # each component is a set of actor names
    component_count: int
    total_actors: int
    severity: str  # 'low' | 'medium' | 'high'


# ── Union-Find ────────────────────────────────────────────────────────────────


class _UF:
    def __init__(self, nodes: Iterable[str]) -> None:
        self._parent = {n: n for n in nodes}
        self._rank: dict[str, int] = {n: 0 for n in self._parent}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def components(self) -> list[frozenset[str]]:
        groups: dict[str, list[str]] = {}
        for node in self._parent:
            root = self.find(node)
            groups.setdefault(root, []).append(node)
        return [frozenset(v) for v in groups.values()]


# ── Public API ────────────────────────────────────────────────────────────────


def detect_silos(
    edges: Iterable[tuple[str, str]],
    actor_clusters: Mapping[str, Iterable[str | int]],
    cluster_names: Mapping[str | int, str] | None = None,
    *,
    min_component_size: int = 2,
    min_components: int = 2,
) -> list[SiloResult]:
    """
    Detect communication silos per topic cluster.

    Args:
        edges: Pairs of actor names that communicated, e.g. [("Alice", "Bob"), ...]
        actor_clusters: Mapping of actor_name → list of cluster IDs they appear in
        cluster_names: Optional mapping of cluster_id → human-readable name
        min_component_size: Ignore components smaller than this (noise filter)
        min_components: Only flag when this many disconnected components share a cluster

    Returns:
        List of SiloResult, one per siloed cluster, sorted by severity desc.
    """
    cluster_names = cluster_names or {}

    # Build per-cluster actor sets
    cluster_actors: dict[str | int, set[str]] = {}
    for actor, clusters in actor_clusters.items():
        for cid in clusters:
            cluster_actors.setdefault(cid, set()).add(actor)

    # Build adjacency for union-find (undirected)
    adj: dict[str, set[str]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    results: list[SiloResult] = []

    for cid, actors in cluster_actors.items():
        if len(actors) < min_component_size * min_components:
            continue

        uf = _UF(actors)
        for actor in actors:
            for neighbor in adj.get(actor, set()):
                if neighbor in actors:
                    uf.union(actor, neighbor)

        comps = [c for c in uf.components() if len(c) >= min_component_size]
        if len(comps) < min_components:
            continue

        total = sum(len(c) for c in comps)
        severity = (
            "high"
            if len(comps) >= 3 or total >= 10
            else "medium" if len(comps) == 2 and total >= 4 else "low"
        )

        results.append(
            SiloResult(
                cluster_id=cid,
                cluster_name=cluster_names.get(cid, str(cid)),
                components=comps,
                component_count=len(comps),
                total_actors=total,
                severity=severity,
            )
        )

    return sorted(results, key=lambda r: ("low", "medium", "high").index(r.severity), reverse=True)
