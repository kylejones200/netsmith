"""
API layer: Public surface.

Keep it small. Keep it consistent. Make it hard to misuse.
"""

from ..engine.contracts import UNREACHABLE
from .compute import communities, degree, pagerank
from .graph import Graph, GraphView
from .load import load_edges
from .metrics import betweenness, clustering, components
from .paths import reachability, shortest_paths, shortest_paths_multi
from .validate import validate_edges

__all__ = [
    "load_edges",
    "Graph",
    "GraphView",
    "degree",
    "pagerank",
    "communities",
    "betweenness",
    "clustering",
    "components",
    "UNREACHABLE",
    "shortest_paths",
    "shortest_paths_multi",
    "reachability",
    "validate_edges",
]
