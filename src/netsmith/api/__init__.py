"""
API layer: Public surface.

Keep it small. Keep it consistent. Make it hard to misuse.
"""

from .load import load_edges
from .graph import Graph, GraphView
from .compute import degree, pagerank, communities
from .validate import validate_edges

__all__ = [
    "load_edges",
    "Graph",
    "GraphView",
    "degree",
    "pagerank",
    "communities",
    "validate_edges",
]

