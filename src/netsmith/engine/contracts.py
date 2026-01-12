"""
Data contracts: Shared input schemas and output dtypes.

Canonical internal edge representation:
- u: int array length m
- v: int array length m
- w: float array length m (optional)
- directed: bool
- n_nodes: int (optional but preferred)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class EdgeList:
    """Canonical edge list representation."""

    u: NDArray[np.int64]  # Source nodes
    v: NDArray[np.int64]  # Destination nodes
    w: Optional[NDArray[np.float64]] = None  # Edge weights
    directed: bool = False
    n_nodes: Optional[int] = None

    def __post_init__(self):
        """Validate edge list."""
        if len(self.u) != len(self.v):
            raise ValueError("u and v must have same length")
        if self.w is not None and len(self.w) != len(self.u):
            raise ValueError("w must have same length as u and v")
        if self.n_nodes is None:
            self.n_nodes = int(max(np.max(self.u), np.max(self.v)) + 1)


@dataclass
class GraphData:
    """Graph data container."""

    edges: EdgeList
    node_attrs: Optional[dict] = None
    edge_attrs: Optional[dict] = None
