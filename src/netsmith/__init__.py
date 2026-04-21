"""
NetSmith: Fast Network Analysis Library

A high-performance network analysis library with Rust acceleration.
"""

__version__ = "0.1.2"

from . import ona  # noqa: F401 — registers netsmith.ona

__all__ = ["ona"]
