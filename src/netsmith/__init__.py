"""
NetSmith: Fast Network Analysis Library

A high-performance network analysis library with Rust acceleration.
"""

# Single-sourced from package metadata so it cannot drift from pyproject.toml again
# (it sat at 0.2.0 through the 0.2.1 and 0.2.2 releases). The fallback covers running
# from a source tree that was never installed.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("netsmith")
except (ImportError, PackageNotFoundError):  # pragma: no cover
    __version__ = "0.0.0+unknown"

from . import ona  # noqa: F401 — registers netsmith.ona

__all__ = ["ona"]
