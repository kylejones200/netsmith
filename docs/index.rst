netsmith documentation
======================

Fast network analysis library with Rust acceleration, focused on pure network analysis.

Overview
--------

netsmith provides high-performance network analysis algorithms with a clean 4-layer architecture:

- **Core Layer**: Pure math, no I/O, no global state
- **Engine Layer**: Performance and execution (Python and Rust backends)
- **API Layer**: Public surface, small and consistent
- **Apps Layer**: Opinionated use cases and CLI tools

Key Features
------------

- Fast implementations with Rust acceleration
- Dual backend system (Python reference + Rust accelerated)
- Clean 4-layer architecture
- Comprehensive graph metrics and algorithms
- Integration with NetworkX for advanced analysis
- Type-safe data contracts

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from netsmith.core import Graph
   from netsmith.core.metrics import degree

   # Create a simple graph
   edges = [(0, 1), (1, 2), (2, 0)]  # Triangle
   graph = Graph(edges=edges, n_nodes=3, directed=False)
   
   # Compute degree sequence
   degrees = degree(graph)
   print(f"Degrees: {degrees}")

Contents:

.. toctree::
   :maxdepth: 2
   :caption: Guide

   usage
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
