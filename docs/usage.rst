Usage Guide
===========

Install
-------

.. code-block:: bash

   pip install netsmith

Quick start
-----------

.. code-block:: python

   import numpy as np
   from netsmith.core import Graph
   from netsmith.core.metrics import degree, clustering

   # Create a simple graph
   edges = [(0, 1), (1, 2), (2, 0)]  # Triangle
   graph = Graph(edges=edges, n_nodes=3, directed=False)
   
   # Compute metrics
   degrees = degree(graph)
   clustering_coeffs = clustering(graph)
   
   print(f"Degrees: {degrees}")
   print(f"Clustering: {clustering_coeffs}")

CLI
---

.. code-block:: bash

   netsmith compute degree --input edges.csv --output degree.csv
   netsmith compute pagerank --input edges.csv --output pr.csv --alpha 0.85
   netsmith compute communities --input edges.csv --output communities.csv
