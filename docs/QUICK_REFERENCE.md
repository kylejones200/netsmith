# netsmith Quick Reference

## Installation

```bash
pip install netsmith                    # Basic
pip install netsmith[networkx]          # With NetworkX for advanced analysis
pip install netsmith[pandas]            # With pandas for data loading
pip install netsmith[polars]            # With polars for data loading
pip install netsmith[dev]               # Development dependencies
```

## Basic Usage

```python
from netsmith.core import Graph
from netsmith.core.metrics import degree, clustering, components
from netsmith.core.paths import shortest_paths
import numpy as np

# Create graph from edge list
edges = [(0, 1), (1, 2), (2, 0)]  # Triangle
graph = Graph(edges=edges, n_nodes=3, directed=False)

# Access results
graph.n_nodes              # Number of nodes
graph.n_edges              # Number of edges
graph.edges                # List of edges

# Compute metrics
degree(graph)              # Degree sequence
clustering(graph)          # Clustering coefficients
components(graph)          # Connected components
shortest_paths(graph, source=0)  # Shortest paths from source
```

## API Layer

### Loading Data

```python
from netsmith.api.load import load_edges

# Load from CSV
edges = load_edges("edges.csv", u_col="source", v_col="target")

# Load from pandas DataFrame
import pandas as pd
df = pd.read_csv("edges.csv")
edges = load_edges(df, u_col="u", v_col="v", w_col="weight")
```

### Computing Metrics

```python
from netsmith.api.compute import degree, pagerank, communities

# Compute degree
degrees = degree(edges, backend="auto")

# Compute PageRank
pr = pagerank(edges, alpha=0.85, backend="auto")

# Compute communities
communities = communities(edges, method="louvain", backend="auto")
```

## Core Layer

### Graph Class

```python
from netsmith.core import Graph

# Create undirected graph
graph = Graph(edges=[(0, 1), (1, 2)], n_nodes=3, directed=False)

# Create directed graph
graph = Graph(edges=[(0, 1), (1, 2)], n_nodes=3, directed=True)

# Create weighted graph
graph = Graph(edges=[(0, 1, 0.5), (1, 2, 1.5)], n_nodes=3, weighted=True)

# Convert to NetworkX
G = graph.as_networkx()
```

### Metrics

```python
from netsmith.core.metrics import degree, strength, clustering, components

# Degree sequence
degrees = degree(graph)

# Strength (weighted degree)
strengths = strength(graph)

# Clustering coefficients
clustering_coeffs = clustering(graph)

# Connected components
n_components, labels = components(graph, return_labels=True)
```

### Paths

```python
from netsmith.core.paths import shortest_paths, reachability

# Shortest paths from source
dist = shortest_paths(graph, source=0)

# Reachability
reachable = reachability(graph, source=0)
```

## Data Contracts

Canonical edge representation:

```python
from netsmith.engine.contracts import EdgeList

edges = EdgeList(
    u=np.array([0, 1, 2], dtype=np.int64),      # Source nodes
    v=np.array([1, 2, 0], dtype=np.int64),      # Destination nodes
    w=np.array([0.5, 1.5, 2.0], dtype=np.float64),  # Weights (optional)
    directed=False,
    n_nodes=3  # Optional, auto-calculated if not provided
)
```

## Backend Selection

netsmith supports multiple backends:

- `"auto"`: Automatically select best available (default)
- `"python"`: Use Python reference implementation
- `"rust"`: Use Rust-accelerated implementation (if available)

```python
from netsmith.api.compute import degree

# Auto-select backend
degrees = degree(edges, backend="auto")

# Force Python backend
degrees = degree(edges, backend="python")

# Force Rust backend (if available)
degrees = degree(edges, backend="rust")
```

## Performance Tips

1. **Use Rust backend**: Install with Rust support for 10-100x speedup
2. **Prefer EdgeList format**: Direct EdgeList is faster than loading from files
3. **Use appropriate graph representation**: Sparse for large graphs
4. **Batch operations**: Process multiple graphs in sequence for better caching

## Common Patterns

```python
# Load and analyze graph
from netsmith.api import load_edges, degree, clustering

edges = load_edges("edges.csv", u_col="source", v_col="target")
degrees = degree(edges)
clustering_coeffs = clustering(edges)

# Create Graph object for multiple operations
from netsmith.core import Graph

graph = Graph(edges=list(zip(edges.u, edges.v)), n_nodes=edges.n_nodes)
degrees = degree(graph)
clustering_coeffs = clustering(graph)
components_result = components(graph)
```
