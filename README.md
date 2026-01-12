# NetSmith: Fast Network Analysis Library

NetSmith is a high-performance network analysis library with Rust acceleration, focused on pure network analysis without time series dependencies.

## Architecture

NetSmith follows a four-layer architecture:

### Layer 1: Core
Pure math, no I/O, no global state. Located in `src/netsmith/core/`:
- `graph.py`: Core graph types (`Graph`, `GraphView`)
- `metrics.py`: Degree, centrality, assortativity, clustering, k-core, components
- `paths.py`: Shortest paths, reachability, walk metrics
- `community.py`: Modularity, Louvain hooks, label propagation hooks
- `nulls.py`: Null models and permutation tests
- `stats.py`: Distributions, confidence intervals, bootstrap

### Layer 2: Engine
Performance and execution. Located in `src/netsmith/engine/`:
- `python/`: Reference Python implementations
- `rust/`: Rust-accelerated kernels (to be implemented)
- `dispatch.py`: Backend selection (auto, python, rust)
- `contracts.py`: Data contracts (EdgeList, GraphData)

### Layer 3: API
Public surface. Located in `src/netsmith/api/`:
- `load.py`: Load edges from pandas, polars, parquet, csv
- `graph.py`: Public Graph API
- `compute.py`: Stable compute functions (degree, pagerank, communities)
- `validate.py`: Input validation

### Layer 4: Apps
Opinionated use cases. Located in `src/netsmith/apps/`:
- `cli/`: Command-line interface
- `reports/`: HTML/markdown report generation
- `datasets/`: Sample graphs and download helpers

## Data Contracts

Canonical edge representation:
```python
EdgeList(
    u: NDArray[np.int64],      # Source nodes (length m)
    v: NDArray[np.int64],      # Destination nodes (length m)
    w: Optional[NDArray[np.float64]],  # Edge weights (optional)
    directed: bool,
    n_nodes: Optional[int]     # Preferred but inferred if not provided
)
```

## Usage Examples

### Basic Usage

```python
import netsmith
import numpy as np

# Create edge list
u = np.array([0, 1, 2], dtype=np.int64)
v = np.array([1, 2, 0], dtype=np.int64)
edges = netsmith.api.load.EdgeList(u=u, v=v, directed=False, n_nodes=3)

# Compute degree
degrees = netsmith.degree(edges, backend="auto")
print(degrees)  # [2, 2, 2]

# Compute PageRank
pr = netsmith.pagerank(edges, alpha=0.85, backend="auto")
print(pr)

# Compute communities
communities = netsmith.communities(edges, method="louvain", backend="auto")
print(communities)
```

### Loading from Files

```python
# Load from parquet
edges = netsmith.load_edges("edges.parquet", u_col="source", v_col="target")

# Load from CSV
edges = netsmith.load_edges("edges.csv", u_col="u", v_col="v", w_col="weight")
```

### CLI Usage

```bash
# Compute degree
netsmith compute degree --input edges.parquet --out degree.parquet

# Compute PageRank
netsmith compute pagerank --input edges.parquet --out pr.parquet --alpha 0.85

# Compute communities
netsmith compute communities --input edges.parquet --out communities.parquet
```

## Installation

**Minimal installation (numpy only - ~10MB):**
```bash
pip install netsmith
```

**With optional dependencies:**
```bash
pip install netsmith[scipy]      # For sparse matrices (adjacency_matrix format='sparse'/'coo')
pip install netsmith[networkx]   # For community detection, null models, k-core decomposition
pip install netsmith[pandas]     # For pandas data loading
pip install netsmith[polars]     # For polars data loading

# Or install all optional dependencies:
pip install netsmith[scipy,networkx,pandas,polars]

# Development
pip install netsmith[dev]
```

**Note:** Core functionality (degree, paths, components, clustering) works with just `numpy`. 
`scipy` is only needed for sparse matrix formats, and `networkx` is only needed for advanced 
community detection and null models.

## Rust Backend

The Rust backend will be automatically used if available. To build:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build and install
maturin develop --release
```

## Status

✅ **Completed:**
- Core layer structure (graph types, metrics, paths, community, nulls, stats)
- Engine layer structure (Python backend, dispatch system, contracts)
- API layer (load, graph, compute, validate)
- Apps layer (CLI skeleton)
- pyproject.toml configuration

🚧 **In Progress:**
- Rust backend implementation
- Full metric implementations
- Community detection algorithms
- Test suite

📋 **Planned:**
- Rust acceleration for Phase 1 kernels (degree, strength, components, BFS, k-core)
- Comprehensive test coverage
- Documentation
- Performance benchmarks

## Design Philosophy

NetSmith focuses on pure network analysis. For time series to network conversion, use downstream libraries that build on NetSmith.

## License

MIT

