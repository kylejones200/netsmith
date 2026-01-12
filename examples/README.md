# NetSmith Examples

This directory contains example scripts demonstrating NetSmith's network analysis capabilities.

## Quick Start

### `basic_network_analysis.ipynb` or `basic_network_analysis.py`
Basic introduction to NetSmith with edge lists. Shows:
- Loading graphs from edge lists
- Computing degree, clustering, and shortest paths
- Community detection
- Basic network metrics

**Run:**
```bash
# Jupyter notebook (recommended)
jupyter notebook examples/basic_network_analysis.ipynb

# Or Python script
python examples/basic_network_analysis.py
```

### `metrics_example.ipynb` or `metrics_example.py`
Comprehensive example of network metrics computation:
- Degree sequences and distributions
- Clustering coefficients
- Centrality measures
- K-core decomposition
- Connected components

**Run:**
```bash
jupyter notebook examples/metrics_example.ipynb
# Or: python examples/metrics_example.py
```

### `community_detection.ipynb` or `community_detection_example.py`
Examples of community detection algorithms:
- Louvain community detection
- Label propagation
- Modularity computation

**Run:**
```bash
jupyter notebook examples/community_detection.ipynb
# Or: python examples/community_detection_example.py
```

### `null_models.ipynb` or `null_models_example.py`
Examples of null models and statistical testing:
- Configuration model
- Erdos-Renyi graphs
- Degree-preserving randomization
- Permutation tests

**Run:**
```bash
jupyter notebook examples/null_models.ipynb
# Or: python examples/null_models_example.py
```

## Installation

```bash
pip install netsmith
```

For optional dependencies:
```bash
pip install netsmith[networkx]  # For NetworkX integration
pip install netsmith[pandas]    # For pandas integration
pip install netsmith[polars]    # For polars integration
```

## Notes

- All examples use edge lists as the primary data format
- Examples demonstrate working with graphs directly, not building them from time series
- NetSmith focuses on fast network analysis, not graph construction methods
