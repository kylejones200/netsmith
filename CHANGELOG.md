# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note on history.** NetSmith was rearchitected out of the `ts2net` codebase on
> 2026-01-12, dropping the time-series conversion features. Versions 0.4.0–0.6.0
> that previously appeared in this file belonged to `ts2net` and are documented in
> that project, not here. NetSmith's own history starts at 0.1.1.

## [Unreleased]

## [0.3.0] - 2026-08-06

### Fixed
- `pip install netsmith` advertised a `netsmith` command that could not run:
  `[project.scripts]` declared the entry point but click was never a
  dependency. There is now a `cli` extra (click, pandas, pyarrow), and the
  command explains itself instead of raising `ModuleNotFoundError` when it is
  missing.

### Removed
- NetworkX is no longer needed to compute anything. It backed k-core, label
  propagation, the configuration and Erdos-Renyi null models, and a redundant
  `backend="networkx"` path on `louvain_hooks` / `modularity`; all four are now
  Rust kernels with pure-Python fallbacks, and the redundant backend is gone.
  `Graph.as_networkx()` stays as interop and raises a clear `ImportError` when
  NetworkX is absent — `tests/test_no_networkx_dependency.py` runs the public
  surface with `import networkx` forced to fail. It remains a dev dependency,
  because an independent reference implementation to test against is worth
  keeping.

### Added
- k-core decomposition, label propagation, and the configuration and
  Erdos-Renyi null models as Rust kernels (`kcore::core_numbers`,
  `community::label_propagation`, `nulls::configuration_model`,
  `nulls::erdos_renyi`), each reachable through the engine dispatch with a
  Python fallback. `k_core` now returns core numbers for every node without
  needing a `k` argument; the k-core is `core_numbers >= k`.
- Degree-preserving null models in Rust. `nulls::degree_preserving_rewire` and
  `degree_preserving_rewire_samples` randomize a graph by double edge swap,
  keeping every node's degree while destroying the wiring, with the samples
  generated in parallel from per-sample seeds derived up front (so results do
  not depend on thread scheduling). `null_models(method="degree_preserving")`
  uses it when the extension is built and needs no NetworkX at all; the other
  methods are unchanged. Measured: 5 samples of a 5,000-node / 25,000-edge
  graph in 0.109s against NetworkX's 9.377s (86x), and 50 samples in 0.978s.
  Rewiring reports the swaps it actually achieved rather than quietly returning
  an under-randomized graph.
- Multi-source shortest paths. `shortest_paths_from_source` rebuilt the
  adjacency list on every call, and that setup dominates when querying many
  sources over one fixed graph. `shortest_paths_from_adjacency` takes a
  prebuilt adjacency, and `shortest_paths_from_sources` builds it once and
  sweeps the sources in parallel. Reachable from Python as
  `netsmith.api.shortest_paths_multi`, `compute_shortest_paths_multi`, and
  `netsmith_rs.shortest_paths_multi_rust`, with a pure-Python fallback.
  Measured over 200 sources (`cargo run --release --example bfs_reuse`):
  10k nodes / 100k edges, 0.314s per-source vs 0.029s reusing the adjacency vs
  0.014s multi-source (11x, 22x); 100k nodes / 1M edges, 4.257s vs 1.243s vs
  0.354s (3x, 12x). `mean_shortest_path` now runs its sweeps in parallel too.
  Originally proposed in PR #1.
- Betweenness centrality — a Rust `centrality` module in `netsmith-core`
  implementing Brandes' algorithm, with the per-source shortest-path sweeps run
  in parallel across cores via rayon. Handles unweighted (BFS) and weighted
  (Dijkstra) graphs, directed and undirected, normalized or raw, and matches
  `networkx.betweenness_centrality` to 1e-9. `centrality(graph,
  method="betweenness")` previously raised `NotImplementedError`; it now also
  reaches `netsmith.api.betweenness` and `compute_betweenness` in the engine,
  with a pure-Python Brandes fallback when the extension is missing.
  Measured on this machine: 0.34s for 5,000 nodes / 25,000 edges where NetworkX
  takes 29.7s (87x), and 110x on the weighted equivalent. 10,000 nodes in 1.4s.
- Rust `community` module in `netsmith-core`: a pure-Rust Louvain implementation
  (local moving + graph aggregation) plus a `modularity` scorer. Both support
  edge weights and the resolution parameter `gamma`, and are exposed to Python
  as `netsmith_rs.louvain_rust` / `netsmith_rs.modularity_rust`.
- `louvain_python` and `modularity_python` in `netsmith.engine.python`: a
  pure-Python Louvain mirroring the Rust kernel, used when the extension is
  unavailable.
- `backend` argument on `netsmith.core.community.louvain_hooks` and
  `modularity` — `"auto"` (Rust when built, else Python), `"rust"`, `"python"`,
  or `"networkx"` for the previous NetworkX-backed behaviour.
- `click`, `pandas` and `pyarrow` in the dev dependencies: the CLI needs click to
  import at all and writes its Parquet output through pandas, so without them
  `tests/unit/test_cli.py` could not even be collected.

### Changed
- The Rust workspace is one crate. `netsmith-core` and `netsmith-py` were split
  across `rust/crates/`, which bought nothing — no third crate consumed the core,
  and the split forced the `[profile.release]` warning about a non-root package.
  Now `rust/` is the crate, `rust/src/python.rs` holds the PyO3 bindings, and
  pyo3 sits behind an optional `python` feature so `cargo test` links without
  libpython. Maturin builds it with `--features python`; anything that referred
  to `rust/crates/netsmith-py/Cargo.toml` now points at `rust/Cargo.toml`.
- `panic = "abort"` is gone from the release profile. It turned any Rust panic
  into a killed interpreter with no traceback; panics now surface as Python
  exceptions.
- flake8 configuration is consolidated in `setup.cfg` at 100 characters, matching
  `[tool.black] line-length`. It disagreed at 88 while both workflows passed
  `--max-line-length=100` on the command line; the workflows now take their
  options from the file. The dead `[tool:pytest]` section is gone — `pytest.ini`
  has always won, and pytest said so on every run.
- `louvain_hooks` and `modularity` now default to the built-in kernels instead of
  NetworkX, so community detection no longer requires NetworkX. Pass
  `backend="networkx"` for the old path. Partitions may differ from NetworkX's
  (Louvain is a stochastic greedy heuristic); modularity values are computed
  from the same definition and match NetworkX to floating-point tolerance.

### Fixed — silent failures
These all returned a plausible number instead of reporting a problem.

- The configuration model silently returned graphs whose degrees fell short of
  the sequence asked for, because simplifying the stub pairing drops self-loops
  and repeats. The count of discarded pairings is now reported alongside the
  samples.

- The two backends disagreed on what "unreachable" means in a distance array:
  the Rust kernel returned `usize::MAX` as uint64, the Python one `int64` max,
  so `distances == UNREACHABLE` matched on one backend and missed on the other.
  Both now return int64 carrying the exported `netsmith.api.UNREACHABLE`. The
  existing tests had been written as `dist > 1000 or dist == int64 max`, which
  accommodated the bug rather than catching it.

- Kernels no longer skip edges that name a node outside the graph. Every Rust
  kernel returns a typed `GraphError` naming the offending edge, and the Python
  kernels raise the same message. Previously `degree`, `clustering`,
  `components`, `paths`, `louvain`, `modularity` and `betweenness` all dropped
  such edges and answered a question about a different graph.
- `strength_sequence` padded a short weight array with 1.0 per missing entry.
  A weights array that does not cover every edge is now an error.
- `backend="rust"` no longer falls back to Python. Asking for a backend and
  silently getting another misreports what ran; the missing kernel is now a
  `BackendError`. `backend="auto"` still chooses freely, which is its job.
- `shortest_paths(weight=...)` accepted the argument and returned hop counts.
  It now raises `NotImplementedError` rather than passing off hops as distances.
- `Graph.edges_coo()` caught malformed edges, warned, and returned empty arrays,
  turning every downstream metric into a confident wrong answer. It raises.
- `null_models(method="degree_preserving")` swallowed the exception from
  `double_edge_swap` and returned the observed graph as its own null model — a
  significance test against itself. Both it and the configuration model now
  report failure instead of returning fewer or fake samples.
- Rust panics on bad input (weight-length asserts) are typed errors instead.

### Fixed
- `pagerank` returned scores that did not sum to 1. It followed only `u -> v`
  even on undirected graphs and discarded the rank of dangling nodes, so a
  6-node undirected graph could total 0.38 instead of 1. It now walks
  undirected edges in both directions, redistributes dangling mass, honours
  edge weights, warns instead of silently returning an unconverged vector, and
  matches `networkx.pagerank` to 1e-8 on directed, undirected and weighted
  graphs.
- `triangles_per_node` (Rust) reported twice the real count: each triangle was
  counted once per incident edge. A single triangle now gives every member 1.
- Clustering coefficients counted self-loops as neighbours, inflating degree and
  inventing triangles. Self-loops are now ignored, matching NetworkX.
- `clustering` disagreed between backends on directed input — the Python
  backend followed only `u -> v` while the Rust kernel symmetrized. Both now
  treat the graph as undirected, which is documented on the function.
- `average_clustering` (Rust) averaged only over nodes with at least two
  neighbours, so it did not equal the mean of `local_clustering`. It now
  averages over all nodes, as NetworkX does.
- CI installed a hand-maintained package list that omitted the CLI's
  dependencies. The workflows now install `.[dev,networkx,scipy]`, so the
  declared dependencies are the single source of truth.
- The `netsmith_rs` stub referenced a name that a star import does not bind. It
  worked only because importing a submodule happens to set it on the package, and
  the guard would have raised `NameError`, not the `ImportError` it caught.
- `load_edges` could not read a CSV or Parquet file on an install without
  polars: the pandas fallback referenced an unbound `pl`, so every path raised
  `UnboundLocalError`. Reading now works with polars or pandas, and raises a
  clear `ImportError` naming both if neither is installed.
- `netsmith compute-communities` called `load_edges` without column names, so it
  failed on every file input with "u_col and v_col must be specified". It now
  takes `--u-col` / `--v-col` / `--w-col` like the other commands.
- `netsmith.engine.python.communities_python` returned an all-zeros placeholder,
  so `compute_communities` / `netsmith.api.communities` silently reported "every
  node in one community" whenever the Rust extension was missing. It now runs
  Louvain, and an unsupported `method` raises `ValueError` instead of returning
  a fake partition.

## [0.2.2] - 2026-06-16

### Changed
- Version bump to publish the 0.2.1 packaging fix to PyPI.
- Applied `black` formatting across 5 files.

## [0.2.1] - 2026-06-16

### Fixed
- Wheels now contain the `netsmith` Python package. `python-source = "src"` alone
  was not enough for Maturin to pick up the package tree; added
  `python-packages = ["netsmith"]` under `[tool.maturin]`. Before this fix
  `import netsmith.ona` failed on a PyPI install even though the module was present
  in the repository.

## [0.2.0] - 2026-04-21

### Added
- `netsmith.ona` — Organizational Network Analysis:
  - `ona/three_es.py`: Energy / Engagement / Exploration scoring in pure NumPy, with
    no ORM or I/O coupling. Provides the `Communication` and `ThreeEsResult`
    dataclasses, `score_team()`, and `gini_coefficient()`. Implements the weights
    from Cross, Borgatti & Parker (2002).
  - `ona/silo.py`: union-find silo detection per topic cluster. `detect_silos()`
    returns `SiloResult` records sorted by severity, flagging clusters where two or
    more disconnected actor groups discuss the same topic (Burt 2004).
  - 30 unit tests.
- `src/netsmith_rs/__init__.py` stub and `python-source = "src"` so Maturin includes
  the Python source tree.
- `pythonpath = src` in `pytest.ini` for src-layout compatibility with the Maturin build.

### Fixed
- Python source packaging: wheels previously shipped only the compiled Rust
  extension, silently omitting everything under `src/netsmith/`.
- `engagement_score()` now returns `two_way_rate` on its empty-input early return, so
  the result shape no longer depends on whether any communications were supplied.

## [0.1.1] - 2026-01-12

### Added
- Initial NetSmith release: a four-layer network analysis library with Rust
  acceleration, rearchitected from `ts2net` with the time-series dependencies removed.
  - **Core** (`src/netsmith/core/`) — pure math, no I/O, no global state:
    `graph.py`, `metrics.py`, `paths.py`, `community.py`, `nulls.py`, `stats.py`.
  - **Engine** (`src/netsmith/engine/`) — `python/` reference implementations,
    `rust/` accelerated kernels, `dispatch.py` for backend selection, and
    `contracts.py` defining the canonical `EdgeList` / `GraphData` representations.
  - **API** (`src/netsmith/api/`) — stable public surface: `load.py`, `graph.py`,
    `compute.py`, `validate.py`.
  - **Apps** (`src/netsmith/apps/`) — `cli/`, `reports/`, `datasets/`.
- Rust workspace: `netsmith-core` (degree, metrics, paths) with `netsmith-py` PyO3
  bindings exposed to Python as `netsmith_rs`.
- `netsmith` CLI entry point.
- Example notebooks: quick start, temporal CNN, unified graphs.

### Fixed
- `reachability()` now handles unreachable nodes correctly.
- Removed Core → Engine imports that violated the layer boundary.

[Unreleased]: https://github.com/kylejones200/netsmith/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/kylejones200/netsmith/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/kylejones200/netsmith/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/kylejones200/netsmith/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/kylejones200/netsmith/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/kylejones200/netsmith/releases/tag/v0.1.1
