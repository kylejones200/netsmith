"""
Python implementations of null model sampling.

Mirrors the Rust kernels in ``rust/src/nulls.rs``. Different RNGs, so a given
seed does not carry across backends — what matches is the defining property of
each model.
"""

import numpy as np
from numpy.typing import NDArray


def configuration_model_python(degrees, seed=None):
    """
    Sample a simple graph with the given degree sequence (Python backend).

    Lays out one stub per unit of degree, pairs them at random, and drops
    pairings that would self-loop or repeat an edge — so realized degrees can
    fall short of the requested ones.

    Returns
    -------
    edges : list of (int, int)
    discarded : int
        Pairings dropped while simplifying. Each costs two nodes one degree.

    Raises
    ------
    ValueError
        If the degree sum is odd, since no graph has an odd degree sum
    """
    degrees = np.asarray(degrees, dtype=np.int64)
    if degrees.size and degrees.min() < 0:
        raise ValueError("degrees must be non-negative")
    if int(degrees.sum()) % 2 != 0:
        raise ValueError("degrees must sum to an even number; no graph has an odd degree sum")

    stubs = np.repeat(np.arange(len(degrees), dtype=np.int64), degrees)
    np.random.default_rng(seed).shuffle(stubs)

    seen = set()
    edges = []
    discarded = 0
    for u, v in stubs.reshape(-1, 2):
        u, v = int(u), int(v)
        edge = (u, v) if u <= v else (v, u)
        if u == v or edge in seen:
            discarded += 1
            continue
        seen.add(edge)
        edges.append(edge)
    return edges, discarded


def degree_preserving_rewire_python(
    edges, n_samples: int, target_swaps: int, max_attempts: int, seed
):
    """
    Randomize by double edge swap, preserving every degree (Python backend).

    Takes edges (u,v) and (x,y) to (u,y) and (x,v), rejecting swaps that would
    self-loop or duplicate an existing edge. Mirrors the Rust kernel; different
    RNG, so a seed does not carry across backends.

    Returns
    -------
    samples : list of ndarray
        One rewired edge array per sample
    swaps : list of int
        Swaps each sample actually achieved. A shortfall means the graph is too
        constrained to randomize, which the caller must not ignore.
    attempts : list of int

    Raises
    ------
    ValueError
        If the input has a self-loop or a repeated edge — degree-preserving
        randomization samples simple graphs
    """
    u = np.asarray(edges.u, dtype=np.int64)
    v = np.asarray(edges.v, dtype=np.int64)
    canonical = [(int(min(a, b)), int(max(a, b))) for a, b in zip(u, v)]

    if any(a == b for a, b in canonical):
        raise ValueError(
            "edges must not contain self-loops; degree-preserving randomization "
            "samples simple graphs"
        )
    if len(set(canonical)) != len(canonical):
        raise ValueError(
            "edges must not repeat an edge; degree-preserving randomization "
            "samples simple graphs"
        )

    samples, swap_counts, attempt_counts = [], [], []
    m = len(canonical)

    for sample_index in range(n_samples):
        rng = np.random.default_rng(seed + sample_index)
        current = list(canonical)
        present = set(current)
        swaps = attempts = 0

        while m >= 2 and swaps < target_swaps and attempts < max_attempts:
            attempts += 1
            first, second = int(rng.integers(m)), int(rng.integers(m))
            if first == second:
                continue

            a, b = current[first]
            x, y = current[second]
            if rng.integers(2):
                b, y = y, b
            if a == y or x == b:
                continue

            new_first = (min(a, y), max(a, y))
            new_second = (min(x, b), max(x, b))
            if new_first in present or new_second in present:
                continue

            present.discard(current[first])
            present.discard(current[second])
            present.add(new_first)
            present.add(new_second)
            current[first] = new_first
            current[second] = new_second
            swaps += 1

        samples.append(np.asarray(current, dtype=np.int64).reshape(-1, 2))
        swap_counts.append(swaps)
        attempt_counts.append(attempts)

    return samples, swap_counts, attempt_counts


def erdos_renyi_python(n: int, m: int, seed=None) -> NDArray[np.int64]:
    """
    Sample a uniformly random simple graph with `n` nodes and `m` edges.

    Raises
    ------
    ValueError
        If `m` exceeds the number of distinct node pairs
    """
    capacity = n * (n - 1) // 2
    if m > capacity:
        raise ValueError(
            f"cannot place {m} edges among {n} nodes; there are only {capacity} distinct pairs"
        )
    if m == 0:
        return np.zeros((0, 2), dtype=np.int64)

    # Sample distinct pair indices, then decode each back into its (v, u).
    # Index k maps to the pair whose larger element u satisfies
    # u(u-1)/2 <= k < u(u+1)/2, with v = k - u(u-1)/2.
    chosen = np.random.default_rng(seed).choice(capacity, size=m, replace=False)
    u = ((1.0 + np.sqrt(1.0 + 8.0 * chosen)) / 2.0).astype(np.int64)
    # sqrt is inexact near the boundaries, so nudge u onto the right row.
    u = np.where(u * (u - 1) // 2 > chosen, u - 1, u)
    u = np.where((u + 1) * u // 2 <= chosen, u + 1, u)
    v = chosen - u * (u - 1) // 2
    return np.column_stack([v, u]).astype(np.int64)
