//! NetSmith Core - Pure network analysis algorithms
//!
//! This crate provides efficient implementations of network analysis algorithms
//! without any Python or time series dependencies.

#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]

use ndarray::{Array1, Array2};

pub mod degree;
pub mod metrics;
pub mod paths;

// Re-export for convenience
pub use degree::*;
pub use metrics::*;
pub use paths::*;

/// Build adjacency list from edge list
pub fn build_adjacency_list(n: usize, edges: &[(usize, usize)], undirected: bool) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::<usize>::new(); n];
    for &(u, v) in edges.iter() {
        if u >= n || v >= n {
            continue;
        }
        adj[u].push(v);
        if undirected {
            adj[v].push(u);
        }
    }
    for nbrs in adj.iter_mut() {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    adj
}

