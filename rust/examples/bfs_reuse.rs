//! What rebuilding the adjacency list per shortest-path query costs.
//!
//! Run with: `cargo run --release --example bfs_reuse`

use netsmith_rs::paths::{shortest_paths_from_source, shortest_paths_from_sources};
use std::time::Instant;

/// A kNN-shaped graph: every node linked to `degree` others.
fn knn_graph(n: usize, degree: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(n * degree);
    for i in 0..n {
        for d in 1..=degree {
            let j = (i + d * 7) % n;
            if i != j {
                edges.push((i, j));
            }
        }
    }
    edges
}

fn main() {
    const SOURCES: usize = 200;
    println!(
        "{:>9} {:>10} {:>16} {:>15} {:>10}",
        "nodes", "edges", "per-source loop", "multi-source", "speedup"
    );

    for n in [10_000usize, 100_000] {
        let edges = knn_graph(n, 10);
        let sources: Vec<usize> = (0..SOURCES).collect();

        let start = Instant::now();
        for &source in &sources {
            let _ = shortest_paths_from_source(n, &edges, source, false).unwrap();
        }
        let looped = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let _ = shortest_paths_from_sources(n, &edges, &sources, false).unwrap();
        let batched = start.elapsed().as_secs_f64();

        println!(
            "{n:>9} {:>10} {looped:>15.3}s {batched:>14.3}s {:>9.0}x",
            edges.len(),
            looped / batched
        );
    }
}
