//! What it costs to rebuild the adjacency list on every shortest-path query.
//!
//! Run with: `cargo run --release --example bfs_reuse`

use std::time::Instant;

use netsmith_rs::{
    build_adjacency_list, mean_shortest_path, shortest_paths_from_adjacency,
    shortest_paths_from_source, shortest_paths_from_sources,
};

/// A kNN-shaped graph: every node linked to `degree` others.
fn knn_graph(n: usize, degree: usize) -> Vec<(usize, usize)> {
    (0..n)
        .flat_map(|i| (1..=degree).map(move |d| (i, (i + d * 7) % n)))
        .filter(|&(i, j)| i != j)
        .collect()
}

fn seconds<T>(work: impl FnOnce() -> T) -> f64 {
    let start = Instant::now();
    let _ = work();
    start.elapsed().as_secs_f64()
}

fn main() {
    const SOURCES: usize = 200;

    println!(
        "{:>9} {:>10} {:>13} {:>17} {:>14}",
        "nodes", "edges", "per-source", "reused adjacency", "multi-source"
    );

    for n in [10_000usize, 100_000] {
        let edges = knn_graph(n, 10);
        let sources: Vec<usize> = (0..SOURCES).collect();

        let per_source = seconds(|| {
            sources
                .iter()
                .for_each(|&s| drop(shortest_paths_from_source(n, &edges, s, false)))
        });

        let reused = seconds(|| {
            let adjacency = build_adjacency_list(n, &edges, true).unwrap();
            sources
                .iter()
                .for_each(|&s| drop(shortest_paths_from_adjacency(&adjacency, s)))
        });

        let multi = seconds(|| shortest_paths_from_sources(n, &edges, &sources, false));

        println!(
            "{n:>9} {:>10} {per_source:>12.3}s {reused:>16.3}s {multi:>13.3}s   \
             ({:.0}x reused, {:.0}x multi)",
            edges.len(),
            per_source / reused,
            per_source / multi
        );
    }

    // mean_shortest_path sweeps every node, so it benefits from the same
    // parallel traversal.
    let n = 3_000;
    let edges = knn_graph(n, 10);
    println!(
        "\nmean_shortest_path over all {n} sources: {:.3}s",
        seconds(|| mean_shortest_path(n, &edges))
    );
}
