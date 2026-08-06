//! Shortest path algorithms.
//!
//! Distances are hop counts; see [`crate::centrality`] for the weighted
//! (Dijkstra) machinery.

use std::collections::VecDeque;

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::{build_adjacency_list, GraphError};

/// Marker for a node the search never reached.
pub const UNREACHABLE: usize = usize::MAX;

/// Breadth-first distances from `source` into `distances`.
///
/// `distances` must be `UNREACHABLE` everywhere on entry.
fn breadth_first(adjacency: &[Vec<usize>], source: usize, distances: &mut [usize]) {
    distances[source] = 0;
    let mut queue = VecDeque::from([source]);

    while let Some(u) = queue.pop_front() {
        let next = distances[u] + 1;
        // Iterated rather than chained: a filter/for_each pair would need to
        // borrow `distances` immutably and mutably at once, and collecting to
        // dodge that would allocate on every node of the search.
        for &v in &adjacency[u] {
            if distances[v] == UNREACHABLE {
                distances[v] = next;
                queue.push_back(v);
            }
        }
    }
}

/// Compute the mean shortest path length over all reachable pairs.
///
/// Unreachable pairs are left out of the average rather than counted as
/// infinite. A graph with no reachable pair at all returns `NaN` — there is no
/// mean to report, and reporting 0.0 would read as "everything is adjacent".
///
/// Returns [`GraphError::NodeOutOfRange`] if any edge names a node that does
/// not exist.
pub fn mean_shortest_path(n: usize, edges: &[(usize, usize)]) -> Result<f64, GraphError> {
    let adjacency = build_adjacency_list(n, edges, true)?;

    // One sweep per source, and the sweeps share nothing but the immutable
    // graph. Integer addition is associative, so the parallel reduction is
    // exactly reproducible.
    let (total, pairs) = (0..n)
        .into_par_iter()
        .map(|source| {
            let mut distances = vec![UNREACHABLE; n];
            breadth_first(&adjacency, source, &mut distances);

            // Each unordered pair is counted once, from its lower-numbered end.
            distances[source + 1..]
                .iter()
                .filter(|&&d| d != UNREACHABLE)
                .fold((0usize, 0usize), |(total, pairs), &d| {
                    (total + d, pairs + 1)
                })
        })
        .reduce(
            || (0usize, 0usize),
            |(total, pairs), (t, p)| (total + t, pairs + p),
        );

    Ok(if pairs > 0 {
        total as f64 / pairs as f64
    } else {
        f64::NAN
    })
}

/// Compute hop distances from `source` to every node.
///
/// Unreachable nodes are marked [`UNREACHABLE`].
///
/// Returns [`GraphError`] if any edge names a node that does not exist, or if
/// `source` is not a node of the graph.
pub fn shortest_paths_from_source(
    n: usize,
    edges: &[(usize, usize)],
    source: usize,
    directed: bool,
) -> Result<Array1<usize>, GraphError> {
    let adjacency = build_adjacency_list(n, edges, !directed)?;
    shortest_paths_from_adjacency(&adjacency, source)
}

/// Compute hop distances from `source` over an adjacency list you already have.
///
/// [`shortest_paths_from_source`] rebuilds the adjacency on every call, which
/// dominates the cost when querying many sources over one graph. Build it once
/// with [`build_adjacency_list`] and call this instead.
///
/// Returns [`GraphError::NodeOutOfRange`] if `source` is not a node of the
/// graph. "There is no such node" and "that node reaches nothing" are different
/// facts, and an all-unreachable row cannot tell you which one you have.
pub fn shortest_paths_from_adjacency(
    adjacency: &[Vec<usize>],
    source: usize,
) -> Result<Array1<usize>, GraphError> {
    let n = adjacency.len();
    if source >= n {
        return Err(GraphError::NodeOutOfRange {
            index: 0,
            u: source,
            v: source,
            n_nodes: n,
        });
    }

    let mut distances = vec![UNREACHABLE; n];
    breadth_first(adjacency, source, &mut distances);
    Ok(Array1::from_vec(distances))
}

/// Compute hop distances from each of several sources.
///
/// Builds the adjacency list once and sweeps the sources in parallel — they are
/// independent, sharing nothing but the immutable graph. Row `i` holds the
/// distances from `sources[i]`.
///
/// Returns [`GraphError::NodeOutOfRange`] if any edge, or any source, names a
/// node that does not exist.
pub fn shortest_paths_from_sources(
    n: usize,
    edges: &[(usize, usize)],
    sources: &[usize],
    directed: bool,
) -> Result<Array2<usize>, GraphError> {
    let adjacency = build_adjacency_list(n, edges, !directed)?;

    let rows = sources
        .par_iter()
        .map(|&source| shortest_paths_from_adjacency(&adjacency, source))
        .collect::<Result<Vec<_>, _>>()?;

    let mut distances = Array2::from_elem((sources.len(), n), UNREACHABLE);
    rows.into_iter()
        .enumerate()
        .for_each(|(i, row)| distances.row_mut(i).assign(&row));
    Ok(distances)
}

/// Label each node with its connected component.
///
/// Components are numbered from 0 in order of their lowest-numbered node.
/// Direction is ignored: these are weakly connected components.
///
/// Returns [`GraphError::NodeOutOfRange`] if any edge names a node that does
/// not exist.
pub fn connected_components(
    n: usize,
    edges: &[(usize, usize)],
) -> Result<(usize, Array1<usize>), GraphError> {
    let adjacency = build_adjacency_list(n, edges, true)?;

    let mut labels = vec![UNREACHABLE; n];
    let mut component_count = 0usize;

    // Same borrow story as the BFS above: seeding a component reads `labels`
    // and the search writes it, so this stays a plain loop.
    for start in 0..n {
        if labels[start] != UNREACHABLE {
            continue;
        }
        let component = component_count;
        component_count += 1;

        let mut queue = VecDeque::from([start]);
        labels[start] = component;
        while let Some(u) = queue.pop_front() {
            for &v in &adjacency[u] {
                if labels[v] == UNREACHABLE {
                    labels[v] = component;
                    queue.push_back(v);
                }
            }
        }
    }

    Ok((component_count, Array1::from_vec(labels)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn distances_along_a_path() {
        let distances = shortest_paths_from_source(4, &[(0, 1), (1, 2), (2, 3)], 0, false).unwrap();
        assert_eq!(distances.to_vec(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn unreached_nodes_are_marked_not_zeroed() {
        let distances = shortest_paths_from_source(3, &[(0, 1)], 0, false).unwrap();
        assert_eq!(distances[2], UNREACHABLE);
    }

    #[test]
    fn direction_is_respected() {
        let forward = shortest_paths_from_source(3, &[(0, 1), (1, 2)], 2, true).unwrap();
        assert_eq!(forward[0], UNREACHABLE);

        let undirected = shortest_paths_from_source(3, &[(0, 1), (1, 2)], 2, false).unwrap();
        assert_eq!(undirected[0], 2);
    }

    #[test]
    fn mean_shortest_path_averages_reachable_pairs() {
        // Path 0-1-2: distances 1, 1 and 2 over three pairs.
        assert_relative_eq!(
            mean_shortest_path(3, &[(0, 1), (1, 2)]).unwrap(),
            4.0 / 3.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn mean_shortest_path_skips_unreachable_pairs() {
        // Two separate edges: only the two within-component pairs count.
        assert_relative_eq!(
            mean_shortest_path(4, &[(0, 1), (2, 3)]).unwrap(),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn mean_shortest_path_of_an_edgeless_graph_is_nan() {
        assert!(mean_shortest_path(3, &[]).unwrap().is_nan());
    }

    #[test]
    fn components_are_numbered_from_zero() {
        let (count, labels) = connected_components(5, &[(0, 1), (3, 4)]).unwrap();
        assert_eq!(count, 3);
        assert_eq!(labels.to_vec(), vec![0, 0, 1, 2, 2]);
    }

    /// Two triangles with no edge between them.
    fn two_components() -> (usize, Vec<(usize, usize)>) {
        (6, vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    }

    #[test]
    fn adjacency_variant_matches_the_edge_list_variant() {
        let cases = [(6, vec![(0, 1), (1, 2), (2, 3), (3, 4)]), two_components()];
        for (n, edges) in cases {
            for directed in [false, true] {
                let adjacency = build_adjacency_list(n, &edges, !directed).unwrap();
                for source in 0..n {
                    assert_eq!(
                        shortest_paths_from_source(n, &edges, source, directed).unwrap(),
                        shortest_paths_from_adjacency(&adjacency, source).unwrap(),
                        "source {source}, directed {directed}"
                    );
                }
            }
        }
    }

    #[test]
    fn multi_source_rows_match_single_source_calls() {
        let (n, edges) = two_components();
        let sources = [0, 3, 5, 1];
        let many = shortest_paths_from_sources(n, &edges, &sources, false).unwrap();

        assert_eq!(many.shape(), &[sources.len(), n]);
        for (i, &source) in sources.iter().enumerate() {
            let one = shortest_paths_from_source(n, &edges, source, false).unwrap();
            assert_eq!(many.row(i).to_vec(), one.to_vec(), "source {source}");
        }
    }

    #[test]
    fn multi_source_respects_direction() {
        let edges = vec![(0, 1), (1, 2)];
        let forward = shortest_paths_from_sources(3, &edges, &[0, 2], true).unwrap();
        assert_eq!(forward[[0, 2]], 2);
        assert_eq!(forward[[1, 0]], UNREACHABLE, "directed edges are one-way");

        let undirected = shortest_paths_from_sources(3, &edges, &[2], false).unwrap();
        assert_eq!(undirected[[0, 0]], 2);
    }

    #[test]
    fn no_sources_yields_no_rows() {
        let (n, edges) = two_components();
        let empty = shortest_paths_from_sources(n, &edges, &[], false).unwrap();
        assert_eq!(empty.shape(), &[0, n]);
    }

    #[test]
    fn a_source_outside_the_graph_is_an_error_not_an_empty_row() {
        // An all-unreachable row cannot distinguish "node 9 reaches nothing"
        // from "there is no node 9".
        let (n, edges) = two_components();
        let adjacency = build_adjacency_list(n, &edges, true).unwrap();

        assert_eq!(
            shortest_paths_from_adjacency(&adjacency, n + 3).unwrap_err(),
            GraphError::NodeOutOfRange {
                index: 0,
                u: n + 3,
                v: n + 3,
                n_nodes: n,
            }
        );
        assert!(shortest_paths_from_sources(n, &edges, &[0, n + 3], false).is_err());
    }

    #[test]
    fn out_of_range_input_is_rejected() {
        assert!(mean_shortest_path(2, &[(0, 4)]).is_err());
        assert!(connected_components(2, &[(0, 4)]).is_err());
        assert!(shortest_paths_from_source(2, &[(0, 1)], 5, false).is_err());
    }
}
