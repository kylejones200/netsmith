//! k-core decomposition.

use ndarray::Array1;

use crate::{build_adjacency_list, GraphError};

/// Compute the core number of every node.
///
/// A node's core number is the largest `k` for which it survives in the
/// k-core: the maximal subgraph where every node has at least `k` neighbours
/// inside it. Computed by repeatedly peeling the lowest-degree node, which
/// gives every core number in one pass.
///
/// Direction is ignored — the k-core is defined on the undirected graph.
///
/// Returns [`GraphError::NodeOutOfRange`] if any edge names a node that does
/// not exist.
pub fn core_numbers(n: usize, edges: &[(usize, usize)]) -> Result<Array1<usize>, GraphError> {
    let adjacency = build_adjacency_list(n, edges, true)?;

    // Self-loops do not add a neighbour, so they cannot hold a node in a core.
    let mut degree: Vec<usize> = adjacency
        .iter()
        .enumerate()
        .map(|(u, neighbours)| neighbours.iter().filter(|&&v| v != u).count())
        .collect();

    // Bucket queue: nodes ordered by current degree, peeled lowest first.
    let max_degree = degree.iter().copied().max().unwrap_or(0);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); max_degree + 1];
    degree
        .iter()
        .enumerate()
        .for_each(|(u, &d)| buckets[d].push(u));

    let mut core = vec![0usize; n];
    let mut peeled = vec![false; n];
    let mut peeled_count = 0usize;
    let mut level = 0usize;

    // A node is pushed into a new bucket each time its degree drops, so older
    // entries linger. They are discarded on sight rather than compacted, and
    // the loop counts peels instead of iterations so stale entries cannot end
    // it early — leaving nodes unpeeled and reported as core 0.
    while peeled_count < n {
        while level <= max_degree {
            while let Some(&candidate) = buckets[level].last() {
                if peeled[candidate] || degree[candidate] != level {
                    buckets[level].pop();
                } else {
                    break;
                }
            }
            if buckets[level].is_empty() {
                level += 1;
            } else {
                break;
            }
        }
        if level > max_degree {
            break;
        }

        let u = buckets[level].pop().expect("bucket checked non-empty");
        peeled[u] = true;
        peeled_count += 1;
        core[u] = level;

        for &v in &adjacency[u] {
            // Only neighbours above the current level can drop; that guard is
            // what keeps the peel level non-decreasing.
            if !peeled[v] && v != u && degree[v] > level {
                degree[v] -= 1;
                buckets[degree[v]].push(v);
            }
        }
    }

    Ok(Array1::from_vec(core))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_triangle_is_a_2_core() {
        let core = core_numbers(3, &[(0, 1), (1, 2), (0, 2)]).unwrap();
        assert_eq!(core.to_vec(), vec![2, 2, 2]);
    }

    #[test]
    fn a_pendant_node_sits_in_the_1_core() {
        // Triangle 0-1-2 plus a tail 2-3.
        let core = core_numbers(4, &[(0, 1), (1, 2), (0, 2), (2, 3)]).unwrap();
        assert_eq!(core.to_vec(), vec![2, 2, 2, 1]);
    }

    #[test]
    fn a_path_is_a_1_core() {
        let core = core_numbers(4, &[(0, 1), (1, 2), (2, 3)]).unwrap();
        assert_eq!(core.to_vec(), vec![1, 1, 1, 1]);
    }

    #[test]
    fn an_isolated_node_has_core_zero() {
        let core = core_numbers(3, &[(0, 1)]).unwrap();
        assert_eq!(core.to_vec(), vec![1, 1, 0]);
    }

    #[test]
    fn a_complete_graph_is_an_n_minus_1_core() {
        let edges: Vec<_> = (0..6)
            .flat_map(|i| ((i + 1)..6).map(move |j| (i, j)))
            .collect();
        assert_eq!(core_numbers(6, &edges).unwrap().to_vec(), vec![5; 6]);
    }

    #[test]
    fn two_cliques_joined_by_a_bridge_keep_their_own_cores() {
        // Two 4-cliques joined by a single edge: the bridge does not lift
        // anyone above the 3-core of their own clique.
        let mut edges: Vec<(usize, usize)> = (0..4)
            .flat_map(|i| ((i + 1)..4).map(move |j| (i, j)))
            .collect();
        edges.extend((4..8).flat_map(|i| ((i + 1)..8).map(move |j| (i, j))));
        edges.push((0, 4));

        assert_eq!(core_numbers(8, &edges).unwrap().to_vec(), vec![3; 8]);
    }

    #[test]
    fn self_loops_do_not_hold_a_node_in_a_core() {
        let plain = core_numbers(3, &[(0, 1)]).unwrap();
        let looped = core_numbers(3, &[(0, 1), (2, 2)]).unwrap();
        assert_eq!(plain.to_vec(), looped.to_vec());
    }

    #[test]
    fn out_of_range_edges_are_rejected() {
        assert!(core_numbers(2, &[(0, 5)]).is_err());
    }

    #[test]
    fn an_empty_graph_has_no_cores() {
        assert_eq!(core_numbers(0, &[]).unwrap().len(), 0);
        assert_eq!(core_numbers(3, &[]).unwrap().to_vec(), vec![0, 0, 0]);
    }

    /// Naive O(n^2) peel: repeatedly remove a lowest-degree node.
    ///
    /// Obviously correct and far too slow, which is exactly what a reference
    /// implementation should be.
    fn core_numbers_naive(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
        let mut neighbours: Vec<std::collections::HashSet<usize>> =
            vec![std::collections::HashSet::new(); n];
        for &(u, v) in edges {
            if u != v {
                neighbours[u].insert(v);
                neighbours[v].insert(u);
            }
        }

        let mut alive: Vec<bool> = vec![true; n];
        let mut core = vec![0usize; n];
        let mut level = 0usize;

        for _ in 0..n {
            let next = (0..n)
                .filter(|&u| alive[u])
                .min_by_key(|&u| neighbours[u].iter().filter(|&&v| alive[v]).count());
            let Some(u) = next else { break };

            let degree = neighbours[u].iter().filter(|&&v| alive[v]).count();
            level = level.max(degree);
            core[u] = level;
            alive[u] = false;
        }
        core
    }

    /// A cheap deterministic PRNG so the property test needs no dependency.
    fn pseudo_random_graph(seed: u64, n: usize, density: u64) -> Vec<(usize, usize)> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        (0..n)
            .flat_map(|u| ((u + 1)..n).map(move |v| (u, v)))
            .filter(|_| next() % 100 < density)
            .collect()
    }

    #[test]
    fn bucket_peel_matches_a_naive_peel_on_random_graphs() {
        // The bucket queue re-pushes a node on every degree drop, so stale
        // entries pile up. An earlier version counted iterations rather than
        // peels and ran out before finishing, silently reporting core 0 for
        // whole swathes of a graph. Regular test graphs did not catch it.
        for seed in 0..40u64 {
            let n = 10 + (seed as usize % 40);
            let density = 5 + (seed % 40);
            let edges = pseudo_random_graph(seed, n, density);

            assert_eq!(
                core_numbers(n, &edges).unwrap().to_vec(),
                core_numbers_naive(n, &edges),
                "seed {seed}, n {n}, density {density}"
            );
        }
    }
}
