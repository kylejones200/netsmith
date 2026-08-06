//! Null models: degree-preserving randomization.
//!
//! Double edge swap takes two edges `(u, v)` and `(x, y)` and rewires them to
//! `(u, y)` and `(x, v)`. Every node keeps the same degree, so the result is a
//! graph drawn from the space of graphs with the observed degree sequence —
//! the null a significance test needs when it wants to ask "is this structure
//! more than my degree distribution would produce anyway?".
//!
//! A swap is rejected when it would create a self-loop or duplicate an edge
//! that already exists, since neither is in that space.

use std::collections::HashSet;

use rayon::prelude::*;

use crate::rng::SplitMix64;
use crate::{validate_edges, GraphError};

/// One randomized graph, with an honest account of how it was produced.
///
/// `swaps` is what the rewiring actually achieved, which can fall short of
/// what was asked: a graph can be too constrained to rewire — a triangle with
/// a pendant node has essentially no valid swap. Reporting the shortfall is
/// the caller's cue that the "null" may be too close to the observed graph to
/// test against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewireResult {
    /// The rewired edge list. Same length and degree sequence as the input.
    pub edges: Vec<(usize, usize)>,
    /// Swaps actually applied.
    pub swaps: usize,
    /// Swap attempts made, including the rejected ones.
    pub attempts: usize,
}

/// Order an edge's endpoints so an undirected edge has one representation.
fn canonical(u: usize, v: usize) -> (usize, usize) {
    if u <= v {
        (u, v)
    } else {
        (v, u)
    }
}

/// Rewire a graph while preserving every node's degree.
///
/// # Arguments
/// * `n` - number of nodes
/// * `edges` - undirected edge list; self-loops and parallel edges are rejected
///   because neither has a place in the space being sampled
/// * `target_swaps` - successful swaps to aim for. A common choice is a small
///   multiple of the edge count, enough to lose the original structure.
/// * `max_attempts` - cap on tries, so a graph that cannot be rewired
///   terminates instead of spinning
/// * `seed` - makes the run reproducible
///
/// Returns [`GraphError`] if an edge names a node that does not exist, if the
/// input contains a self-loop or a repeated edge, or if `max_attempts` is
/// below `target_swaps` (which could never succeed).
pub fn degree_preserving_rewire(
    n: usize,
    edges: &[(usize, usize)],
    target_swaps: usize,
    max_attempts: usize,
    seed: u64,
) -> Result<RewireResult, GraphError> {
    validate_edges(n, edges)?;
    if max_attempts < target_swaps {
        return Err(GraphError::InvalidParameter {
            name: "max_attempts",
            requirement: "must be at least target_swaps, or no run could succeed",
        });
    }

    let mut current: Vec<(usize, usize)> = edges.iter().map(|&(u, v)| canonical(u, v)).collect();
    let mut present: HashSet<(usize, usize)> = HashSet::with_capacity(current.len());

    for (index, &(u, v)) in current.iter().enumerate() {
        if u == v {
            return Err(GraphError::InvalidParameter {
                name: "edges",
                requirement: "must not contain self-loops; degree-preserving \
                              randomization samples simple graphs",
            });
        }
        if !present.insert((u, v)) {
            let _ = index;
            return Err(GraphError::InvalidParameter {
                name: "edges",
                requirement: "must not repeat an edge; degree-preserving \
                              randomization samples simple graphs",
            });
        }
    }

    let m = current.len();
    if m < 2 || target_swaps == 0 {
        // Nothing to swap against: report zero rather than pretending.
        return Ok(RewireResult {
            edges: current,
            swaps: 0,
            attempts: 0,
        });
    }

    let mut rng = SplitMix64::new(seed);
    let mut swaps = 0usize;
    let mut attempts = 0usize;

    while swaps < target_swaps && attempts < max_attempts {
        attempts += 1;

        let first = rng.below(m as u64) as usize;
        let second = rng.below(m as u64) as usize;
        if first == second {
            continue;
        }

        let (u, v) = current[first];
        let (x, y) = current[second];

        // Orient the pairing randomly, so the swap can produce either
        // (u,y),(x,v) or (u,x),(v,y) — otherwise the canonical ordering biases
        // which rewirings are ever reachable.
        let (v, y) = if rng.next_u64() & 1 == 0 {
            (v, y)
        } else {
            (y, v)
        };

        if u == y || x == v {
            continue; // would be a self-loop
        }

        let new_first = canonical(u, y);
        let new_second = canonical(x, v);
        if present.contains(&new_first) || present.contains(&new_second) {
            continue; // would duplicate an existing edge
        }

        present.remove(&current[first]);
        present.remove(&current[second]);
        present.insert(new_first);
        present.insert(new_second);
        current[first] = new_first;
        current[second] = new_second;
        swaps += 1;
    }

    Ok(RewireResult {
        edges: current,
        swaps,
        attempts,
    })
}

/// Generate several independently randomized graphs.
///
/// The samples share nothing but the immutable input, so they run in parallel.
/// Each derives its own stream from `seed`, which keeps the whole set
/// reproducible regardless of how the work is scheduled.
///
/// Returns [`GraphError`] under the same conditions as
/// [`degree_preserving_rewire`].
pub fn degree_preserving_rewire_samples(
    n: usize,
    edges: &[(usize, usize)],
    n_samples: usize,
    target_swaps: usize,
    max_attempts: usize,
    seed: u64,
) -> Result<Vec<RewireResult>, GraphError> {
    // One derived seed per sample, drawn before the fan-out so the assignment
    // does not depend on thread scheduling.
    let mut seeds = SplitMix64::new(seed);
    let sample_seeds: Vec<u64> = (0..n_samples).map(|_| seeds.next_u64()).collect();

    sample_seeds
        .par_iter()
        .map(|&sample_seed| {
            degree_preserving_rewire(n, edges, target_swaps, max_attempts, sample_seed)
        })
        .collect()
}

/// Sample a simple graph with the given degree sequence.
///
/// The configuration model lays out `degree[i]` stubs for node `i`, pairs them
/// at random, and reads off the edges. Pairings that would make a self-loop or
/// repeat an edge are discarded, so the result is simple — which means the
/// realized degrees can fall slightly short of the requested ones. `shortfall`
/// reports by how much rather than letting the caller assume an exact match.
///
/// Returns [`GraphError::InvalidParameter`] if the degree sum is odd, since
/// there is then no graph with that degree sequence.
pub fn configuration_model(
    degrees: &[usize],
    seed: u64,
) -> Result<ConfigurationResult, GraphError> {
    let total: usize = degrees.iter().sum();
    if !total.is_multiple_of(2) {
        return Err(GraphError::InvalidParameter {
            name: "degrees",
            requirement: "must sum to an even number; no graph has an odd degree sum",
        });
    }

    let mut stubs: Vec<usize> = degrees
        .iter()
        .enumerate()
        .flat_map(|(node, &d)| std::iter::repeat_n(node, d))
        .collect();

    let mut rng = SplitMix64::new(seed);
    rng.shuffle(&mut stubs);

    let mut present: HashSet<(usize, usize)> = HashSet::with_capacity(total / 2);
    let mut edges = Vec::with_capacity(total / 2);
    let mut discarded = 0usize;

    for pair in stubs.chunks_exact(2) {
        let (u, v) = (pair[0], pair[1]);
        if u == v {
            discarded += 1;
            continue;
        }
        if !present.insert(canonical(u, v)) {
            discarded += 1;
            continue;
        }
        edges.push(canonical(u, v));
    }

    Ok(ConfigurationResult {
        edges,
        discarded_pairings: discarded,
    })
}

/// One configuration-model draw, with what simplifying it cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigurationResult {
    /// The sampled edge list, simple and undirected.
    pub edges: Vec<(usize, usize)>,
    /// Stub pairings dropped because they were self-loops or repeats. Each one
    /// means two nodes ended a degree short of the sequence asked for.
    pub discarded_pairings: usize,
}

/// Sample a uniformly random simple graph with `n` nodes and `m` edges.
///
/// This is G(n, m): the same edge count as the observed graph, wired without
/// regard to degree. Sampling is by rejection, which stays efficient as long
/// as `m` is well below the complete graph.
///
/// Returns [`GraphError::InvalidParameter`] if `m` exceeds the number of
/// distinct node pairs.
pub fn erdos_renyi(n: usize, m: usize, seed: u64) -> Result<Vec<(usize, usize)>, GraphError> {
    let capacity = n.saturating_mul(n.saturating_sub(1)) / 2;
    if m > capacity {
        return Err(GraphError::InvalidParameter {
            name: "m",
            requirement: "must not exceed the number of distinct node pairs",
        });
    }
    if m == 0 {
        return Ok(Vec::new());
    }

    let mut rng = SplitMix64::new(seed);
    let mut present: HashSet<(usize, usize)> = HashSet::with_capacity(m);
    let mut edges = Vec::with_capacity(m);

    while edges.len() < m {
        let u = rng.below(n as u64) as usize;
        let v = rng.below(n as u64) as usize;
        if u == v {
            continue;
        }
        let edge = canonical(u, v);
        if present.insert(edge) {
            edges.push(edge);
        }
    }

    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::degree::degree_sequence;

    /// A graph with enough slack that swaps are actually available.
    fn ring_with_chords(n: usize) -> Vec<(usize, usize)> {
        (0..n)
            .flat_map(|i| [(i, (i + 1) % n), (i, (i + 3) % n)])
            .map(|(u, v)| canonical(u, v))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }

    #[test]
    fn degrees_are_preserved_exactly() {
        let n = 60;
        let edges = ring_with_chords(n);
        let before = degree_sequence(n, &edges, false).unwrap();

        let result = degree_preserving_rewire(n, &edges, 200, 20_000, 7).unwrap();
        let after = degree_sequence(n, &result.edges, false).unwrap();

        assert_eq!(before.to_vec(), after.to_vec());
        assert_eq!(result.edges.len(), edges.len());
    }

    #[test]
    fn the_wiring_actually_changes() {
        let n = 60;
        let edges = ring_with_chords(n);
        let result = degree_preserving_rewire(n, &edges, 200, 20_000, 7).unwrap();

        let before: HashSet<_> = edges.iter().copied().collect();
        let after: HashSet<_> = result.edges.iter().copied().collect();

        assert!(result.swaps > 0);
        assert_ne!(before, after, "a null identical to the input is no null");
    }

    #[test]
    fn no_self_loops_or_duplicates_are_created() {
        let n = 40;
        let edges = ring_with_chords(n);
        let result = degree_preserving_rewire(n, &edges, 500, 50_000, 3).unwrap();

        let unique: HashSet<_> = result.edges.iter().copied().collect();
        assert_eq!(unique.len(), result.edges.len(), "duplicate edge created");
        assert!(
            result.edges.iter().all(|&(u, v)| u != v),
            "self-loop created"
        );
    }

    #[test]
    fn the_same_seed_reproduces_the_same_graph() {
        let n = 40;
        let edges = ring_with_chords(n);

        let first = degree_preserving_rewire(n, &edges, 100, 10_000, 99).unwrap();
        let second = degree_preserving_rewire(n, &edges, 100, 10_000, 99).unwrap();
        let other = degree_preserving_rewire(n, &edges, 100, 10_000, 100).unwrap();

        assert_eq!(first, second);
        assert_ne!(first.edges, other.edges, "different seeds, same graph");
    }

    #[test]
    fn a_graph_that_cannot_be_rewired_reports_the_shortfall() {
        // Triangle plus a pendant: no swap avoids a self-loop or a duplicate.
        let edges = vec![(0, 1), (1, 2), (0, 2), (2, 3)];
        let result = degree_preserving_rewire(4, &edges, 50, 500, 1).unwrap();

        assert_eq!(result.swaps, 0);
        assert_eq!(result.attempts, 500, "should have exhausted its attempts");
        // Degrees still hold, because nothing moved.
        assert_eq!(
            degree_sequence(4, &edges, false).unwrap().to_vec(),
            degree_sequence(4, &result.edges, false).unwrap().to_vec()
        );
    }

    #[test]
    fn samples_differ_from_each_other() {
        let n = 50;
        let edges = ring_with_chords(n);
        let samples = degree_preserving_rewire_samples(n, &edges, 4, 150, 20_000, 11).unwrap();

        assert_eq!(samples.len(), 4);
        let wirings: HashSet<Vec<(usize, usize)>> = samples
            .iter()
            .map(|s| {
                let mut sorted = s.edges.clone();
                sorted.sort_unstable();
                sorted
            })
            .collect();
        assert_eq!(wirings.len(), 4, "samples should be independent draws");
    }

    #[test]
    fn samples_are_reproducible_and_order_independent() {
        let n = 40;
        let edges = ring_with_chords(n);

        let first = degree_preserving_rewire_samples(n, &edges, 5, 100, 10_000, 21).unwrap();
        let second = degree_preserving_rewire_samples(n, &edges, 5, 100, 10_000, 21).unwrap();

        assert_eq!(first, second, "parallel scheduling must not change results");
    }

    #[test]
    fn malformed_input_is_rejected() {
        // Self-loop.
        assert!(degree_preserving_rewire(3, &[(0, 0), (1, 2)], 10, 100, 1).is_err());
        // Repeated edge.
        assert!(degree_preserving_rewire(3, &[(0, 1), (1, 0)], 10, 100, 1).is_err());
        // Node outside the graph.
        assert!(degree_preserving_rewire(2, &[(0, 5)], 10, 100, 1).is_err());
        // A cap that could never reach the target.
        assert!(degree_preserving_rewire(4, &[(0, 1), (2, 3)], 100, 10, 1).is_err());
    }

    #[test]
    fn trivial_graphs_do_not_spin() {
        let result = degree_preserving_rewire(2, &[(0, 1)], 100, 1_000, 1).unwrap();
        assert_eq!(result.swaps, 0);
        assert_eq!(result.attempts, 0, "one edge has nothing to swap against");
    }

    #[test]
    fn configuration_model_approximates_the_degree_sequence() {
        let degrees = vec![3, 3, 3, 3, 3, 3, 3, 3];
        let result = configuration_model(&degrees, 5).unwrap();

        let realized = degree_sequence(degrees.len(), &result.edges, false).unwrap();
        // Simplifying can only lose degree, never add it.
        assert!(realized
            .iter()
            .zip(&degrees)
            .all(|(&got, &want)| got <= want));
        let lost: usize = degrees.iter().sum::<usize>() - realized.sum();
        assert_eq!(
            lost,
            2 * result.discarded_pairings,
            "the shortfall is reported"
        );
    }

    #[test]
    fn configuration_model_produces_a_simple_graph() {
        let degrees = vec![4; 20];
        let result = configuration_model(&degrees, 9).unwrap();

        let unique: HashSet<_> = result.edges.iter().copied().collect();
        assert_eq!(unique.len(), result.edges.len());
        assert!(result.edges.iter().all(|&(u, v)| u != v));
    }

    #[test]
    fn configuration_model_rejects_an_odd_degree_sum() {
        assert!(configuration_model(&[1, 1, 1], 1).is_err());
    }

    #[test]
    fn configuration_model_is_reproducible() {
        let degrees = vec![3; 10];
        assert_eq!(
            configuration_model(&degrees, 4).unwrap(),
            configuration_model(&degrees, 4).unwrap()
        );
        assert_ne!(
            configuration_model(&degrees, 4).unwrap(),
            configuration_model(&degrees, 5).unwrap()
        );
    }

    #[test]
    fn erdos_renyi_has_exactly_the_edges_asked_for() {
        let edges = erdos_renyi(50, 120, 3).unwrap();
        assert_eq!(edges.len(), 120);

        let unique: HashSet<_> = edges.iter().copied().collect();
        assert_eq!(unique.len(), 120, "no repeats");
        assert!(edges.iter().all(|&(u, v)| u != v), "no self-loops");
        assert!(edges.iter().all(|&(u, v)| u < 50 && v < 50));
    }

    #[test]
    fn erdos_renyi_can_fill_a_complete_graph_but_no_more() {
        assert_eq!(erdos_renyi(5, 10, 1).unwrap().len(), 10);
        assert!(erdos_renyi(5, 11, 1).is_err());
        assert!(erdos_renyi(5, 0, 1).unwrap().is_empty());
    }

    #[test]
    fn erdos_renyi_is_reproducible() {
        assert_eq!(
            erdos_renyi(30, 60, 8).unwrap(),
            erdos_renyi(30, 60, 8).unwrap()
        );
        assert_ne!(
            erdos_renyi(30, 60, 8).unwrap(),
            erdos_renyi(30, 60, 9).unwrap()
        );
    }
}
