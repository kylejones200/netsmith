//! Community detection: modularity and the Louvain method.
//!
//! Both routines operate on a weighted, undirected view of the graph. Parallel
//! edges (and reciprocal edges of a directed input) are merged by summing their
//! weights. A self-loop of weight `w` contributes `2w` to its node's degree,
//! which is the convention NetworkX uses, so modularity values match.
//!
//! Weights must be non-negative; modularity is not defined otherwise. Both
//! entry points validate that, along with the node ids, and return an error
//! rather than skipping what they cannot use.

use ndarray::Array1;

use crate::{validate_edges, validate_weights, GraphError, WeightRule};

/// Result of a Louvain run.
#[derive(Debug, Clone)]
pub struct LouvainResult {
    /// Community id per node, numbered `0..n_communities`.
    pub labels: Array1<usize>,
    /// Modularity of `labels` on the input graph.
    pub modularity: f64,
    /// Number of communities found.
    pub n_communities: usize,
    /// Number of aggregation levels performed.
    pub n_levels: usize,
}

/// Weighted undirected graph in adjacency-list form.
///
/// `adj` excludes self-loops; those live in `self_loops` so that the local
/// moving phase can ignore them (a self-loop follows its node into whichever
/// community the node joins, so it never affects a move decision).
struct WeightedGraph {
    n: usize,
    adj: Vec<Vec<(usize, f64)>>,
    self_loops: Vec<f64>,
    degrees: Vec<f64>,
    /// Total edge weight `m` (each undirected edge counted once).
    total_weight: f64,
}

impl WeightedGraph {
    /// Build from an edge list, summing duplicate and reciprocal edges.
    ///
    /// The caller validates node ids and weights first, so nothing here is
    /// skipped or defaulted.
    fn from_edges(n: usize, edges: &[(usize, usize)], weights: Option<&[f64]>) -> Self {
        let mut adj = vec![Vec::<(usize, f64)>::new(); n];
        let mut self_loops = vec![0.0f64; n];

        for (idx, &(u, v)) in edges.iter().enumerate() {
            let w = weights.map_or(1.0, |ws| ws[idx]);
            if u == v {
                self_loops[u] += w;
            } else {
                adj[u].push((v, w));
                adj[v].push((u, w));
            }
        }

        // Merge parallel edges: sort by neighbor, then collapse runs.
        for nbrs in adj.iter_mut() {
            nbrs.sort_unstable_by_key(|&(j, _)| j);
            let mut merged: Vec<(usize, f64)> = Vec::with_capacity(nbrs.len());
            for &(j, w) in nbrs.iter() {
                match merged.last_mut() {
                    Some(last) if last.0 == j => last.1 += w,
                    _ => merged.push((j, w)),
                }
            }
            *nbrs = merged;
        }

        let mut degrees = vec![0.0f64; n];
        for i in 0..n {
            let mut d = 2.0 * self_loops[i];
            for &(_, w) in adj[i].iter() {
                d += w;
            }
            degrees[i] = d;
        }
        let total_weight = 0.5 * degrees.iter().sum::<f64>();

        WeightedGraph {
            n,
            adj,
            self_loops,
            degrees,
            total_weight,
        }
    }

    /// Modularity of a partition of this graph.
    fn modularity(&self, labels: &[usize], resolution: f64) -> f64 {
        let m = self.total_weight;
        if m <= 0.0 || self.n == 0 {
            return 0.0;
        }
        let n_comms = labels.iter().copied().max().map_or(0, |c| c + 1);
        let mut internal = vec![0.0f64; n_comms];
        let mut tot = vec![0.0f64; n_comms];

        for i in 0..self.n {
            let ci = labels[i];
            tot[ci] += self.degrees[i];
            internal[ci] += self.self_loops[i];
            for &(j, w) in self.adj[i].iter() {
                if labels[j] == ci {
                    // Each intra-community edge is seen from both endpoints.
                    internal[ci] += 0.5 * w;
                }
            }
        }

        let two_m = 2.0 * m;
        (0..n_comms)
            .map(|c| internal[c] / m - resolution * (tot[c] / two_m).powi(2))
            .sum()
    }

    /// Collapse each community into a single node.
    ///
    /// Intra-community edges become the new node's self-loop, so degrees and
    /// total weight are preserved exactly.
    fn aggregate(&self, labels: &[usize], n_comms: usize) -> WeightedGraph {
        let mut adj = vec![Vec::<(usize, f64)>::new(); n_comms];
        let mut self_loops = vec![0.0f64; n_comms];

        for i in 0..self.n {
            let ci = labels[i];
            self_loops[ci] += self.self_loops[i];
            for &(j, w) in self.adj[i].iter() {
                let cj = labels[j];
                if ci == cj {
                    // Seen once from each endpoint; halve to count the edge once.
                    self_loops[ci] += 0.5 * w;
                } else {
                    adj[ci].push((cj, w));
                }
            }
        }

        for nbrs in adj.iter_mut() {
            nbrs.sort_unstable_by_key(|&(j, _)| j);
            let mut merged: Vec<(usize, f64)> = Vec::with_capacity(nbrs.len());
            for &(j, w) in nbrs.iter() {
                match merged.last_mut() {
                    Some(last) if last.0 == j => last.1 += w,
                    _ => merged.push((j, w)),
                }
            }
            *nbrs = merged;
        }

        let mut degrees = vec![0.0f64; n_comms];
        for c in 0..n_comms {
            let mut d = 2.0 * self_loops[c];
            for &(_, w) in adj[c].iter() {
                d += w;
            }
            degrees[c] = d;
        }

        WeightedGraph {
            n: n_comms,
            adj,
            self_loops,
            degrees,
            total_weight: self.total_weight,
        }
    }
}

/// SplitMix64 — a small deterministic PRNG for the node visit order.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `[0, bound)` via Lemire's debiased multiply-shift.
    fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let mut product = (self.next_u64() as u128).wrapping_mul(bound as u128);
        let mut low = product as u64;
        if low < bound {
            let threshold = (u64::MAX - bound + 1) % bound;
            while low < threshold {
                product = (self.next_u64() as u128).wrapping_mul(bound as u128);
                low = product as u64;
            }
        }
        (product >> 64) as u64
    }

    fn shuffle(&mut self, items: &mut [usize]) {
        for i in (1..items.len()).rev() {
            let j = self.below((i + 1) as u64) as usize;
            items.swap(i, j);
        }
    }
}

/// Relabel a partition so community ids are `0..k`, ordered by first appearance.
fn renumber(labels: &mut [usize], n_old: usize) -> usize {
    let mut mapping = vec![usize::MAX; n_old];
    let mut next = 0usize;
    for label in labels.iter_mut() {
        if mapping[*label] == usize::MAX {
            mapping[*label] = next;
            next += 1;
        }
        *label = mapping[*label];
    }
    next
}

/// Guard against float-noise moves cycling forever; strict improvement means
/// real runs stop long before this.
const MAX_PASSES: usize = 1_000;

/// One local-moving phase: repeatedly move single nodes to the neighbouring
/// community that most improves modularity, until no node moves.
///
/// Returns the total modularity gain achieved.
fn local_moving(
    graph: &WeightedGraph,
    labels: &mut [usize],
    resolution: f64,
    rng: &mut Option<SplitMix64>,
) -> f64 {
    let m = graph.total_weight;
    if m <= 0.0 {
        return 0.0;
    }
    let two_m = 2.0 * m;

    let mut tot = vec![0.0f64; graph.n];
    for i in 0..graph.n {
        tot[labels[i]] += graph.degrees[i];
    }

    let mut order: Vec<usize> = (0..graph.n).collect();
    if let Some(r) = rng.as_mut() {
        r.shuffle(&mut order);
    }

    // Scratch buffers reused across nodes: weight from the current node to each
    // candidate community, plus the list of communities touched this round.
    let mut weight_to = vec![0.0f64; graph.n];
    let mut touched: Vec<usize> = Vec::new();

    let mut total_gain = 0.0;
    for _pass in 0..MAX_PASSES {
        let mut moves = 0usize;
        let mut pass_gain = 0.0;

        for &i in order.iter() {
            let k_i = graph.degrees[i];
            let old_comm = labels[i];

            for &c in touched.iter() {
                weight_to[c] = 0.0;
            }
            touched.clear();

            for &(j, w) in graph.adj[i].iter() {
                let cj = labels[j];
                if weight_to[cj] == 0.0 {
                    touched.push(cj);
                }
                weight_to[cj] += w;
            }

            // Isolate the node before scoring, so the "stay" option is scored
            // on the same footing as every move.
            tot[old_comm] -= k_i;

            let stay_gain = weight_to[old_comm] - resolution * k_i * tot[old_comm] / two_m;
            let mut best_comm = old_comm;
            let mut best_gain = stay_gain;
            for &c in touched.iter() {
                if c == old_comm {
                    continue;
                }
                let gain = weight_to[c] - resolution * k_i * tot[c] / two_m;
                if gain > best_gain {
                    best_gain = gain;
                    best_comm = c;
                }
            }

            tot[best_comm] += k_i;
            labels[i] = best_comm;

            if best_comm != old_comm {
                moves += 1;
                pass_gain += (best_gain - stay_gain) / m;
            }
        }

        total_gain += pass_gain;
        if moves == 0 {
            break;
        }
    }

    total_gain
}

/// Compute modularity of a partition.
///
/// # Arguments
/// * `n` - number of nodes
/// * `edges` - edge list; duplicates and reciprocal pairs are summed
/// * `weights` - per-edge weights, or `None` for unweighted (all 1.0)
/// * `labels` - community id per node, length `n`
/// * `resolution` - resolution parameter `gamma` (1.0 = standard modularity)
///
/// Returns 0.0 for an empty graph or one with no edge weight.
///
/// Returns [`GraphError`] if an edge names a node that does not exist, if the
/// weights do not line up with the edges, or if `labels` does not have one
/// entry per node.
pub fn modularity(
    n: usize,
    edges: &[(usize, usize)],
    weights: Option<&[f64]>,
    labels: &[usize],
    resolution: f64,
) -> Result<f64, GraphError> {
    validate_edges(n, edges)?;
    validate_weights(edges.len(), weights, WeightRule::NonNegative)?;
    validate_resolution(resolution)?;
    if labels.len() != n {
        return Err(GraphError::LabelsLengthMismatch {
            labels: labels.len(),
            n_nodes: n,
        });
    }

    let graph = WeightedGraph::from_edges(n, edges, weights);
    Ok(graph.modularity(labels, resolution))
}

/// Reject a resolution that would make modularity meaningless.
fn validate_resolution(resolution: f64) -> Result<(), GraphError> {
    if resolution > 0.0 && resolution.is_finite() {
        Ok(())
    } else {
        Err(GraphError::InvalidParameter {
            name: "resolution",
            requirement: "must be finite and strictly positive",
        })
    }
}

/// Detect communities with the Louvain method.
///
/// Alternates a local-moving phase (greedy single-node moves that increase
/// modularity) with an aggregation phase (each community becomes one node),
/// until a level yields no further improvement.
///
/// # Arguments
/// * `n` - number of nodes
/// * `edges` - edge list; duplicates and reciprocal pairs are summed
/// * `weights` - per-edge weights, or `None` for unweighted (all 1.0)
/// * `resolution` - resolution `gamma`; higher values yield smaller communities
/// * `seed` - `Some(s)` randomizes node visit order deterministically from `s`;
///   `None` visits nodes in index order (also deterministic)
/// * `max_levels` - cap on aggregation levels
///
/// A graph with no edges (or all-zero weights) yields one community per node
/// and modularity 0.0.
///
/// Returns [`GraphError`] if an edge names a node that does not exist, if the
/// weights do not line up with the edges, or if `resolution` or `max_levels`
/// is out of range.
pub fn louvain(
    n: usize,
    edges: &[(usize, usize)],
    weights: Option<&[f64]>,
    resolution: f64,
    seed: Option<u64>,
    max_levels: usize,
) -> Result<LouvainResult, GraphError> {
    validate_edges(n, edges)?;
    validate_weights(edges.len(), weights, WeightRule::NonNegative)?;
    validate_resolution(resolution)?;
    if max_levels == 0 {
        return Err(GraphError::InvalidParameter {
            name: "max_levels",
            requirement: "must be at least 1",
        });
    }

    let original = WeightedGraph::from_edges(n, edges, weights);

    if n == 0 {
        return Ok(LouvainResult {
            labels: Array1::from_vec(Vec::new()),
            modularity: 0.0,
            n_communities: 0,
            n_levels: 0,
        });
    }

    let mut node_labels: Vec<usize> = (0..n).collect();

    if original.total_weight <= 0.0 {
        return Ok(LouvainResult {
            labels: Array1::from_vec(node_labels),
            modularity: 0.0,
            n_communities: n,
            n_levels: 0,
        });
    }

    let mut rng = seed.map(SplitMix64::new);
    let mut graph = WeightedGraph::from_edges(n, edges, weights);
    let mut n_levels = 0usize;

    for _ in 0..max_levels {
        let mut labels: Vec<usize> = (0..graph.n).collect();
        local_moving(&graph, &mut labels, resolution, &mut rng);
        let n_comms = renumber(&mut labels, graph.n);

        // No community merged: further levels cannot change anything.
        if n_comms == graph.n {
            break;
        }

        for label in node_labels.iter_mut() {
            *label = labels[*label];
        }
        n_levels += 1;

        graph = graph.aggregate(&labels, n_comms);
    }

    let n_communities = renumber(&mut node_labels, n);
    let modularity = original.modularity(&node_labels, resolution);

    Ok(LouvainResult {
        labels: Array1::from_vec(node_labels),
        modularity,
        n_communities,
        n_levels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two disjoint triangles joined by a single bridge edge.
    fn two_triangles() -> (usize, Vec<(usize, usize)>) {
        let edges = vec![
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (2, 3), // bridge
        ];
        (6, edges)
    }

    /// Four cliques of `size` nodes each, joined in a ring by single edges.
    fn ring_of_cliques(n_cliques: usize, size: usize) -> (usize, Vec<(usize, usize)>) {
        let n = n_cliques * size;
        let mut edges = Vec::new();
        for c in 0..n_cliques {
            let base = c * size;
            for i in 0..size {
                for j in (i + 1)..size {
                    edges.push((base + i, base + j));
                }
            }
            let next = ((c + 1) % n_cliques) * size;
            edges.push((base, next));
        }
        (n, edges)
    }

    #[test]
    fn modularity_matches_hand_computation() {
        // Path 0-1-2: m = 2, degrees 1, 2, 1.
        let edges = vec![(0, 1), (1, 2)];
        // All in one community: internal = 2, tot = 4 => 2/2 - (4/4)^2 = 0.0
        assert_relative_eq!(
            modularity(3, &edges, None, &[0, 0, 0], 1.0).unwrap(),
            0.0,
            epsilon = 1e-12
        );
        // Split {0,1} and {2}: internal = 1 and 0; tot = 3 and 1
        // => 1/2 - (3/4)^2 + 0 - (1/4)^2 = 0.5 - 0.5625 - 0.0625 = -0.125
        assert_relative_eq!(
            modularity(3, &edges, None, &[0, 0, 1], 1.0).unwrap(),
            -0.125,
            epsilon = 1e-12
        );
    }

    #[test]
    fn modularity_counts_self_loops_once_internally() {
        // Single node with a self-loop of weight 1: m = 1, degree = 2.
        let edges = vec![(0, 0)];
        // internal = 1, tot = 2 => 1/1 - (2/2)^2 = 0.0
        assert_relative_eq!(
            modularity(1, &edges, None, &[0], 1.0).unwrap(),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn empty_and_edgeless_graphs_are_handled() {
        let result = louvain(0, &[], None, 1.0, None, 10).unwrap();
        assert_eq!(result.n_communities, 0);
        assert_eq!(result.labels.len(), 0);

        let result = louvain(4, &[], None, 1.0, None, 10).unwrap();
        assert_eq!(result.n_communities, 4);
        assert_relative_eq!(result.modularity, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn finds_two_triangles() {
        let (n, edges) = two_triangles();
        let result = louvain(n, &edges, None, 1.0, Some(42), 10).unwrap();
        assert_eq!(result.n_communities, 2);
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[2], result.labels[3]);
        // Hand check: m = 7, each community internal = 3, tot = 7
        // => 2 * (3/7 - (7/14)^2) = 2 * (3/7 - 0.25) = 0.357142857...
        assert_relative_eq!(result.modularity, 2.0 * (3.0 / 7.0 - 0.25), epsilon = 1e-12);
    }

    #[test]
    fn finds_ring_of_cliques() {
        let (n, edges) = ring_of_cliques(4, 6);
        let result = louvain(n, &edges, None, 1.0, Some(7), 20).unwrap();
        assert_eq!(result.n_communities, 4);
        for c in 0..4 {
            let base = c * 6;
            for i in 1..6 {
                assert_eq!(result.labels[base], result.labels[base + i]);
            }
        }
        assert!(result.modularity > 0.6, "got {}", result.modularity);
    }

    #[test]
    fn labels_are_consecutive_from_zero() {
        let (n, edges) = ring_of_cliques(4, 6);
        let result = louvain(n, &edges, None, 1.0, Some(3), 20).unwrap();
        let mut seen = vec![false; result.n_communities];
        for &label in result.labels.iter() {
            assert!(label < result.n_communities);
            seen[label] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn reported_modularity_matches_recomputation() {
        let (n, edges) = ring_of_cliques(5, 5);
        let result = louvain(n, &edges, None, 1.0, Some(11), 20).unwrap();
        let labels: Vec<usize> = result.labels.to_vec();
        assert_relative_eq!(
            result.modularity,
            modularity(n, &edges, None, &labels, 1.0).unwrap(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn same_seed_gives_same_partition() {
        let (n, edges) = ring_of_cliques(4, 5);
        let a = louvain(n, &edges, None, 1.0, Some(99), 20).unwrap();
        let b = louvain(n, &edges, None, 1.0, Some(99), 20).unwrap();
        assert_eq!(a.labels.to_vec(), b.labels.to_vec());
        assert_relative_eq!(a.modularity, b.modularity, epsilon = 1e-15);
    }

    #[test]
    fn resolution_controls_community_size() {
        // 4 cliques of 6, joined in a ring. At gamma = 1 the cliques are the
        // natural partition; below it they merge, above it they split.
        let (n, edges) = ring_of_cliques(4, 6);
        let coarse = louvain(n, &edges, None, 0.05, Some(5), 20).unwrap();
        let natural = louvain(n, &edges, None, 1.0, Some(5), 20).unwrap();
        let fine = louvain(n, &edges, None, 8.0, Some(5), 20).unwrap();
        assert!(
            coarse.n_communities < natural.n_communities,
            "coarse {} natural {}",
            coarse.n_communities,
            natural.n_communities
        );
        assert_eq!(natural.n_communities, 4);
        assert!(
            fine.n_communities > natural.n_communities,
            "fine {} natural {}",
            fine.n_communities,
            natural.n_communities
        );
    }

    #[test]
    fn weights_drive_the_partition() {
        // Square 0-1-2-3-0; heavy weights on 0-1 and 2-3 should pair those up.
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let weights = vec![10.0, 0.1, 10.0, 0.1];
        let result = louvain(4, &edges, Some(&weights), 1.0, Some(1), 10).unwrap();
        assert_eq!(result.n_communities, 2);
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
    }

    #[test]
    fn parallel_and_reciprocal_edges_are_summed() {
        // Two representations of the same weighted graph must agree.
        let split = vec![(0, 1), (1, 0), (1, 2)];
        let split_w = vec![0.5, 0.5, 1.0];
        let merged = vec![(0, 1), (1, 2)];
        let merged_w = vec![1.0, 1.0];
        assert_relative_eq!(
            modularity(3, &split, Some(&split_w), &[0, 0, 1], 1.0).unwrap(),
            modularity(3, &merged, Some(&merged_w), &[0, 0, 1], 1.0).unwrap(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn out_of_range_edges_are_rejected_not_skipped() {
        // Dropping edge (1, 9) would answer a question about a different graph.
        let edges = vec![(0, 1), (1, 9), (0, 2)];
        assert_eq!(
            louvain(3, &edges, None, 1.0, Some(1), 10).unwrap_err(),
            GraphError::NodeOutOfRange {
                index: 1,
                u: 1,
                v: 9,
                n_nodes: 3,
            }
        );
        assert!(modularity(3, &edges, None, &[0, 0, 0], 1.0).is_err());
    }

    #[test]
    fn bad_parameters_are_rejected() {
        let edges = vec![(0, 1), (1, 2)];
        assert!(louvain(3, &edges, None, 0.0, None, 10).is_err());
        assert!(louvain(3, &edges, None, f64::NAN, None, 10).is_err());
        assert!(louvain(3, &edges, None, 1.0, None, 0).is_err());
        assert!(louvain(3, &edges, Some(&[1.0]), 1.0, None, 10).is_err());
        assert!(louvain(3, &edges, Some(&[1.0, -1.0]), 1.0, None, 10).is_err());
        assert!(modularity(3, &edges, None, &[0, 0], 1.0).is_err());
    }
}
