//! Centrality measures.
//!
//! Betweenness uses Brandes' algorithm (2001), which computes exact betweenness
//! in O(n·m) for unweighted graphs and O(n·m + n²·log n) for weighted ones —
//! one shortest-path sweep per source, then a single back-propagation of
//! dependencies down each sweep's DAG.
//!
//! The sweeps are independent, so they run in parallel across sources.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use ndarray::Array1;
use rayon::prelude::*;

/// A weighted directed adjacency list, self-loops removed.
///
/// Self-loops can never lie on a shortest path between two distinct nodes, and
/// parallel edges collapse to their lightest member for the same reason.
struct AdjacencyList {
    n: usize,
    neighbours: Vec<Vec<(usize, f64)>>,
}

impl AdjacencyList {
    fn build(
        n: usize,
        edges: &[(usize, usize)],
        weights: Option<&[f64]>,
        directed: bool,
    ) -> AdjacencyList {
        let mut neighbours = vec![Vec::<(usize, f64)>::new(); n];

        for (idx, &(u, v)) in edges.iter().enumerate() {
            if u >= n || v >= n || u == v {
                continue;
            }
            let w = weights.map_or(1.0, |ws| ws[idx]);
            neighbours[u].push((v, w));
            if !directed {
                neighbours[v].push((u, w));
            }
        }

        // Collapse parallel edges to the lightest one.
        for nbrs in neighbours.iter_mut() {
            nbrs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
            nbrs.dedup_by_key(|&mut (j, _)| j);
        }

        AdjacencyList { n, neighbours }
    }
}

/// Heap entry ordered by distance, smallest first.
///
/// Distances are finite and non-negative (the caller validates weights), so the
/// partial order is total.
#[derive(PartialEq)]
struct Visit {
    distance: f64,
    predecessor: usize,
    node: usize,
}

impl Eq for Visit {}

impl Ord for Visit {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: BinaryHeap is a max-heap and we want the nearest node.
        other.distance.total_cmp(&self.distance)
    }
}

impl PartialOrd for Visit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Per-worker buffers, reused across the sources one worker handles.
struct Sweep {
    /// Betweenness accumulated so far by this worker.
    accumulator: Vec<f64>,
    /// Number of shortest paths from the source to each node.
    sigma: Vec<f64>,
    /// Dependency of the source on each node.
    delta: Vec<f64>,
    /// Predecessors of each node on shortest paths from the source.
    predecessors: Vec<Vec<usize>>,
    /// Nodes in non-decreasing distance order.
    order: Vec<usize>,
    /// Hop distance for the unweighted sweep; -1 means unreached.
    hops: Vec<i64>,
    /// Tentative distance for the weighted sweep.
    tentative: Vec<f64>,
    /// Whether a node's distance is final (weighted sweep).
    settled: Vec<bool>,
    queue: VecDeque<usize>,
    heap: BinaryHeap<Visit>,
}

impl Sweep {
    fn new(n: usize) -> Sweep {
        Sweep {
            accumulator: vec![0.0; n],
            sigma: vec![0.0; n],
            delta: vec![0.0; n],
            predecessors: vec![Vec::new(); n],
            order: Vec::with_capacity(n),
            hops: vec![-1; n],
            tentative: vec![f64::INFINITY; n],
            settled: vec![false; n],
            queue: VecDeque::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Clear the per-source state, touching only the nodes the last sweep saw.
    fn reset(&mut self) {
        for &node in self.order.iter() {
            self.sigma[node] = 0.0;
            self.delta[node] = 0.0;
            self.predecessors[node].clear();
            self.hops[node] = -1;
            self.tentative[node] = f64::INFINITY;
            self.settled[node] = false;
        }
        self.order.clear();
        self.queue.clear();
        self.heap.clear();
    }

    /// BFS from `source`, filling `order`, `sigma` and `predecessors`.
    fn shortest_paths_unweighted(&mut self, graph: &AdjacencyList, source: usize) {
        self.sigma[source] = 1.0;
        self.hops[source] = 0;
        self.queue.push_back(source);

        while let Some(v) = self.queue.pop_front() {
            self.order.push(v);
            for &(w, _) in graph.neighbours[v].iter() {
                if self.hops[w] < 0 {
                    self.hops[w] = self.hops[v] + 1;
                    self.queue.push_back(w);
                }
                if self.hops[w] == self.hops[v] + 1 {
                    self.sigma[w] += self.sigma[v];
                    self.predecessors[w].push(v);
                }
            }
        }
    }

    /// Dijkstra from `source`, filling `order`, `sigma` and `predecessors`.
    fn shortest_paths_weighted(&mut self, graph: &AdjacencyList, source: usize) {
        self.sigma[source] = 1.0;
        self.tentative[source] = 0.0;
        self.heap.push(Visit {
            distance: 0.0,
            predecessor: source,
            node: source,
        });

        while let Some(Visit {
            distance,
            predecessor,
            node: v,
        }) = self.heap.pop()
        {
            if self.settled[v] {
                continue;
            }
            if v != source {
                self.sigma[v] += self.sigma[predecessor];
            }
            self.settled[v] = true;
            self.order.push(v);

            for &(w, weight) in graph.neighbours[v].iter() {
                if self.settled[w] {
                    continue;
                }
                let through_v = distance + weight;
                if through_v < self.tentative[w] {
                    // A strictly better route discards everything found so far.
                    self.tentative[w] = through_v;
                    self.sigma[w] = 0.0;
                    self.predecessors[w].clear();
                    self.predecessors[w].push(v);
                    self.heap.push(Visit {
                        distance: through_v,
                        predecessor: v,
                        node: w,
                    });
                } else if through_v == self.tentative[w] {
                    // An equally short route: another way to reach w.
                    self.sigma[w] += self.sigma[v];
                    self.predecessors[w].push(v);
                }
            }
        }
    }

    /// Walk the sweep back to front, accumulating each node's dependency.
    fn accumulate(&mut self, source: usize) {
        while let Some(w) = self.order.pop() {
            let coefficient = (1.0 + self.delta[w]) / self.sigma[w];
            for &v in self.predecessors[w].iter() {
                self.delta[v] += self.sigma[v] * coefficient;
            }
            if w != source {
                self.accumulator[w] += self.delta[w];
            }
        }
    }

    /// One source's full contribution to betweenness.
    fn run(&mut self, graph: &AdjacencyList, source: usize, weighted: bool) {
        self.reset();
        if weighted {
            self.shortest_paths_weighted(graph, source);
        } else {
            self.shortest_paths_unweighted(graph, source);
        }
        // `accumulate` drains `order`, so re-record it for the next reset.
        let visited: Vec<usize> = self.order.clone();
        self.accumulate(source);
        self.order = visited;
    }
}

/// Scale raw betweenness the way NetworkX does.
fn rescale(betweenness: &mut [f64], n: usize, normalized: bool, directed: bool) {
    let scale = if normalized {
        if n <= 2 {
            None
        } else {
            Some(1.0 / ((n - 1) as f64 * (n - 2) as f64))
        }
    } else if !directed {
        // Every pair is swept from both endpoints.
        Some(0.5)
    } else {
        None
    };

    if let Some(scale) = scale {
        for value in betweenness.iter_mut() {
            *value *= scale;
        }
    }
}

/// Compute betweenness centrality with Brandes' algorithm.
///
/// Betweenness is the share of shortest paths between other pairs of nodes that
/// pass through a node — a measure of brokerage. Endpoints are excluded, and
/// pairs with no path between them simply contribute nothing.
///
/// # Arguments
/// * `n` - number of nodes
/// * `edges` - edge list; self-loops are ignored and parallel edges collapse to
///   the lightest one
/// * `weights` - per-edge weights read as *distances* (a heavier edge is a
///   longer step), or `None` to treat every edge as length 1. Weights must be
///   strictly positive; the caller validates this.
/// * `directed` - follow edges only from `u` to `v`
/// * `normalized` - divide by the number of ordered pairs excluding the node,
///   `1/((n-1)(n-2))`, giving scores in [0, 1]. When false, an undirected graph
///   is halved instead, since each pair is swept twice.
///
/// Matches `networkx.betweenness_centrality` for the same arguments.
///
/// # Panics
/// If `weights` is shorter than `edges`.
pub fn betweenness(
    n: usize,
    edges: &[(usize, usize)],
    weights: Option<&[f64]>,
    directed: bool,
    normalized: bool,
) -> Array1<f64> {
    if let Some(ws) = weights {
        assert!(
            ws.len() >= edges.len(),
            "weights ({}) must be at least as long as edges ({})",
            ws.len(),
            edges.len()
        );
    }
    if n == 0 {
        return Array1::zeros(0);
    }

    let graph = AdjacencyList::build(n, edges, weights, directed);
    let weighted = weights.is_some();

    // Sources are independent; each worker keeps a private accumulator.
    let mut totals = (0..graph.n)
        .into_par_iter()
        .fold(
            || Sweep::new(graph.n),
            |mut sweep, source| {
                sweep.run(&graph, source, weighted);
                sweep
            },
        )
        .map(|sweep| sweep.accumulator)
        .reduce(
            || vec![0.0; graph.n],
            |mut left, right| {
                for (l, r) in left.iter_mut().zip(right.iter()) {
                    *l += r;
                }
                left
            },
        );

    rescale(&mut totals, n, normalized, directed);
    Array1::from_vec(totals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Star with node 0 at the centre and `leaves` spokes.
    fn star(leaves: usize) -> (usize, Vec<(usize, usize)>) {
        (leaves + 1, (1..=leaves).map(|i| (0, i)).collect())
    }

    /// Path 0-1-2-...-(n-1).
    fn path(n: usize) -> (usize, Vec<(usize, usize)>) {
        (n, (0..n - 1).map(|i| (i, i + 1)).collect())
    }

    #[test]
    fn star_centre_takes_everything() {
        let (n, edges) = star(4);
        let bc = betweenness(n, &edges, None, false, true);
        // Every shortest path between two leaves runs through the centre.
        assert_relative_eq!(bc[0], 1.0, epsilon = 1e-12);
        for leaf in 1..n {
            assert_relative_eq!(bc[leaf], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn path_middle_beats_the_ends() {
        let (n, edges) = path(5);
        let bc = betweenness(n, &edges, None, false, false);
        // Unnormalized: node 2 lies between {0,1} and {3,4} => 2*2 = 4 pairs.
        assert_relative_eq!(bc[2], 4.0, epsilon = 1e-12);
        assert_relative_eq!(bc[1], 3.0, epsilon = 1e-12);
        assert_relative_eq!(bc[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(bc[4], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn ties_split_credit_between_equal_paths() {
        // A 4-cycle: 0 and 2 are joined by two equally short routes, so each of
        // 1 and 3 carries half of that pair's path.
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let bc = betweenness(4, &edges, None, false, false);
        for node in 0..4 {
            assert_relative_eq!(bc[node], 0.5, epsilon = 1e-12);
        }
    }

    #[test]
    fn complete_graph_has_no_brokers() {
        // Everyone is adjacent, so no node sits between any other two.
        let mut edges = Vec::new();
        for i in 0..5 {
            for j in (i + 1)..5 {
                edges.push((i, j));
            }
        }
        let bc = betweenness(5, &edges, None, false, true);
        for node in 0..5 {
            assert_relative_eq!(bc[node], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn disconnected_components_only_count_their_own_pairs() {
        // Two separate paths; each middle brokers only its own component.
        let edges = vec![(0, 1), (1, 2), (3, 4), (4, 5)];
        let bc = betweenness(6, &edges, None, false, false);
        assert_relative_eq!(bc[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(bc[4], 1.0, epsilon = 1e-12);
        assert_relative_eq!(bc[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn weights_are_distances_and_can_reroute_paths() {
        // Triangle 0-1-2 where the direct 0-2 edge is long: unweighted, 0->2 is
        // one hop and node 1 brokers nothing; weighted, the route through 1 is
        // shorter, so node 1 brokers the pair.
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let unweighted = betweenness(3, &edges, None, false, false);
        assert_relative_eq!(unweighted[1], 0.0, epsilon = 1e-12);

        let weights = vec![1.0, 1.0, 10.0];
        let weighted = betweenness(3, &edges, Some(&weights), false, false);
        assert_relative_eq!(weighted[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn direction_is_respected() {
        // 0 -> 1 -> 2 only: node 1 brokers the single ordered pair (0, 2).
        let edges = vec![(0, 1), (1, 2)];
        let directed = betweenness(3, &edges, None, true, false);
        assert_relative_eq!(directed[1], 1.0, epsilon = 1e-12);

        // Undirected, the same pair is swept from both ends and then halved.
        let undirected = betweenness(3, &edges, None, false, false);
        assert_relative_eq!(undirected[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn normalization_divides_by_the_pair_count() {
        let (n, edges) = path(5);
        let raw = betweenness(n, &edges, None, false, false);
        let normalized = betweenness(n, &edges, None, false, true);
        let pairs = ((n - 1) * (n - 2)) as f64;
        for node in 0..n {
            // Raw scores are already halved for undirected input, so the
            // normalized value is twice the raw one over the pair count.
            assert_relative_eq!(normalized[node], 2.0 * raw[node] / pairs, epsilon = 1e-12);
        }
    }

    #[test]
    fn tiny_graphs_are_handled() {
        assert_eq!(betweenness(0, &[], None, false, true).len(), 0);
        assert_relative_eq!(betweenness(1, &[], None, false, true)[0], 0.0);
        let two = betweenness(2, &[(0, 1)], None, false, true);
        assert_relative_eq!(two[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(two[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn self_loops_and_parallel_edges_are_ignored() {
        let (n, edges) = path(5);
        let mut messy = edges.clone();
        messy.push((2, 2));
        messy.push((1, 2));

        assert_eq!(
            betweenness(n, &edges, None, false, true).to_vec(),
            betweenness(n, &messy, None, false, true).to_vec()
        );
    }

    #[test]
    fn parallel_and_serial_sweeps_agree() {
        // A graph large enough that rayon actually splits the source range.
        let n = 200;
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n));
            edges.push((i, (i + 7) % n));
        }
        let parallel = betweenness(n, &edges, None, false, true);

        let graph = AdjacencyList::build(n, &edges, None, false);
        let mut sweep = Sweep::new(n);
        for source in 0..n {
            sweep.run(&graph, source, false);
        }
        let mut serial = sweep.accumulator.clone();
        rescale(&mut serial, n, true, false);

        for node in 0..n {
            assert_relative_eq!(parallel[node], serial[node], epsilon = 1e-9);
        }
    }
}
