//! Network metrics computation

use ndarray::Array1;
use super::build_adjacency_list;

/// Build an undirected adjacency list with self-loops removed.
///
/// Triangle and clustering counts are defined on simple graphs: a self-loop is
/// not a neighbour, and leaving it in both inflates a node's degree and creates
/// triangles that do not exist.
fn simple_adjacency(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adj = build_adjacency_list(n, edges, true); // undirected
    for (u, nbrs) in adj.iter_mut().enumerate() {
        nbrs.retain(|&v| v != u);
    }
    adj
}

/// Count triangles per node
pub fn triangles_per_node(n: usize, edges: &[(usize, usize)]) -> Array1<usize> {
    let adj = simple_adjacency(n, edges);
    let mut tri = Array1::zeros(n);
    
    for u in 0..n {
        let nu = &adj[u];
        for &v in nu.iter() {
            if v <= u {
                continue;
            }
            // Count common neighbors
            let mut i = 0usize;
            let mut j = 0usize;
            let mut c = 0usize;
            while i < nu.len() && j < adj[v].len() {
                if nu[i] == adj[v][j] {
                    if nu[i] != u && nu[i] != v {
                        c += 1;
                    }
                    i += 1;
                    j += 1;
                } else if nu[i] < adj[v][j] {
                    i += 1;
                } else {
                    j += 1;
                }
            }
            tri[u] += c;
            tri[v] += c;
        }
    }

    // A triangle {u, a, b} is reached once from each of its edges incident to
    // the node, so every count above is exactly twice the real one.
    tri.mapv_inplace(|c| c / 2);
    tri
}

/// Compute average clustering coefficient
pub fn average_clustering(n: usize, edges: &[(usize, usize)]) -> f64 {
    let adj = simple_adjacency(n, edges);
    let mut s = 0.0;
    
    for u in 0..n {
        let k = adj[u].len();
        // Nodes with fewer than two neighbours have coefficient 0 and still
        // count towards the average, so this equals local_clustering().mean().
        if k < 2 {
            continue;
        }
        let mut tri = 0usize;
        for i in 0..k {
            let a = adj[u][i];
            for j in (i + 1)..k {
                let b = adj[u][j];
                // Check if edge a-b exists
                if adj[a].binary_search(&b).is_ok() {
                    tri += 1;
                }
            }
        }
        s += (2.0 * tri as f64) / ((k * (k - 1)) as f64);
    }
    
    if n > 0 {
        s / (n as f64)
    } else {
        0.0
    }
}

/// Compute local clustering coefficients
pub fn local_clustering(n: usize, edges: &[(usize, usize)]) -> Array1<f64> {
    let adj = simple_adjacency(n, edges);
    let mut clustering = Array1::zeros(n);
    
    for u in 0..n {
        let k = adj[u].len();
        if k < 2 {
            clustering[u] = 0.0;
            continue;
        }
        let mut tri = 0usize;
        for i in 0..k {
            let a = adj[u][i];
            for j in (i + 1)..k {
                let b = adj[u][j];
                if adj[a].binary_search(&b).is_ok() {
                    tri += 1;
                }
            }
        }
        clustering[u] = (2.0 * tri as f64) / ((k * (k - 1)) as f64);
    }
    
    clustering
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A triangle (0-1-2) with a pendant node 3 hanging off node 0.
    fn triangle_with_tail() -> (usize, Vec<(usize, usize)>) {
        (4, vec![(0, 1), (1, 2), (0, 2), (0, 3)])
    }

    #[test]
    fn triangles_are_counted_per_node() {
        let (n, edges) = triangle_with_tail();
        let tri = triangles_per_node(n, &edges);
        assert_eq!(tri.to_vec(), vec![1, 1, 1, 0]);
    }

    #[test]
    fn local_clustering_matches_hand_computation() {
        let (n, edges) = triangle_with_tail();
        let c = local_clustering(n, &edges);
        // Node 0 has neighbours {1, 2, 3} and one edge among them: 2*1/(3*2).
        assert_relative_eq!(c[0], 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(c[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[2], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[3], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn average_clustering_is_the_mean_over_all_nodes() {
        let (n, edges) = triangle_with_tail();
        let local = local_clustering(n, &edges);
        let mean = local.iter().sum::<f64>() / n as f64;
        assert_relative_eq!(average_clustering(n, &edges), mean, epsilon = 1e-12);
    }

    #[test]
    fn self_loops_do_not_affect_clustering() {
        let (n, edges) = triangle_with_tail();
        let mut with_loops = edges.clone();
        with_loops.push((0, 0));
        with_loops.push((3, 3));

        assert_eq!(
            local_clustering(n, &edges).to_vec(),
            local_clustering(n, &with_loops).to_vec()
        );
        assert_eq!(
            triangles_per_node(n, &edges).to_vec(),
            triangles_per_node(n, &with_loops).to_vec()
        );
        assert_relative_eq!(
            average_clustering(n, &edges),
            average_clustering(n, &with_loops),
            epsilon = 1e-12
        );
    }

    #[test]
    fn empty_graph_has_zero_average_clustering() {
        assert_relative_eq!(average_clustering(0, &[]), 0.0, epsilon = 1e-12);
        assert_relative_eq!(average_clustering(5, &[]), 0.0, epsilon = 1e-12);
    }
}
