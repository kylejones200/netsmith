//! Triangle and clustering metrics.
//!
//! All three are defined on the simple undirected graph: self-loops are not
//! neighbours, and leaving them in would both inflate a node's degree and
//! invent triangles that do not exist.

use ndarray::Array1;

use crate::{build_adjacency_list, GraphError};

/// Build the undirected adjacency list with self-loops removed.
fn simple_adjacency(n: usize, edges: &[(usize, usize)]) -> Result<Vec<Vec<usize>>, GraphError> {
    let mut adjacency = build_adjacency_list(n, edges, true)?;
    adjacency
        .iter_mut()
        .enumerate()
        .for_each(|(u, neighbours)| neighbours.retain(|&v| v != u));
    Ok(adjacency)
}

/// Count the edges among a node's neighbours.
///
/// Each such edge closes exactly one triangle through the node, so this is the
/// node's triangle count. Neighbour lists are sorted, hence the binary search.
fn triangles_among(neighbours: &[usize], adjacency: &[Vec<usize>]) -> usize {
    neighbours
        .iter()
        .enumerate()
        .flat_map(|(i, &a)| neighbours[i + 1..].iter().map(move |&b| (a, b)))
        .filter(|&(a, b)| adjacency[a].binary_search(&b).is_ok())
        .count()
}

/// Count the triangles each node takes part in.
///
/// Returns [`GraphError::NodeOutOfRange`] if any edge names a node that does
/// not exist.
pub fn triangles_per_node(n: usize, edges: &[(usize, usize)]) -> Result<Array1<usize>, GraphError> {
    let adjacency = simple_adjacency(n, edges)?;
    Ok(adjacency
        .iter()
        .map(|neighbours| triangles_among(neighbours, &adjacency))
        .collect())
}

/// Compute the local clustering coefficient of every node.
///
/// The fraction of a node's neighbour pairs that are themselves connected.
/// Nodes with fewer than two neighbours have no pairs and score 0.
///
/// Returns [`GraphError::NodeOutOfRange`] if any edge names a node that does
/// not exist.
pub fn local_clustering(n: usize, edges: &[(usize, usize)]) -> Result<Array1<f64>, GraphError> {
    let adjacency = simple_adjacency(n, edges)?;
    Ok(adjacency
        .iter()
        .map(|neighbours| {
            let k = neighbours.len();
            if k < 2 {
                return 0.0;
            }
            let pairs = (k * (k - 1)) as f64;
            2.0 * triangles_among(neighbours, &adjacency) as f64 / pairs
        })
        .collect())
}

/// Compute the average clustering coefficient.
///
/// The mean of [`local_clustering`] over every node, including those with
/// fewer than two neighbours, which is what NetworkX averages.
///
/// Returns [`GraphError::NodeOutOfRange`] if any edge names a node that does
/// not exist.
pub fn average_clustering(n: usize, edges: &[(usize, usize)]) -> Result<f64, GraphError> {
    if n == 0 {
        return Ok(0.0);
    }
    let local = local_clustering(n, edges)?;
    Ok(local.sum() / n as f64)
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
        assert_eq!(
            triangles_per_node(n, &edges).unwrap().to_vec(),
            vec![1, 1, 1, 0]
        );
    }

    #[test]
    fn local_clustering_matches_hand_computation() {
        let (n, edges) = triangle_with_tail();
        let c = local_clustering(n, &edges).unwrap();
        // Node 0 has neighbours {1, 2, 3} and one edge among them: 2*1/(3*2).
        assert_relative_eq!(c[0], 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(c[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[2], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[3], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn average_clustering_is_the_mean_over_all_nodes() {
        let (n, edges) = triangle_with_tail();
        let local = local_clustering(n, &edges).unwrap();
        let mean = local.sum() / n as f64;
        assert_relative_eq!(
            average_clustering(n, &edges).unwrap(),
            mean,
            epsilon = 1e-12
        );
    }

    #[test]
    fn self_loops_do_not_affect_clustering() {
        let (n, edges) = triangle_with_tail();
        let with_loops: Vec<_> = edges.iter().copied().chain([(0, 0), (3, 3)]).collect();

        assert_eq!(
            local_clustering(n, &edges).unwrap().to_vec(),
            local_clustering(n, &with_loops).unwrap().to_vec()
        );
        assert_eq!(
            triangles_per_node(n, &edges).unwrap().to_vec(),
            triangles_per_node(n, &with_loops).unwrap().to_vec()
        );
    }

    #[test]
    fn complete_graph_is_fully_clustered() {
        let edges: Vec<_> = (0..5)
            .flat_map(|i| ((i + 1)..5).map(move |j| (i, j)))
            .collect();
        let c = local_clustering(5, &edges).unwrap();
        c.iter()
            .for_each(|&value| assert_relative_eq!(value, 1.0, epsilon = 1e-12));
        // Each node sits in C(4,2) = 6 triangles.
        assert_eq!(triangles_per_node(5, &edges).unwrap().to_vec(), vec![6; 5]);
    }

    #[test]
    fn empty_graph_has_zero_average_clustering() {
        assert_relative_eq!(average_clustering(0, &[]).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(average_clustering(5, &[]).unwrap(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn out_of_range_edges_are_rejected() {
        assert!(triangles_per_node(3, &[(0, 7)]).is_err());
        assert!(local_clustering(3, &[(0, 7)]).is_err());
        assert!(average_clustering(3, &[(0, 7)]).is_err());
    }
}
