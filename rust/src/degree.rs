//! Degree computation functions.
//!
//! A self-loop counts once towards an undirected degree, matching the strength
//! and in/out-degree conventions here.

use ndarray::Array1;

use crate::{validate_edges, validate_weights, GraphError, WeightRule};

/// Compute the degree sequence.
///
/// Returns [`GraphError::NodeOutOfRange`] if any edge names a node that does
/// not exist.
pub fn degree_sequence(
    n: usize,
    edges: &[(usize, usize)],
    directed: bool,
) -> Result<Array1<usize>, GraphError> {
    validate_edges(n, edges)?;
    Ok(edges.iter().fold(Array1::zeros(n), |mut degrees, &(u, v)| {
        degrees[u] += 1;
        if !directed && u != v {
            degrees[v] += 1;
        }
        degrees
    }))
}

/// Compute the in-degree sequence of a directed graph.
///
/// Returns [`GraphError::NodeOutOfRange`] if any edge names a node that does
/// not exist.
pub fn in_degree_sequence(n: usize, edges: &[(usize, usize)]) -> Result<Array1<usize>, GraphError> {
    validate_edges(n, edges)?;
    Ok(edges.iter().fold(Array1::zeros(n), |mut degrees, &(_, v)| {
        degrees[v] += 1;
        degrees
    }))
}

/// Compute the out-degree sequence of a directed graph.
///
/// Returns [`GraphError::NodeOutOfRange`] if any edge names a node that does
/// not exist.
pub fn out_degree_sequence(
    n: usize,
    edges: &[(usize, usize)],
) -> Result<Array1<usize>, GraphError> {
    validate_edges(n, edges)?;
    Ok(edges.iter().fold(Array1::zeros(n), |mut degrees, &(u, _)| {
        degrees[u] += 1;
        degrees
    }))
}

/// Compute the strength sequence (degree weighted by edge weight).
///
/// Returns [`GraphError`] if any edge names a node that does not exist, or if
/// `weights` does not have exactly one finite, non-negative entry per edge —
/// a short weight array is an error, not an invitation to assume 1.0.
pub fn strength_sequence(
    n: usize,
    edges: &[(usize, usize)],
    weights: &[f64],
    directed: bool,
) -> Result<Array1<f64>, GraphError> {
    validate_edges(n, edges)?;
    validate_weights(edges.len(), Some(weights), WeightRule::NonNegative)?;
    Ok(edges
        .iter()
        .zip(weights)
        .fold(Array1::zeros(n), |mut strengths, (&(u, v), &w)| {
            strengths[u] += w;
            if !directed && u != v {
                strengths[v] += w;
            }
            strengths
        }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn undirected_degrees_count_both_endpoints() {
        let degrees = degree_sequence(4, &[(0, 1), (1, 2), (2, 3)], false).unwrap();
        assert_eq!(degrees.to_vec(), vec![1, 2, 2, 1]);
    }

    #[test]
    fn directed_degrees_count_only_the_source() {
        let degrees = degree_sequence(3, &[(0, 1), (0, 2)], true).unwrap();
        assert_eq!(degrees.to_vec(), vec![2, 0, 0]);
    }

    #[test]
    fn self_loops_count_once() {
        let degrees = degree_sequence(2, &[(0, 0), (0, 1)], false).unwrap();
        assert_eq!(degrees.to_vec(), vec![2, 1]);
    }

    #[test]
    fn in_and_out_degrees_split_the_edge() {
        let edges = [(0, 1), (2, 1)];
        assert_eq!(
            in_degree_sequence(3, &edges).unwrap().to_vec(),
            vec![0, 2, 0]
        );
        assert_eq!(
            out_degree_sequence(3, &edges).unwrap().to_vec(),
            vec![1, 0, 1]
        );
    }

    #[test]
    fn strength_sums_the_weights() {
        let strengths = strength_sequence(3, &[(0, 1), (1, 2)], &[2.0, 3.0], false).unwrap();
        assert_relative_eq!(strengths[0], 2.0);
        assert_relative_eq!(strengths[1], 5.0);
        assert_relative_eq!(strengths[2], 3.0);
    }

    #[test]
    fn short_weight_arrays_are_rejected_not_padded() {
        let error = strength_sequence(3, &[(0, 1), (1, 2)], &[2.0], false).unwrap_err();
        assert_eq!(
            error,
            GraphError::WeightsLengthMismatch {
                weights: 1,
                edges: 2
            }
        );
    }

    #[test]
    fn out_of_range_edges_are_rejected() {
        assert!(degree_sequence(2, &[(0, 5)], false).is_err());
        assert!(in_degree_sequence(2, &[(0, 5)]).is_err());
        assert!(out_degree_sequence(2, &[(5, 0)]).is_err());
        assert!(strength_sequence(2, &[(0, 5)], &[1.0], false).is_err());
    }
}
