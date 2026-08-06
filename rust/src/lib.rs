//! NetSmith - network analysis kernels.
//!
//! The algorithms are plain Rust with no Python coupling. The optional
//! `python` feature adds the PyO3 bindings that Maturin builds into the
//! `netsmith_rs` extension module; without it this is an ordinary Rust crate.
//!
//! # Failure policy
//!
//! Every kernel validates its input and returns [`GraphError`] on anything it
//! cannot compute. Nothing is skipped, clamped, or defaulted behind the
//! caller's back: an edge naming a node that does not exist is an error, not a
//! dropped edge, because silently ignoring it returns a plausible number for a
//! graph the caller never asked about.

#![warn(missing_docs)]

use std::error::Error;
use std::fmt;

pub mod centrality;
pub mod community;
pub mod degree;
pub mod kcore;
pub mod metrics;
pub mod nulls;
pub mod paths;

#[cfg(feature = "python")]
mod python;
mod rng;

// Re-export for convenience
pub use centrality::*;
pub use community::*;
pub use degree::*;
pub use kcore::*;
pub use metrics::*;
pub use nulls::*;
pub use paths::*;

/// Everything a kernel can reject.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// An edge names a node id that is not below `n_nodes`.
    NodeOutOfRange {
        /// Position of the offending edge in the edge list.
        index: usize,
        /// Source node id.
        u: usize,
        /// Target node id.
        v: usize,
        /// Number of nodes in the graph.
        n_nodes: usize,
    },
    /// The weight array does not line up with the edge list.
    WeightsLengthMismatch {
        /// Number of weights supplied.
        weights: usize,
        /// Number of edges supplied.
        edges: usize,
    },
    /// A weight is outside the range the kernel accepts.
    InvalidWeight {
        /// Position of the offending weight.
        index: usize,
        /// The offending value.
        weight: f64,
        /// What the kernel required of it.
        requirement: &'static str,
    },
    /// A per-node array does not have one entry per node.
    LabelsLengthMismatch {
        /// Number of labels supplied.
        labels: usize,
        /// Number of nodes in the graph.
        n_nodes: usize,
    },
    /// A parameter is outside its valid range.
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// What the kernel required of it.
        requirement: &'static str,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::NodeOutOfRange {
                index,
                u,
                v,
                n_nodes,
            } => write!(
                f,
                "edge {index} ({u}, {v}) names a node id outside [0, {n_nodes})"
            ),
            GraphError::WeightsLengthMismatch { weights, edges } => write!(
                f,
                "weights length ({weights}) must match edges length ({edges})"
            ),
            GraphError::InvalidWeight {
                index,
                weight,
                requirement,
            } => write!(f, "weight {index} is {weight}, but {requirement}"),
            GraphError::LabelsLengthMismatch { labels, n_nodes } => write!(
                f,
                "labels length ({labels}) must equal the node count ({n_nodes})"
            ),
            GraphError::InvalidParameter { name, requirement } => {
                write!(f, "{name} {requirement}")
            }
        }
    }
}

impl Error for GraphError {}

/// What a kernel demands of its edge weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightRule {
    /// Zero is allowed; negative is not (modularity, strength).
    NonNegative,
    /// Weights are distances, so zero would collapse nodes together.
    StrictlyPositive,
}

impl WeightRule {
    fn accepts(self, weight: f64) -> bool {
        match self {
            WeightRule::NonNegative => weight >= 0.0,
            WeightRule::StrictlyPositive => weight > 0.0,
        }
    }

    fn requirement(self) -> &'static str {
        match self {
            WeightRule::NonNegative => "weights must be finite and non-negative",
            WeightRule::StrictlyPositive => {
                "weights are distances and must be finite and strictly positive"
            }
        }
    }
}

/// Check that every edge names a node below `n`.
pub fn validate_edges(n: usize, edges: &[(usize, usize)]) -> Result<(), GraphError> {
    edges
        .iter()
        .enumerate()
        .find(|(_, &(u, v))| u >= n || v >= n)
        .map_or(Ok(()), |(index, &(u, v))| {
            Err(GraphError::NodeOutOfRange {
                index,
                u,
                v,
                n_nodes: n,
            })
        })
}

/// Check that weights line up with the edges and satisfy `rule`.
pub fn validate_weights(
    n_edges: usize,
    weights: Option<&[f64]>,
    rule: WeightRule,
) -> Result<(), GraphError> {
    let Some(weights) = weights else {
        return Ok(());
    };
    if weights.len() != n_edges {
        return Err(GraphError::WeightsLengthMismatch {
            weights: weights.len(),
            edges: n_edges,
        });
    }
    weights
        .iter()
        .enumerate()
        .find(|(_, &w)| !w.is_finite() || !rule.accepts(w))
        .map_or(Ok(()), |(index, &weight)| {
            Err(GraphError::InvalidWeight {
                index,
                weight,
                requirement: rule.requirement(),
            })
        })
}

/// Build a sorted, deduplicated adjacency list from an edge list.
///
/// Returns [`GraphError::NodeOutOfRange`] if any edge names a node that does
/// not exist.
pub fn build_adjacency_list(
    n: usize,
    edges: &[(usize, usize)],
    undirected: bool,
) -> Result<Vec<Vec<usize>>, GraphError> {
    validate_edges(n, edges)?;

    let mut adjacency = vec![Vec::<usize>::new(); n];
    edges.iter().for_each(|&(u, v)| {
        adjacency[u].push(v);
        if undirected {
            adjacency[v].push(u);
        }
    });
    adjacency.iter_mut().for_each(|neighbours| {
        neighbours.sort_unstable();
        neighbours.dedup();
    });

    Ok(adjacency)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_range_edges_are_rejected_not_dropped() {
        let error = build_adjacency_list(3, &[(0, 1), (1, 9)], true).unwrap_err();
        assert_eq!(
            error,
            GraphError::NodeOutOfRange {
                index: 1,
                u: 1,
                v: 9,
                n_nodes: 3,
            }
        );
        assert!(error.to_string().contains("outside [0, 3)"));
    }

    #[test]
    fn adjacency_is_sorted_deduplicated_and_symmetric() {
        let adjacency = build_adjacency_list(3, &[(0, 2), (0, 1), (0, 1)], true).unwrap();
        assert_eq!(adjacency[0], vec![1, 2]);
        assert_eq!(adjacency[1], vec![0]);
        assert_eq!(adjacency[2], vec![0]);
    }

    #[test]
    fn directed_adjacency_keeps_direction() {
        let adjacency = build_adjacency_list(2, &[(0, 1)], false).unwrap();
        assert_eq!(adjacency[0], vec![1]);
        assert!(adjacency[1].is_empty());
    }

    #[test]
    fn weight_rules_reject_what_they_say_they_reject() {
        assert!(validate_weights(2, Some(&[1.0, 0.0]), WeightRule::NonNegative).is_ok());
        assert!(validate_weights(2, Some(&[1.0, 0.0]), WeightRule::StrictlyPositive).is_err());
        assert!(validate_weights(2, Some(&[1.0, -1.0]), WeightRule::NonNegative).is_err());
        assert!(validate_weights(2, Some(&[1.0, f64::NAN]), WeightRule::NonNegative).is_err());
        assert_eq!(
            validate_weights(3, Some(&[1.0]), WeightRule::NonNegative).unwrap_err(),
            GraphError::WeightsLengthMismatch {
                weights: 1,
                edges: 3
            }
        );
        assert!(validate_weights(3, None, WeightRule::StrictlyPositive).is_ok());
    }
}
