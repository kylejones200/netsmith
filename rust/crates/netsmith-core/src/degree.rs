//! Degree computation functions

use ndarray::Array1;

/// Compute degree sequence from edge list
pub fn degree_sequence(n: usize, edges: &[(usize, usize)], directed: bool) -> Array1<usize> {
    let mut degrees = Array1::zeros(n);
    
    for &(u, v) in edges.iter() {
        if u < n {
            degrees[u] += 1;
        }
        if !directed && v < n && u != v {
            degrees[v] += 1;
        }
    }
    
    degrees
}

/// Compute in-degree sequence for directed graphs
pub fn in_degree_sequence(n: usize, edges: &[(usize, usize)]) -> Array1<usize> {
    let mut degrees = Array1::zeros(n);
    
    for &(_, v) in edges.iter() {
        if v < n {
            degrees[v] += 1;
        }
    }
    
    degrees
}

/// Compute out-degree sequence for directed graphs
pub fn out_degree_sequence(n: usize, edges: &[(usize, usize)]) -> Array1<usize> {
    let mut degrees = Array1::zeros(n);
    
    for &(u, _) in edges.iter() {
        if u < n {
            degrees[u] += 1;
        }
    }
    
    degrees
}

/// Compute strength (sum of edge weights) sequence
pub fn strength_sequence(
    n: usize,
    edges: &[(usize, usize)],
    weights: &[f64],
    directed: bool,
) -> Array1<f64> {
    let mut strengths = Array1::zeros(n);
    
    for (i, &(u, v)) in edges.iter().enumerate() {
        let w = weights.get(i).copied().unwrap_or(1.0);
        if u < n {
            strengths[u] += w;
        }
        if !directed && v < n && u != v {
            strengths[v] += w;
        }
    }
    
    strengths
}

