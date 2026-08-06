//! NetSmith Python bindings
//!
//! This module provides PyO3 bindings for NetSmith network analysis functions.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;

use netsmith_core::{
    community::{louvain, modularity},
    degree::{degree_sequence, in_degree_sequence, out_degree_sequence, strength_sequence},
    metrics::{triangles_per_node, average_clustering, local_clustering},
    paths::{mean_shortest_path, shortest_paths_from_source, connected_components},
};

/// Convert edge array to edge list
fn edges_from_array(edges: PyReadonlyArray2<usize>) -> PyResult<Vec<(usize, usize)>> {
    let e = edges.as_array();
    if e.ncols() != 2 {
        return Err(PyValueError::new_err("edges shape must be [m, 2]"));
    }
    let mut edge_list = Vec::with_capacity(e.nrows());
    for r in 0..e.nrows() {
        edge_list.push((e[[r, 0]], e[[r, 1]]));
    }
    Ok(edge_list)
}

/// Compute degree sequence
#[pyfunction]
fn degree_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
    directed: bool,
) -> PyResult<Py<PyArray1<usize>>> {
    let edge_list = edges_from_array(edges)?;
    let degrees = degree_sequence(n, &edge_list, directed);
    Ok(degrees.into_pyarray(py).to_owned())
}

/// Compute in-degree sequence
#[pyfunction]
fn in_degree_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<Py<PyArray1<usize>>> {
    let edge_list = edges_from_array(edges)?;
    let degrees = in_degree_sequence(n, &edge_list);
    Ok(degrees.into_pyarray(py).to_owned())
}

/// Compute out-degree sequence
#[pyfunction]
fn out_degree_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<Py<PyArray1<usize>>> {
    let edge_list = edges_from_array(edges)?;
    let degrees = out_degree_sequence(n, &edge_list);
    Ok(degrees.into_pyarray(py).to_owned())
}

/// Compute strength sequence (weighted degree)
#[pyfunction]
fn strength_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
    weights: PyReadonlyArray1<f64>,
    directed: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let edge_list = edges_from_array(edges)?;
    let w = weights.as_array();
    if w.len() != edge_list.len() {
        return Err(PyValueError::new_err("weights length must match edges length"));
    }
    let weights_vec: Vec<f64> = w.iter().copied().collect();
    let strengths = strength_sequence(n, &edge_list, &weights_vec, directed);
    Ok(strengths.into_pyarray(py).to_owned())
}

/// Count triangles per node
#[pyfunction]
fn triangles_per_node_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<Py<PyArray1<usize>>> {
    let edge_list = edges_from_array(edges)?;
    let triangles = triangles_per_node(n, &edge_list);
    Ok(triangles.into_pyarray(py).to_owned())
}

/// Compute average clustering coefficient
#[pyfunction]
fn clustering_avg_rust(
    _py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<f64> {
    let edge_list = edges_from_array(edges)?;
    Ok(average_clustering(n, &edge_list))
}

/// Compute local clustering coefficients
#[pyfunction]
fn clustering_local_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let edge_list = edges_from_array(edges)?;
    let clustering = local_clustering(n, &edge_list);
    Ok(clustering.into_pyarray(py).to_owned())
}

/// Compute mean shortest path length
#[pyfunction]
fn mean_shortest_path_rust(
    _py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<f64> {
    let edge_list = edges_from_array(edges)?;
    Ok(mean_shortest_path(n, &edge_list))
}

/// Compute shortest paths from source
#[pyfunction]
fn shortest_paths_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
    source: usize,
    directed: bool,
) -> PyResult<Py<PyArray1<usize>>> {
    let edge_list = edges_from_array(edges)?;
    let dist = shortest_paths_from_source(n, &edge_list, source, directed);
    Ok(dist.into_pyarray(py).to_owned())
}

/// Compute connected components
#[pyfunction]
fn connected_components_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<(usize, Py<PyArray1<usize>>)> {
    let edge_list = edges_from_array(edges)?;
    let (n_components, labels) = connected_components(n, &edge_list);
    Ok((n_components, labels.into_pyarray(py).to_owned()))
}

/// Validate that every edge endpoint is a valid node id
fn check_edge_bounds(n: usize, edge_list: &[(usize, usize)]) -> PyResult<()> {
    for &(u, v) in edge_list.iter() {
        if u >= n || v >= n {
            return Err(PyValueError::new_err(format!(
                "edge ({u}, {v}) references a node id >= n_nodes ({n})"
            )));
        }
    }
    Ok(())
}

/// Extract and validate optional edge weights
fn weights_from_array(
    weights: Option<PyReadonlyArray1<f64>>,
    n_edges: usize,
) -> PyResult<Option<Vec<f64>>> {
    let Some(w) = weights else {
        return Ok(None);
    };
    let w = w.as_array();
    if w.len() != n_edges {
        return Err(PyValueError::new_err(format!(
            "weights length ({}) must match edges length ({})",
            w.len(),
            n_edges
        )));
    }
    for &value in w.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err("weights must be finite"));
        }
        if value < 0.0 {
            return Err(PyValueError::new_err(
                "weights must be non-negative (modularity is undefined for negative weights)",
            ));
        }
    }
    Ok(Some(w.iter().copied().collect()))
}

/// Detect communities with the Louvain method
///
/// Returns (labels, modularity, n_communities, n_levels).
#[pyfunction]
#[pyo3(signature = (n, edges, weights=None, resolution=1.0, seed=None, max_levels=20))]
fn louvain_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
    weights: Option<PyReadonlyArray1<f64>>,
    resolution: f64,
    seed: Option<u64>,
    max_levels: usize,
) -> PyResult<(Py<PyArray1<usize>>, f64, usize, usize)> {
    let edge_list = edges_from_array(edges)?;
    check_edge_bounds(n, &edge_list)?;
    let weights_vec = weights_from_array(weights, edge_list.len())?;
    if resolution <= 0.0 || !resolution.is_finite() {
        return Err(PyValueError::new_err("resolution must be a positive finite number"));
    }
    if max_levels == 0 {
        return Err(PyValueError::new_err("max_levels must be >= 1"));
    }

    let result = louvain(
        n,
        &edge_list,
        weights_vec.as_deref(),
        resolution,
        seed,
        max_levels,
    );
    Ok((
        result.labels.into_pyarray(py).to_owned(),
        result.modularity,
        result.n_communities,
        result.n_levels,
    ))
}

/// Compute the modularity of a partition
#[pyfunction]
#[pyo3(signature = (n, edges, labels, weights=None, resolution=1.0))]
fn modularity_rust(
    _py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
    labels: PyReadonlyArray1<usize>,
    weights: Option<PyReadonlyArray1<f64>>,
    resolution: f64,
) -> PyResult<f64> {
    let edge_list = edges_from_array(edges)?;
    check_edge_bounds(n, &edge_list)?;
    let weights_vec = weights_from_array(weights, edge_list.len())?;
    if resolution <= 0.0 || !resolution.is_finite() {
        return Err(PyValueError::new_err("resolution must be a positive finite number"));
    }

    let labels = labels.as_array();
    if labels.len() != n {
        return Err(PyValueError::new_err(format!(
            "labels length ({}) must equal n_nodes ({})",
            labels.len(),
            n
        )));
    }
    for &c in labels.iter() {
        if c >= n {
            return Err(PyValueError::new_err(format!(
                "community id {c} is out of range for {n} nodes"
            )));
        }
    }
    let labels_vec: Vec<usize> = labels.iter().copied().collect();

    Ok(modularity(
        n,
        &edge_list,
        weights_vec.as_deref(),
        &labels_vec,
        resolution,
    ))
}

/// Python module for netsmith_rs
#[pymodule]
fn netsmith_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Degree functions
    m.add_function(wrap_pyfunction!(degree_rust, m)?)?;
    m.add_function(wrap_pyfunction!(in_degree_rust, m)?)?;
    m.add_function(wrap_pyfunction!(out_degree_rust, m)?)?;
    m.add_function(wrap_pyfunction!(strength_rust, m)?)?;
    
    // Metrics functions
    m.add_function(wrap_pyfunction!(triangles_per_node_rust, m)?)?;
    m.add_function(wrap_pyfunction!(clustering_avg_rust, m)?)?;
    m.add_function(wrap_pyfunction!(clustering_local_rust, m)?)?;
    
    // Path functions
    m.add_function(wrap_pyfunction!(mean_shortest_path_rust, m)?)?;
    m.add_function(wrap_pyfunction!(shortest_paths_rust, m)?)?;
    m.add_function(wrap_pyfunction!(connected_components_rust, m)?)?;

    // Community detection
    m.add_function(wrap_pyfunction!(louvain_rust, m)?)?;
    m.add_function(wrap_pyfunction!(modularity_rust, m)?)?;

    Ok(())
}

