//! NetSmith Python bindings
//!
//! This module provides PyO3 bindings for NetSmith network analysis functions.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;

use netsmith_core::{
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
    
    Ok(())
}

