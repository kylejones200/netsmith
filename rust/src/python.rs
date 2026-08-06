//! PyO3 bindings, compiled only with the `python` feature.
//!
//! These are a thin translation layer: reshape NumPy arrays into slices, hand
//! them to the kernels, and turn any [`GraphError`] into a Python exception.
//! The kernels do the validating, so nothing is checked twice and nothing is
//! silently skipped.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::{
    centrality::betweenness,
    community::{louvain, modularity},
    degree::{degree_sequence, in_degree_sequence, out_degree_sequence, strength_sequence},
    metrics::{average_clustering, local_clustering, triangles_per_node},
    paths::{connected_components, mean_shortest_path, shortest_paths_from_source},
    GraphError,
};

impl From<GraphError> for PyErr {
    fn from(error: GraphError) -> PyErr {
        PyValueError::new_err(error.to_string())
    }
}

/// Reshape an [m, 2] array into an edge list.
fn edges_from_array(edges: PyReadonlyArray2<usize>) -> PyResult<Vec<(usize, usize)>> {
    let array = edges.as_array();
    if array.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "edges must have shape [m, 2], got [{}, {}]",
            array.nrows(),
            array.ncols()
        )));
    }
    Ok(array
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect())
}

/// Copy an optional weight array into a Vec.
fn weights_from_array(weights: Option<PyReadonlyArray1<f64>>) -> Option<Vec<f64>> {
    weights.map(|w| w.as_array().to_vec())
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
    let degrees = degree_sequence(n, &edge_list, directed)?;
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
    let degrees = in_degree_sequence(n, &edge_list)?;
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
    let degrees = out_degree_sequence(n, &edge_list)?;
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
    let weights = weights.as_array().to_vec();
    let strengths = strength_sequence(n, &edge_list, &weights, directed)?;
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
    let triangles = triangles_per_node(n, &edge_list)?;
    Ok(triangles.into_pyarray(py).to_owned())
}

/// Compute average clustering coefficient
#[pyfunction]
fn clustering_avg_rust(_py: Python<'_>, n: usize, edges: PyReadonlyArray2<usize>) -> PyResult<f64> {
    let edge_list = edges_from_array(edges)?;
    Ok(average_clustering(n, &edge_list)?)
}

/// Compute local clustering coefficients
#[pyfunction]
fn clustering_local_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let edge_list = edges_from_array(edges)?;
    let clustering = local_clustering(n, &edge_list)?;
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
    Ok(mean_shortest_path(n, &edge_list)?)
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
    let distances = shortest_paths_from_source(n, &edge_list, source, directed)?;
    Ok(distances.into_pyarray(py).to_owned())
}

/// Compute connected components
#[pyfunction]
fn connected_components_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
) -> PyResult<(usize, Py<PyArray1<usize>>)> {
    let edge_list = edges_from_array(edges)?;
    let (count, labels) = connected_components(n, &edge_list)?;
    Ok((count, labels.into_pyarray(py).to_owned()))
}

/// Compute betweenness centrality (Brandes)
#[pyfunction]
#[pyo3(signature = (n, edges, weights=None, directed=false, normalized=true))]
fn betweenness_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
    weights: Option<PyReadonlyArray1<f64>>,
    directed: bool,
    normalized: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let edge_list = edges_from_array(edges)?;
    let weights = weights_from_array(weights);

    let scores =
        py.allow_threads(|| betweenness(n, &edge_list, weights.as_deref(), directed, normalized))?;
    Ok(scores.into_pyarray(py).to_owned())
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
    let weights = weights_from_array(weights);

    let result = py.allow_threads(|| {
        louvain(
            n,
            &edge_list,
            weights.as_deref(),
            resolution,
            seed,
            max_levels,
        )
    })?;
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
    let weights = weights_from_array(weights);
    let labels = labels.as_array().to_vec();

    Ok(modularity(
        n,
        &edge_list,
        weights.as_deref(),
        &labels,
        resolution,
    )?)
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

    // Centrality
    m.add_function(wrap_pyfunction!(betweenness_rust, m)?)?;

    // Community detection
    m.add_function(wrap_pyfunction!(louvain_rust, m)?)?;
    m.add_function(wrap_pyfunction!(modularity_rust, m)?)?;

    Ok(())
}
