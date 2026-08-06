//! PyO3 bindings, compiled only with the `python` feature.
//!
//! These are a thin translation layer: reshape NumPy arrays into slices, hand
//! them to the kernels, and turn any [`GraphError`] into a Python exception.
//! The kernels do the validating, so nothing is checked twice and nothing is
//! silently skipped.

use ndarray::Array3;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::{
    centrality::betweenness,
    community::{louvain, modularity},
    degree::{degree_sequence, in_degree_sequence, out_degree_sequence, strength_sequence},
    metrics::{average_clustering, local_clustering, triangles_per_node},
    nulls::degree_preserving_rewire_samples,
    paths::{
        connected_components, mean_shortest_path, shortest_paths_from_source,
        shortest_paths_from_sources,
    },
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

/// Compute shortest paths from several sources at once
///
/// Builds the adjacency list once and sweeps the sources in parallel, so this
/// costs far less than one `shortest_paths_rust` call per source. Row `i` holds
/// the distances from `sources[i]`.
#[pyfunction]
fn shortest_paths_multi_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
    sources: PyReadonlyArray1<usize>,
    directed: bool,
) -> PyResult<Py<PyArray2<usize>>> {
    let edge_list = edges_from_array(edges)?;
    let sources = sources.as_array().to_vec();

    let distances =
        py.allow_threads(|| shortest_paths_from_sources(n, &edge_list, &sources, directed))?;
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

/// Generate degree-preserving null models by double edge swap
///
/// Returns (edges, swaps, attempts): an [n_samples, n_edges, 2] array of
/// rewired edge lists, plus what each sample actually achieved. A sample whose
/// swap count falls short of `target_swaps` is too constrained to randomize —
/// the caller decides whether that is usable, but it is never hidden.
#[pyfunction]
#[pyo3(signature = (n, edges, n_samples, target_swaps, max_attempts, seed))]
fn rewire_degree_preserving_rust(
    py: Python<'_>,
    n: usize,
    edges: PyReadonlyArray2<usize>,
    n_samples: usize,
    target_swaps: usize,
    max_attempts: usize,
    seed: u64,
) -> PyResult<(Py<PyArray3<usize>>, Vec<usize>, Vec<usize>)> {
    let edge_list = edges_from_array(edges)?;
    let m = edge_list.len();

    let samples = py.allow_threads(|| {
        degree_preserving_rewire_samples(n, &edge_list, n_samples, target_swaps, max_attempts, seed)
    })?;

    let mut rewired = Array3::<usize>::zeros((n_samples, m, 2));
    samples.iter().enumerate().for_each(|(s, sample)| {
        sample.edges.iter().enumerate().for_each(|(e, &(u, v))| {
            rewired[[s, e, 0]] = u;
            rewired[[s, e, 1]] = v;
        });
    });

    let swaps = samples.iter().map(|s| s.swaps).collect();
    let attempts = samples.iter().map(|s| s.attempts).collect();
    Ok((rewired.into_pyarray(py).to_owned(), swaps, attempts))
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
    m.add_function(wrap_pyfunction!(shortest_paths_multi_rust, m)?)?;
    m.add_function(wrap_pyfunction!(connected_components_rust, m)?)?;

    // Centrality
    m.add_function(wrap_pyfunction!(betweenness_rust, m)?)?;

    // Null models
    m.add_function(wrap_pyfunction!(rewire_degree_preserving_rust, m)?)?;

    // Community detection
    m.add_function(wrap_pyfunction!(louvain_rust, m)?)?;
    m.add_function(wrap_pyfunction!(modularity_rust, m)?)?;

    Ok(())
}
