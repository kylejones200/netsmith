//! Network metrics computation

use ndarray::Array1;
use super::build_adjacency_list;

/// Count triangles per node
pub fn triangles_per_node(n: usize, edges: &[(usize, usize)]) -> Array1<usize> {
    let adj = build_adjacency_list(n, edges, true); // undirected
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
    
    tri
}

/// Compute average clustering coefficient
pub fn average_clustering(n: usize, edges: &[(usize, usize)]) -> f64 {
    let adj = build_adjacency_list(n, edges, true); // undirected
    let mut s = 0.0;
    let mut cnt = 0usize;
    
    for u in 0..n {
        let k = adj[u].len();
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
        cnt += 1;
    }
    
    if cnt > 0 {
        s / (cnt as f64)
    } else {
        0.0
    }
}

/// Compute local clustering coefficients
pub fn local_clustering(n: usize, edges: &[(usize, usize)]) -> Array1<f64> {
    let adj = build_adjacency_list(n, edges, true); // undirected
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
