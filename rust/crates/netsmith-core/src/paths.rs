//! Shortest path algorithms

use ndarray::Array1;
use std::collections::VecDeque;
use super::build_adjacency_list;

/// Compute mean shortest path length
pub fn mean_shortest_path(n: usize, edges: &[(usize, usize)]) -> f64 {
    let adj = build_adjacency_list(n, edges, true); // undirected
    let mut total = 0usize;
    let mut pairs = 0usize;
    
    for s in 0..n {
        let mut dist = vec![usize::MAX; n];
        let mut q = VecDeque::new();
        dist[s] = 0;
        q.push_back(s);
        
        while let Some(u) = q.pop_front() {
            for &v in adj[u].iter() {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    q.push_back(v);
                }
            }
        }
        
        for t in (s + 1)..n {
            if dist[t] != usize::MAX {
                total += dist[t];
                pairs += 1;
            }
        }
    }
    
    if pairs > 0 {
        (total as f64) / (pairs as f64)
    } else {
        f64::NAN
    }
}

/// Compute shortest paths from source to all nodes
pub fn shortest_paths_from_source(
    n: usize,
    edges: &[(usize, usize)],
    source: usize,
    directed: bool,
) -> Array1<usize> {
    let adj = build_adjacency_list(n, edges, !directed);
    let mut dist = Array1::from_elem(n, usize::MAX);
    let mut q = VecDeque::new();
    
    dist[source] = 0;
    q.push_back(source);
    
    while let Some(u) = q.pop_front() {
        for &v in adj[u].iter() {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                q.push_back(v);
            }
        }
    }
    
    dist
}

/// Compute connected components
pub fn connected_components(n: usize, edges: &[(usize, usize)]) -> (usize, Array1<usize>) {
    let adj = build_adjacency_list(n, edges, true); // undirected
    let mut labels = Array1::from_elem(n, usize::MAX);
    let mut component_id = 0usize;
    
    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        
        let mut q = VecDeque::new();
        q.push_back(start);
        labels[start] = component_id;
        
        while let Some(u) = q.pop_front() {
            for &v in adj[u].iter() {
                if labels[v] == usize::MAX {
                    labels[v] = component_id;
                    q.push_back(v);
                }
            }
        }
        
        component_id += 1;
    }
    
    (component_id, labels)
}

