"""
Load edge lists from various formats: pandas, polars, parquet, csv.
"""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ..engine.contracts import EdgeList

if TYPE_CHECKING:
    import pandas  # noqa: F401
    import polars  # noqa: F401


def _read_table(path: str, fmt: str):
    """
    Read a CSV or Parquet file with whichever DataFrame library is installed.

    Prefers polars, falls back to pandas. Both expose ``df[col].to_numpy()``,
    which is all the caller needs.

    Raises
    ------
    ImportError
        If neither polars nor pandas is installed
    """
    try:
        import polars as pl

        return pl.read_parquet(path) if fmt == "parquet" else pl.read_csv(path)
    except ImportError:
        pass

    try:
        import pandas as pd

        return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path)
    except ImportError:
        raise ImportError(
            f"reading {fmt} files requires polars or pandas. "
            f"Install with: pip install netsmith[polars] or pip install netsmith[pandas]"
        )


def load_edges(
    source: Union[str, np.ndarray, "pandas.DataFrame", "polars.DataFrame"],
    u_col: Optional[str] = None,
    v_col: Optional[str] = None,
    w_col: Optional[str] = None,
    directed: bool = False,
    n_nodes: Optional[int] = None,
) -> EdgeList:
    """
    Load edges from various sources.

    Parameters
    ----------
    source : str, array, DataFrame
        Source: file path (parquet, csv) or DataFrame
    u_col : str, optional
        Column name for source nodes (if DataFrame)
    v_col : str, optional
        Column name for destination nodes (if DataFrame)
    w_col : str, optional
        Column name for edge weights (if DataFrame)
    directed : bool, default False
        Whether graph is directed
    n_nodes : int, optional
        Number of nodes (if not provided, inferred from edges)

    Returns
    -------
    edges : EdgeList
        Edge list
    """
    # Handle numpy array
    if isinstance(source, np.ndarray):
        if source.shape[1] < 2:
            raise ValueError("Array must have at least 2 columns (u, v)")
        u = source[:, 0].astype(np.int64)
        v = source[:, 1].astype(np.int64)
        w = source[:, 2].astype(np.float64) if source.shape[1] > 2 else None
        return EdgeList(u=u, v=v, w=w, directed=directed, n_nodes=n_nodes)

    # Handle string (file path)
    if isinstance(source, str):
        if source.endswith(".parquet"):
            df = _read_table(source, "parquet")
        elif source.endswith(".csv"):
            df = _read_table(source, "csv")
        else:
            raise ValueError(f"Unsupported file format: {source}")

        # Extract columns
        if u_col is None or v_col is None:
            raise ValueError("u_col and v_col must be specified for file sources")

        u = df[u_col].to_numpy().astype(np.int64)
        v = df[v_col].to_numpy().astype(np.int64)
        w = df[w_col].to_numpy().astype(np.float64) if w_col else None

        return EdgeList(u=u, v=v, w=w, directed=directed, n_nodes=n_nodes)

    # Handle pandas DataFrame
    try:
        import pandas as pd

        if isinstance(source, pd.DataFrame):
            if u_col is None or v_col is None:
                raise ValueError("u_col and v_col must be specified for DataFrame")
            u = source[u_col].values.astype(np.int64)
            v = source[v_col].values.astype(np.int64)
            w = source[w_col].values.astype(np.float64) if w_col else None
            return EdgeList(u=u, v=v, w=w, directed=directed, n_nodes=n_nodes)
    except ImportError:
        pass

    # Handle polars DataFrame
    try:
        import polars as pl

        if isinstance(source, pl.DataFrame):
            if u_col is None or v_col is None:
                raise ValueError("u_col and v_col must be specified for DataFrame")
            u = source[u_col].to_numpy().astype(np.int64)
            v = source[v_col].to_numpy().astype(np.int64)
            w = source[w_col].to_numpy().astype(np.float64) if w_col else None
            return EdgeList(u=u, v=v, w=w, directed=directed, n_nodes=n_nodes)
    except ImportError:
        pass

    raise ValueError(f"Unsupported source type: {type(source)}")
