"""
CLI implementation: netsmith as a command-line tool.
"""

import click
from ...api.compute import degree, pagerank, communities
from ...api.load import load_edges


@click.group()
def main():
    """NetSmith: Fast Network Analysis"""
    pass


@main.command()
@click.option("--input", required=True, help="Input edge list file")
@click.option("--out", required=True, help="Output file")
@click.option("--u-col", default="u", help="Source node column")
@click.option("--v-col", default="v", help="Destination node column")
@click.option("--w-col", default=None, help="Edge weight column")
@click.option("--directed/--undirected", default=False, help="Directed graph")
@click.option("--backend", default="auto", type=click.Choice(["auto", "python", "rust"]))
def compute_degree(input, out, u_col, v_col, w_col, directed, backend):
    """Compute degree sequence."""
    edges = load_edges(input, u_col=u_col, v_col=v_col, w_col=w_col, directed=directed)
    degrees = degree(edges, backend=backend)
    
    import pandas as pd
    df = pd.DataFrame({"node": range(len(degrees)), "degree": degrees})
    df.to_parquet(out, index=False)
    click.echo(f"Computed degrees for {len(degrees)} nodes")


@main.command()
@click.option("--input", required=True, help="Input edge list file")
@click.option("--out", required=True, help="Output file")
@click.option("--u-col", default="u", help="Source node column")
@click.option("--v-col", default="v", help="Destination node column")
@click.option("--alpha", default=0.85, help="Damping factor")
@click.option("--backend", default="auto", type=click.Choice(["auto", "python", "rust"]))
def compute_pagerank(input, out, u_col, v_col, alpha, backend):
    """Compute PageRank."""
    edges = load_edges(input, u_col=u_col, v_col=v_col)
    pr = pagerank(edges, alpha=alpha, backend=backend)
    
    import pandas as pd
    df = pd.DataFrame({"node": range(len(pr)), "pagerank": pr})
    df.to_parquet(out, index=False)
    click.echo(f"Computed PageRank for {len(pr)} nodes")


@main.command()
@click.option("--input", required=True, help="Input edge list file")
@click.option("--out", required=True, help="Output file")
@click.option("--method", default="louvain", help="Community detection method")
@click.option("--backend", default="auto", type=click.Choice(["auto", "python", "rust"]))
def compute_communities(input, out, method, backend):
    """Compute community assignments."""
    edges = load_edges(input)
    comms = communities(edges, method=method, backend=backend)
    
    import pandas as pd
    df = pd.DataFrame({"node": range(len(comms)), "community": comms})
    df.to_parquet(out, index=False)
    click.echo(f"Computed communities for {len(comms)} nodes")


if __name__ == "__main__":
    main()

