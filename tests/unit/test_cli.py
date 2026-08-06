"""Unit tests for the command-line interface.

Covers the three commands the CLI actually exposes: ``compute-degree``,
``compute-pagerank`` and ``compute-communities``. Each is exercised end to end
(read a CSV edge list, compute, write a Parquet result) as well as through its
argument handling.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from click.testing import CliRunner

from netsmith.apps.cli import main as cli

# The CLI writes its results with pandas.to_parquet, so the round-trip tests
# need pandas and a Parquet engine. Both ship in the dev dependencies.
pd = pytest.importorskip("pandas", reason="CLI output requires pandas")
pytest.importorskip("pyarrow", reason="Parquet output requires pyarrow")

COMMANDS = ["compute-degree", "compute-pagerank", "compute-communities"]

# Two triangles joined by a bridge: nodes 0-2 and 3-5.
EDGE_CSV = "u,v,w\n0,1,1.0\n1,2,1.0\n0,2,1.0\n3,4,1.0\n4,5,1.0\n3,5,1.0\n2,3,1.0\n"
N_NODES = 6


class TestCLI:
    """Test cases for the command-line interface."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.edge_file = os.path.join(self.temp_dir.name, "edges.csv")
        with open(self.edge_file, "w") as f:
            f.write(EDGE_CSV)

    def teardown_method(self):
        """Clean up after each test method."""
        self.temp_dir.cleanup()

    def out_path(self, name):
        """Path for a command's output file inside the temp directory."""
        return os.path.join(self.temp_dir.name, name)

    def run(self, command, *args):
        """Invoke a CLI command against the fixture edge list."""
        out = self.out_path(f"{command}.parquet")
        result = self.runner.invoke(cli, [command, "--input", self.edge_file, "--out", out, *args])
        return result, out

    def test_cli_help(self):
        """The top-level CLI shows help listing every command."""
        result = self.runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "NetSmith" in result.output
        for command in COMMANDS:
            assert command in result.output

    @pytest.mark.parametrize("command", COMMANDS)
    def test_command_help(self, command):
        """Each command documents its own options."""
        result = self.runner.invoke(cli, [command, "--help"])

        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--out" in result.output

    @pytest.mark.parametrize("command", COMMANDS)
    def test_command_writes_one_row_per_node(self, command):
        """Every command reads the CSV and writes a row per node."""
        result, out = self.run(command)

        assert result.exit_code == 0, result.output
        assert os.path.exists(out)
        df = pd.read_parquet(out)
        assert len(df) == N_NODES
        assert list(df["node"]) == list(range(N_NODES))

    def test_compute_degree_values(self):
        """Degrees match the fixture graph."""
        result, out = self.run("compute-degree")

        assert result.exit_code == 0, result.output
        df = pd.read_parquet(out)
        # Nodes 2 and 3 carry the bridge, so they have degree 3.
        assert list(df["degree"]) == [2, 2, 3, 3, 2, 2]

    def test_compute_pagerank_values(self):
        """PageRank scores are positive and finite for every node."""
        result, out = self.run("compute-pagerank")

        assert result.exit_code == 0, result.output
        df = pd.read_parquet(out)
        assert (df["pagerank"] > 0).all()
        assert np.isfinite(df["pagerank"]).all()

    def test_compute_pagerank_is_normalized(self):
        """PageRank should be a distribution over the nodes, summing to 1."""
        result, out = self.run("compute-pagerank")

        assert result.exit_code == 0, result.output
        assert pd.read_parquet(out)["pagerank"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_compute_communities_finds_the_two_triangles(self):
        """Louvain separates the two triangles."""
        result, out = self.run("compute-communities")

        assert result.exit_code == 0, result.output
        df = pd.read_parquet(out)
        labels = list(df["community"])
        assert len(set(labels)) == 2
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    @pytest.mark.parametrize("backend", ["python", "auto"])
    def test_backend_option_is_accepted(self, backend):
        """Both the Python backend and auto-detection run to completion."""
        result, out = self.run("compute-communities", "--backend", backend)

        assert result.exit_code == 0, result.output
        assert len(pd.read_parquet(out)) == N_NODES

    def test_custom_column_names(self):
        """Non-default column names are honoured."""
        renamed = os.path.join(self.temp_dir.name, "renamed.csv")
        with open(renamed, "w") as f:
            f.write("src,dst\n0,1\n1,2\n0,2\n")
        out = self.out_path("renamed.parquet")

        result = self.runner.invoke(
            cli,
            [
                "compute-degree",
                "--input",
                renamed,
                "--out",
                out,
                "--u-col",
                "src",
                "--v-col",
                "dst",
            ],
        )

        assert result.exit_code == 0, result.output
        assert list(pd.read_parquet(out)["degree"]) == [2, 2, 2]

    def test_reads_csv_without_polars(self, monkeypatch):
        """The loader falls back to pandas when polars is not installed.

        Regression test: the fallback branch used to reference an unbound `pl`,
        so every CLI command died with UnboundLocalError on a polars-free
        install.
        """
        monkeypatch.setitem(sys.modules, "polars", None)  # makes `import polars` raise

        result, out = self.run("compute-degree")

        assert result.exit_code == 0, result.output
        assert list(pd.read_parquet(out)["degree"]) == [2, 2, 3, 3, 2, 2]

    @pytest.mark.parametrize("command", COMMANDS)
    def test_missing_required_option_fails(self, command):
        """Omitting --input is a usage error, not a crash."""
        result = self.runner.invoke(cli, [command, "--out", self.out_path("x.parquet")])

        assert result.exit_code != 0
        assert "--input" in result.output

    def test_invalid_backend_is_rejected(self):
        """An unknown backend fails on the click.Choice, before any work."""
        result, _ = self.run("compute-degree", "--backend", "julia")

        assert result.exit_code != 0
        assert "julia" in result.output

    def test_unknown_command_fails(self):
        """An unrecognized subcommand exits non-zero with a clear message."""
        result = self.runner.invoke(cli, ["compute-nonsense"])

        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_unsupported_input_format_fails(self):
        """A file extension the loader cannot read reports the format."""
        bad = os.path.join(self.temp_dir.name, "edges.txt")
        with open(bad, "w") as f:
            f.write("0 1\n1 2\n")

        result = self.runner.invoke(
            cli,
            ["compute-degree", "--input", bad, "--out", self.out_path("x.parquet")],
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert "Unsupported file format" in str(result.exception)

    def test_missing_input_file_fails(self):
        """A nonexistent input path fails rather than writing an empty result."""
        out = self.out_path("x.parquet")
        result = self.runner.invoke(
            cli,
            [
                "compute-degree",
                "--input",
                os.path.join(self.temp_dir.name, "nope.csv"),
                "--out",
                out,
            ],
        )

        assert result.exit_code != 0
        assert not os.path.exists(out)
