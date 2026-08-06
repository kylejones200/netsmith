# scripts/dev_build.sh
set -euo pipefail
pip install -U pip maturin
maturin develop --release -m rust/Cargo.toml
pip install -e .[speed,dtw,approx,rdata]
pytest -q
