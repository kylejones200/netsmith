# Contributing to NetSmith

Thank you for your interest in contributing to NetSmith! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kylejones200/netsmith.git
   cd netsmith
   ```

2. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install Rust (for building extensions):**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

4. **Build the Rust extension:**
   ```bash
   maturin develop --release
   ```

### Running Tests

```bash
# Run fast tests (default)
pytest

# Run all tests including slow/benchmark
pytest -m "not (slow or benchmark)"  # Fast tests only
pytest -m slow                        # Slow tests
pytest -m benchmark                   # Benchmark tests
pytest -m hard_validation             # Hard validation tests

# With coverage
pytest --cov=netsmith --cov-report=html
```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting: `black src/netsmith/ tests/`
- Use `isort` for import sorting: `isort src/netsmith/ tests/`
- Use `flake8` for linting: `flake8 src/netsmith/ tests/`
- Type hints are encouraged but not required

### Testing

- **Write tests for new features** - Aim for high coverage
- **Test invariants, not implementation details** - Tests should pass even if internals change
- **Use fixed random seeds** - All tests should be deterministic
- **Mark slow tests** - Use `@pytest.mark.slow` for tests that take >1 second
- **Mark benchmark tests** - Use `@pytest.mark.benchmark` for performance tests

### Commit Messages

Follow conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Example:
```
feat: add support for weighted graphs

- Add weighted parameter to graph construction
- Implement weight-based metrics
- Add tests for weighted graph operations
```

### Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure tests pass

3. **Update documentation** if needed (README, docstrings, etc.)

4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Test results
   - Any breaking changes

### Code Review

- All PRs require review before merging
- Address review comments promptly
- Keep PRs focused and reasonably sized
- Update CHANGELOG.md for user-facing changes

## Project Structure

```
netsmith/
├── src/netsmith/        # Main package
│   ├── core/            # Core graph types and algorithms
│   ├── engine/          # Backend implementations (Python/Rust)
│   ├── api/             # Public API surface
│   └── apps/            # Applications (CLI, reports, datasets)
├── rust/crates/         # Rust extensions (performance-critical code)
│   ├── netsmith-core/   # Core Rust algorithms
│   └── netsmith-py/     # Python bindings
├── tests/               # Test suite
├── examples/            # Example scripts and notebooks
└── docs/                # Sphinx documentation
```

## Areas for Contribution

### High Priority

- **Performance improvements** - Especially for large graphs
- **Additional algorithms** - New graph algorithms and metrics
- **Documentation** - Examples, tutorials, API docs
- **Bug fixes** - See GitHub issues

### Medium Priority

- **More examples** - Real-world use cases
- **Visualization improvements** - Better plotting functions
- **Error handling** - More helpful error messages
- **Type hints** - Improve type coverage

### Low Priority

- **Code cleanup** - Refactoring, consolidation
- **Test coverage** - Increase coverage in under-tested areas
- **CI/CD improvements** - Build and test automation

## Questions?

- Open an issue for bug reports or feature requests
- Check existing issues and discussions
- Review the documentation

Thank you for contributing to NetSmith!


