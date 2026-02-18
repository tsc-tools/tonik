# Tonik - GitHub Copilot Instructions

## Project Overview

Tonik is a Python library that provides a solution to store and retrieve scientific time-series data, serving it through a FastAPI-based API. The API is optimized to visualize time-series and data labels with Grafana, featuring on-demand downsampling for large requests.

**Key Technologies:**
- Python 3.9+ (automated tests run on Python 3.9 and 3.11 via hatch)
- FastAPI for the REST API
- HDF5/NetCDF4/Zarr for data storage
- xarray, pandas, datashader for data processing
- pytest for testing
- hatch for environment management
- mkdocs for documentation

## Repository Structure

```
/home/runner/work/tonik/tonik/
├── src/tonik/           # Main source code
│   ├── api.py          # FastAPI application (TonikAPI class)
│   ├── storage.py      # Storage class for managing HDF5/NetCDF files
│   ├── utils.py        # Utility functions
│   ├── xarray2netcdf.py # NetCDF conversion utilities
│   ├── xarray2zarr.py   # Zarr conversion utilities
│   └── package_data/    # Static files (HTML, etc.)
├── tests/              # Test suite
│   ├── conftest.py     # pytest fixtures (setup, setup_api)
│   ├── test_api.py     # API endpoint tests
│   ├── test_save.py    # Data saving tests
│   ├── test_storage.py # Storage class tests
│   └── test_utils.py   # Utility function tests
├── docs/               # Documentation source
├── grafana_example/    # Example Grafana integration with Docker Compose
├── pyproject.toml      # Project configuration and dependencies
└── mkdocs.yml         # Documentation configuration
```

## Development Setup

### Installing the Package

**ALWAYS** install the package in editable mode before making changes or running tests:

```bash
pip install -e .
```

This installs tonik with all its dependencies. The installation takes 30-60 seconds.

### Installing Development Dependencies

For testing and development tools:

```bash
pip install pytest httpx hatch
```

Or use the dev extras:

```bash
pip install -e ".[dev]"
```

## Building and Testing

### Running Tests

**Primary test command** (recommended):
```bash
hatch run test:run-pytest
```

This runs tests across multiple Python versions (3.9, 3.11) as configured in pyproject.toml.

**Direct pytest command** (for quick iteration):
```bash
pytest tests/
```

**Run specific test file:**
```bash
pytest tests/test_utils.py -v
```

**Run with slow tests:**
```bash
pytest --runslow
```

**Important notes:**
- Some tests are marked as `slow` and are skipped by default
- Tests use fixtures defined in `conftest.py` that generate test data in temporary directories
- The `setup` fixture (package scope) creates a Storage instance with test data
- The `setup_api` fixture (module scope) creates a TestClient for API tests
- Tests expect approximately 0.01-1 second execution time for fast tests

### Common Test Patterns

Tests use:
- `TestClient` from `fastapi.testclient` for API testing
- `tmp_path_factory` from pytest for temporary directories
- Generated test data via `generate_test_data()` and `get_labels()` utilities

### Linting

No formal linter is configured in the project. However, VS Code devcontainer includes the Ruff extension for linting.

## Building Documentation

### Prerequisites

Install documentation dependencies:
```bash
pip install mkdocs "mkdocstrings[python]" mkdocs-jupyter
```

### Documentation Commands

**Serve locally** (for development):
```bash
mkdocs serve -a 0.0.0.0:8000
```

**Build documentation**:
```bash
mkdocs build
```

**Publish to GitHub Pages**:
```bash
mkdocs gh-deploy -r origin
```

Documentation will be available at: https://tsc-tools.github.io/tonik

## Release Process

See `HOW_TO_RELEASE.md` for the complete release process. Key steps:

1. Run tests with hatch: `hatch run test:run-pytest`
2. Update version in `pyproject.toml`
3. Build package: `python3 -m build`
4. Upload to PyPI: `python3 -m twine upload dist/*`

## API Endpoints

The TonikAPI class (in `src/tonik/api.py`) provides these endpoints:

- `GET /` - HTML landing page
- `GET /feature` - Retrieve time-series feature data
- `GET /inventory` - List available data
- `GET /labels` - Retrieve data labels

All API endpoints support CORS with `allow_origins=["*"]`.

## Key Architectural Patterns

1. **Storage Management**: The `Storage` class manages a collection of substores, each representing a location/channel combination (e.g., 'WIZ', '00', 'HHZ')

2. **Data Formats**: Supports 1D features (rsam, dsar, etc.) and 2D features (sonogram, ssam, filterbank) with frequency dimensions

3. **Time Handling**: Uses cftime for time conversions and datetime objects for API interfaces

4. **Downsampling**: Uses datashader for efficient downsampling of large time-series data

## Common Pitfalls and Workarounds

1. **Import paths**: Always import from `tonik` (not `src.tonik`) after installing with `pip install -e .`

2. **Test data**: Tests create temporary HDF5/NetCDF files. These are cleaned up automatically by pytest's `tmp_path_factory`

3. **Dependency conflicts**: The project uses `zarr>=3.0.3` for Python 3.11+ and `zarr<3` for Python 3.9 and 3.10 (see pyproject.toml dependencies)

4. **Python version**: Project requires Python 3.9+. Devcontainer uses Python 3.9.

## Environment Setup

### Devcontainer

The project includes a devcontainer configuration (`.devcontainer/devcontainer.json`):
- Base image: `mcr.microsoft.com/devcontainers/python:1-3.9-bullseye`
- Post-create command: `pip3 install -e . && pip3 install httpx pytest ipykernel hatch`
- Exposed port: 8000 (for API)
- VS Code extensions: Python, Jupyter, Ruff

### Running the API Server

```bash
tonik_api <rootdir>
```

Or programmatically:
```python
from tonik.api import TonikAPI
ta = TonikAPI('/path/to/data')
# ta.app is the FastAPI application
```

The API will be available at http://localhost:8000 by default.

## Making Changes

When implementing changes:

1. **Install in editable mode first**: `pip install -e .`
2. **Make focused, minimal changes** to the relevant files
3. **Add tests** in the appropriate test file following existing patterns
4. **Run tests** to validate: `pytest tests/`
5. **Test API changes** using the TestClient pattern in conftest.py
6. **Update documentation** if adding new features or changing APIs

## Final Notes

- The project currently has no CI/CD workflows in GitHub Actions
- No formal code coverage requirements are enforced
- The project is used for scientific data visualization with Grafana
- Main data formats are HDF5 and NetCDF4, with Zarr support for Python 3.11+
