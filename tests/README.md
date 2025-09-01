# OPUS Testing Infrastructure

This directory contains the testing infrastructure for the OPUS project.

## Structure

```
tests/
├── README.md                    # This file
├── __init__.py                  # Test package initialization
├── conftest.py                  # Shared pytest fixtures
├── test_basic_infrastructure.py # Basic tests (no external deps)
├── test_infrastructure.py       # Full infrastructure tests (requires deps)
├── unit/                        # Unit tests
│   └── __init__.py
└── integration/                 # Integration tests
    └── __init__.py
```

## Running Tests

### Option 1: With Poetry (Recommended for full setup)
```bash
# Install dependencies
poetry install

# Run all tests
poetry run test
# or
poetry run tests

# Run with coverage
poetry run pytest --cov

# Run specific test categories
poetry run pytest -m unit      # Unit tests only
poetry run pytest -m integration  # Integration tests only
poetry run pytest -m "not slow"   # Skip slow tests
```

### Option 2: With pip (if Poetry unavailable)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install testing dependencies
pip install pytest pytest-cov pytest-mock torch numpy

# Run tests
pytest
```

### Option 3: Basic validation (no dependencies)
```bash
# Run basic infrastructure tests
python3 tests/test_basic_infrastructure.py
```

## Test Categories

Tests are organized with markers:
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.gpu` - Tests requiring GPU

## Coverage

Coverage is configured to:
- Measure coverage of `loaders/` and `models/` packages
- Generate HTML reports in `htmlcov/`
- Generate XML reports as `coverage.xml`
- Require minimum 80% coverage
- Exclude test files and CUDA extensions

## Fixtures

Common fixtures are available in `conftest.py`:
- `temp_dir` - Temporary directory for test files
- `mock_config` - Mock configuration dictionary
- `sample_tensor` - Sample PyTorch tensor
- `sample_batch_dict` - Sample batch data
- `mock_model` - Mock OPUS model
- And many more...

## Configuration

Test configuration is in `pyproject.toml` under:
- `[tool.pytest.ini_options]` - pytest settings
- `[tool.coverage.run]` - Coverage measurement
- `[tool.coverage.report]` - Coverage reporting

## Adding New Tests

1. Create test files with `test_*.py` naming
2. Use appropriate markers for categorization
3. Place unit tests in `tests/unit/`
4. Place integration tests in `tests/integration/`
5. Use fixtures from `conftest.py` for common test data