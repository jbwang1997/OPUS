"""Basic infrastructure validation without external dependencies."""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock


def test_basic_python_functionality():
    """Test basic Python functionality."""
    assert 2 + 2 == 4
    assert isinstance("hello", str)
    assert len([1, 2, 3]) == 3


def test_pathlib_functionality():
    """Test pathlib functionality."""
    current_dir = Path.cwd()
    assert current_dir.exists()
    
    test_path = Path("/tmp/test.txt")
    assert test_path.name == "test.txt"


def test_temporary_files():
    """Test temporary file creation."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("test content")
        tmp_path = tmp_file.name
    
    try:
        assert os.path.exists(tmp_path)
        with open(tmp_path, 'r') as f:
            content = f.read()
            assert content == "test content"
    finally:
        os.unlink(tmp_path)


def test_mock_functionality():
    """Test mock functionality."""
    mock_obj = Mock()
    mock_obj.method.return_value = "test_result"
    
    result = mock_obj.method()
    assert result == "test_result"
    mock_obj.method.assert_called_once()


def test_project_structure():
    """Test that project structure is correct."""
    project_root = Path.cwd()
    
    # Check main directories
    assert (project_root / "loaders").exists()
    assert (project_root / "models").exists()
    assert (project_root / "tests").exists()
    
    # Check test directories
    assert (project_root / "tests" / "unit").exists()
    assert (project_root / "tests" / "integration").exists()
    
    # Check configuration files
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / ".gitignore").exists()


def test_gitignore_entries():
    """Test that .gitignore contains required entries."""
    gitignore_path = Path.cwd() / ".gitignore"
    assert gitignore_path.exists()
    
    content = gitignore_path.read_text()
    required_entries = [
        ".pytest_cache/",
        ".coverage",
        "htmlcov/",
        "coverage.xml",
        ".claude/"
    ]
    
    for entry in required_entries:
        assert entry in content, f"Missing {entry} in .gitignore"


def test_pyproject_toml_structure():
    """Test that pyproject.toml has correct structure."""
    pyproject_path = Path.cwd() / "pyproject.toml"
    assert pyproject_path.exists()
    
    content = pyproject_path.read_text()
    
    # Check required sections
    required_sections = [
        "[tool.poetry]",
        "[tool.pytest.ini_options]", 
        "[tool.coverage.run]",
        "[tool.coverage.report]",
        "[tool.poetry.scripts]"
    ]
    
    for section in required_sections:
        assert section in content, f"Missing {section} in pyproject.toml"


def test_conftest_exists():
    """Test that conftest.py exists and is readable."""
    conftest_path = Path.cwd() / "tests" / "conftest.py"
    assert conftest_path.exists()
    
    content = conftest_path.read_text()
    assert "pytest" in content
    assert "fixture" in content


class TestInfrastructureValidation:
    """Test class for infrastructure validation."""
    
    def test_class_based_tests_work(self):
        """Test that class-based tests work."""
        assert True
    
    def test_assertions_work(self):
        """Test various assertion types."""
        assert 1 == 1
        assert "test" != "fail"
        assert len("hello") == 5
        assert 5 > 3
    
    def test_exception_handling(self):
        """Test exception handling."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            assert str(e) == "test error"
        else:
            assert False, "Exception should have been raised"


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    test_functions = [
        test_basic_python_functionality,
        test_pathlib_functionality,
        test_temporary_files,
        test_mock_functionality,
        test_project_structure,
        test_gitignore_entries,
        test_pyproject_toml_structure,
        test_conftest_exists,
    ]
    
    print("Running basic infrastructure tests...")
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
    
    # Run class-based tests
    test_class = TestInfrastructureValidation()
    class_methods = [
        test_class.test_class_based_tests_work,
        test_class.test_assertions_work,
        test_class.test_exception_handling,
    ]
    
    for test_method in class_methods:
        try:
            test_method()
            print(f"✓ {test_method.__name__}")
        except Exception as e:
            print(f"✗ {test_method.__name__}: {e}")
    
    print("Basic infrastructure tests completed!")