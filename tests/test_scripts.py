"""Tests for utility scripts.

Validates that scripts are importable and contain expected functions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_script_module(script_path: str):
    """Dynamically loads a script as a module."""
    spec = importlib.util.spec_from_file_location("script", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCheckSetup:
    """Tests for check_setup.py script."""

    def test_check_setup_imports_successfully(self):
        """Verify check_setup.py can be imported."""
        script_path = Path(__file__).parent.parent / "scripts" / "check_setup.py"
        module = load_script_module(str(script_path))
        assert module is not None

    def test_check_setup_has_main_guard(self):
        """Verify check_setup.py has entry point."""
        script_path = Path(__file__).parent.parent / "scripts" / "check_setup.py"
        with open(script_path) as f:
            content = f.read()
        assert "__name__" in content and "main()" in content


class TestCleanValidateDataset:
    """Tests for clean_validate_dataset.py script."""

    def test_clean_validate_imports_successfully(self):
        """Verify clean_validate_dataset.py can be imported."""
        script_path = Path(__file__).parent.parent / "scripts" / "clean_validate_dataset.py"
        module = load_script_module(str(script_path))
        assert module is not None

    def test_clean_validate_has_main_function(self):
        """Verify script defines a main function."""
        script_path = Path(__file__).parent.parent / "scripts" / "clean_validate_dataset.py"
        with open(script_path) as f:
            content = f.read()
        assert "def main" in content


class TestRunBenchmark:
    """Tests for run_benchmark.py script."""

    def test_run_benchmark_imports_successfully(self):
        """Verify run_benchmark.py can be imported."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_benchmark.py"
        module = load_script_module(str(script_path))
        assert module is not None

    def test_run_benchmark_has_main_entry_point(self):
        """Verify run_benchmark.py has entry point."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_benchmark.py"
        with open(script_path) as f:
            content = f.read()
        assert "__name__" in content and "main()" in content


class TestRunDatasetEda:
    """Tests for run_dataset_eda.py script."""

    def test_run_dataset_eda_imports_successfully(self):
        """Verify run_dataset_eda.py can be imported."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_dataset_eda.py"
        module = load_script_module(str(script_path))
        assert module is not None

    def test_run_dataset_eda_is_valid_python(self):
        """Verify run_dataset_eda.py is syntactically valid."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_dataset_eda.py"
        with open(script_path) as f:
            content = f.read()
        compile(content, str(script_path), 'exec')


class TestAllScriptsHaveDocstrings:
    """Verify all scripts have module-level docstrings."""

    @staticmethod
    def _get_script_paths():
        scripts_dir = Path(__file__).parent.parent / "scripts"
        return sorted(scripts_dir.glob("*.py"))

    def test_all_scripts_documented(self):
        """Verify all scripts have docstrings or shebangs."""
        for script_path in self._get_script_paths():
            with open(script_path) as f:
                content = f.read()
            # Should have docstring or shebang
            has_documentation = '"""' in content or "'''" in content or "#!" in content
            assert has_documentation, f"{script_path.name} should have documentation"
