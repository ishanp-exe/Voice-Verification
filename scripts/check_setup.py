#!/usr/bin/env python3
"""VoxKey Environment & Setup Verification Script.

Validates that:
1. Python version is supported (>= 3.10)
2. A virtual environment is active
3. Required dependency packages import correctly (without downloading models)
4. The local data/ storage directory can be created and written to

Exits with code 0 if all checks pass, 1 otherwise.
Standard library only.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def check_python_version() -> bool:
    """Check that Python version is 3.10 or higher."""
    req_major, req_minor = 3, 10
    current_major, current_minor = sys.version_info.major, sys.version_info.minor
    is_supported = (current_major, current_minor) >= (req_major, req_minor)
    ver_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if is_supported:
        print(f"  [PASS] Python version: {ver_str} (>= {req_major}.{req_minor} required)")
        return True
    else:
        print(f"  [FAIL] Python version: {ver_str} (requires Python {req_major}.{req_minor}+)")
        return False


def check_virtual_env() -> bool:
    """Check if execution is running inside a virtual environment."""
    is_venv = (
        sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        or bool(os.environ.get("VIRTUAL_ENV"))
        or hasattr(sys, "real_prefix")
    )
    if is_venv:
        venv_path = os.environ.get("VIRTUAL_ENV") or sys.prefix
        print(f"  [PASS] Virtual environment active: {venv_path}")
        return True
    else:
        print("  [FAIL] No active virtual environment detected.")
        print("         Activate your environment before running VoxKey:")
        print("           Windows (PowerShell):  .\\.venv\\Scripts\\Activate.ps1")
        print("           macOS / Linux:         source .venv/bin/activate")
        return False


def check_required_packages() -> bool:
    """Verify all project dependencies import cleanly without side-effects."""
    packages = [
        ("torch", "PyTorch"),
        ("torchaudio", "Torchaudio"),
        ("speechbrain", "SpeechBrain"),
        ("streamlit", "Streamlit"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("pytest", "Pytest"),
        ("httpx", "HTTPX"),
        ("multipart", "python-multipart"),
    ]

    all_passed = True
    print("  Checking required dependency packages:")
    for mod_name, label in packages:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"    [PASS] {label:<18} ({mod_name}) {ver}")
        except ImportError as e:
            print(f"    [FAIL] {label:<18} ({mod_name}) - Import failed: {e}")
            all_passed = False
        except Exception as e:
            print(f"    [FAIL] {label:<18} ({mod_name}) - Error on import: {e}")
            all_passed = False

    return all_passed


def check_data_directory_access() -> bool:
    """Check that the data/ directory can be created and written to."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_dir = repo_root / "data"

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / ".setup_write_test.tmp"
        test_payload = b"voxkey_setup_verification"
        test_file.write_bytes(test_payload)
        read_back = test_file.read_bytes()
        test_file.unlink()

        if read_back == test_payload:
            print(f"  [PASS] Data directory writable: {data_dir}")
            return True
        else:
            print(f"  [FAIL] Data directory read/write verification failed: {data_dir}")
            return False
    except Exception as e:
        print(f"  [FAIL] Cannot create or write to data directory ({data_dir}): {e}")
        return False


def main() -> int:
    print("=" * 70)
    print("            VOXKEY - ENVIRONMENT & SETUP VERIFICATION")
    print("=" * 70)
    print()

    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_env),
        ("Required Packages", check_required_packages),
        ("Storage / Data Access", check_data_directory_access),
    ]

    results = []
    for name, check_func in checks:
        print(f"Checking {name}...")
        passed = check_func()
        results.append((name, passed))
        print()

    all_passed = all(passed for _, passed in results)

    print("=" * 70)
    if all_passed:
        print("  RESULT: ALL CHECKS PASSED [OK]")
        print("  Your environment is properly configured to run VoxKey.")
        print("=" * 70)
        return 0
    else:
        print("  RESULT: SETUP CHECKS FAILED [FAIL]")
        print("  Please review the failures listed above and fix missing requirements.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
