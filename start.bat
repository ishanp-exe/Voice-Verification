@echo off
setlocal enabledelayedexpansion

title VoxKey Server

echo ==============================================================================
echo                 VOXKEY - EDUCATIONAL PROTOTYPE
echo ==============================================================================
echo.

:: 1. Check if virtual environment exists
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at: %~dp0.venv
    echo.
    echo Please create the virtual environment using Python 3.10:
    echo     py -3.10 -m venv .venv
    echo     .\.venv\Scripts\python -m pip install --upgrade pip
    echo     .\.venv\Scripts\pip install -e .
    echo.
    pause
    exit /b 1
)

:: 2. Set environment and paths
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "STREAMLIT_EXE=%~dp0.venv\Scripts\streamlit.exe"

echo [1/3] Virtual environment detected.
echo [2/3] Launching web browser at http://127.0.0.1:8501 ...

:: Launch browser in background after short delay
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8501"

echo [3/3] Starting Streamlit application on http://127.0.0.1:8501 ...
echo       Press Ctrl+C to stop the server.
echo.

"%STREAMLIT_EXE%" run app.py --server.address 127.0.0.1 --server.port 8501

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application terminated unexpectedly (Exit Code: %ERRORLEVEL%).
    pause
)
