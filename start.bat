@echo off
setlocal enabledelayedexpansion

title Customer Voice Authentication Server

echo ==============================================================================
echo        CUSTOMER VOICE AUTHENTICATION - EDUCATIONAL PROTOTYPE
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
set "UVICORN_EXE=%~dp0.venv\Scripts\uvicorn.exe"

echo [1/3] Virtual environment detected.
echo [2/3] Launching web browser at http://127.0.0.1:8000 ...

:: Launch browser in background after short delay
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000"

echo [3/3] Starting FastAPI server on http://127.0.0.1:8000 ...
echo       Press Ctrl+C to stop the server.
echo.

"%UVICORN_EXE%" src.server:app --host 127.0.0.1 --port 8000

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server terminated unexpectedly (Exit Code: %ERRORLEVEL%).
    pause
)
