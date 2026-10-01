@echo off
setlocal enabledelayedexpansion

title VoxKey Setup

echo ==============================================================================
echo                 VOXKEY - ENVIRONMENT SETUP
echo ==============================================================================
echo.

cd /d "%~dp0"

:: 1. Check for Python 3.10
echo [1/4] Checking Python 3.10 availability...

set "PYTHON_CMD="
py -3.10 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=py -3.10"
    goto :python_found
)

python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
    if "!PY_VER:~0,4!"=="3.10" (
        set "PYTHON_CMD=python"
        goto :python_found
    )
)

echo [ERROR] Python 3.10 was not found on this system.
echo.
echo Explanation:
echo   PyTorch and SpeechBrain require Python 3.10 on Windows for reliable prebuilt binaries.
echo.
echo Recommended Action:
echo   1. Download Python 3.10 from https://www.python.org/downloads/
echo   2. Run the installer and check the box: "Add Python 3.10 to PATH"
echo   3. Re-run setup.bat
echo.
pause
exit /b 1

:python_found
echo       Found compatible Python: %PYTHON_CMD%

:: 2. Check Virtual Environment (.venv)
echo [2/4] Inspecting local virtual environment (.venv)...

if exist ".venv" (
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" --version >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            echo       Existing valid .venv detected. Preserving environment.
            goto :venv_ready
        ) else (
            echo [WARNING] An existing .venv folder was found, but its python.exe is non-functional.
            echo           To protect your files, setup will NOT silently delete it.
            echo           Please rename or remove the broken .venv folder manually, then re-run setup.bat.
            echo.
            pause
            exit /b 1
        )
    ) else (
        echo [WARNING] An existing .venv folder is missing Scripts\python.exe.
        echo           Preserving directory. Please inspect or delete .venv manually, then re-run setup.bat.
        echo.
        pause
        exit /b 1
    )
) else (
    echo       Creating clean Python 3.10 virtual environment in .venv...
    %PYTHON_CMD% -m venv .venv
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Virtual environment successfully created.
)

:venv_ready

:: 3. Upgrade pip
echo [3/4] Ensuring pip and core packaging tools are up to date...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Pip upgrade encountered a warning. Continuing with dependency installation...
)

:: 4. Install Dependencies
echo [4/4] Installing application dependencies from requirements.txt and pyproject.toml...
echo       This may take a few minutes if downloading PyTorch / SpeechBrain...
".venv\Scripts\pip.exe" install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install requirements. Please check your Internet connection.
    pause
    exit /b 1
)

".venv\Scripts\pip.exe" install -e .
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install package in editable mode.
    pause
    exit /b 1
)

echo.
echo ==============================================================================
echo [SUCCESS] VoxKey setup completed successfully!
echo ==============================================================================
echo.
echo You can now launch the app anytime by double-clicking:
echo     start.bat
echo.
echo Or by running in PowerShell:
echo     .\start.ps1
echo.
pause
