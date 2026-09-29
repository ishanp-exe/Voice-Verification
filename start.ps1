<#
.SYNOPSIS
    Starts the Customer Voice Authentication FastAPI web application on Windows.
#>

$ErrorActionPreference = "Stop"

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "       CUSTOMER VOICE AUTHENTICATION - EDUCATIONAL PROTOTYPE" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$VenvUvicorn = Join-Path $ScriptDir ".venv\Scripts\uvicorn.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found at: $ScriptDir\.venv" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please set up the environment with Python 3.10:"
    Write-Host "    py -3.10 -m venv .venv"
    Write-Host "    .\.venv\Scripts\python -m pip install --upgrade pip"
    Write-Host "    .\.venv\Scripts\pip install -e ."
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[1/3] Virtual environment detected." -ForegroundColor Green
Write-Host "[2/3] Scheduling browser launch at http://127.0.0.1:8000 ..." -ForegroundColor Green

# Launch browser in separate background process after a brief 2s delay
Start-Process powershell -ArgumentList "-NoProfile", "-Command", "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8000'" -WindowStyle Hidden

Write-Host "[3/3] Launching FastAPI server on http://127.0.0.1:8000 ..." -ForegroundColor Green
Write-Host "      Press Ctrl+C to stop the application." -ForegroundColor Yellow
Write-Host ""

try {
    & $VenvUvicorn src.server:app --host 127.0.0.1 --port 8000
} catch {
    Write-Host ""
    Write-Host "[ERROR] Server terminated: $_" -ForegroundColor Red
    Read-Host "Press Enter to close"
}
