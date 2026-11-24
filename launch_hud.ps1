# ===============================================================================
# JARVIS Holographic Command Center & Server Launcher (PowerShell)
# ===============================================================================

$ErrorActionPreference = "Stop"

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "           J.A.R.V.I.S. // Holographic Command Center Launcher" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor DarkCyan

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[WARNING] Local virtual environment not found. Falling back to system python." -ForegroundColor Yellow
    $PythonCmd = "python"
} else {
    $PythonCmd = $VenvPython
}

Write-Host "[INIT] Launching FastAPI Web Server & WebSocket Streaming Gateway..." -ForegroundColor Green
Start-Process "http://localhost:8000/hud"

& $PythonCmd main.py serve
