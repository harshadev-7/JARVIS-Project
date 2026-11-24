@echo off
setlocal enabledelayedexpansion
title J.A.R.V.I.S. Production Launcher // Iron Man AI System
color 0B

echo ===============================================================================
echo            __       ___      .______     ____    ____  __       _______.
echo           ^|  ^|     /   \     ^|   _  \    \   \  /   / ^|  ^|     /       ^|
echo           ^|  ^|    /  ^|_  \    ^|  ^|_)  ^|    \   \/   /  ^|  ^|    ^|   (----`
echo     .--.  ^|  ^|   /  /_\  \   ^|      /      \      /   ^|  ^|     \   \    
echo     ^|  `--'  ^|  /  _____  \  ^|  ^|\  \----.   \    /    ^|  ^| .----)   ^|   
echo      \______/  /__/     \__\ ^| _^| `._____^|    \__/     ^|__^| ^|_______/    
echo ===============================================================================
echo [SYSTEM] J.A.R.V.I.S. Production Master Runtime ^| All 30 Phases Unified
echo ===============================================================================

:: 1. Locate Virtual Environment
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [WARNING] Local virtual environment not found in .venv. Trying system python...
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python was not found on your PATH. Please install Python 3.11+ or create .venv.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

:: 2. Verify Directory Structure
if not exist "logs" mkdir logs
if not exist "media\temporary" mkdir "media\temporary"
if not exist "checkpoints\local_models" mkdir "checkpoints\local_models"

:: 3. Present Launcher Menu
echo.
echo Please choose an operational mode:
echo   [1] Launch Interactive Terminal CLI (Full Agent Swarm, Reasoning, & Diagnostics)
echo   [2] Launch Holographic Web HUD Server (FastAPI + WebSockets at http://localhost:8000/hud)
echo   [3] Launch Desktop Arc Reactor HUD (Native Sci-Fi GUI Window)
echo   [4] Run Full System Diagnostics & Automated Self-Tests
echo   [5] Run Continuous Autonomous Sentinel Daemon
echo   [6] Exit
echo.
set /p MODE="Enter mode (1-6) [default: 1]: "
if "%MODE%"=="" set "MODE=1"

if "%MODE%"=="1" (
    echo [START] Booting JARVIS Master Runtime in Terminal CLI...
    "%PYTHON_EXE%" main.py
    pause
    exit /b 0
)

if "%MODE%"=="2" (
    echo [START] Booting FastAPI Server & Holographic Web HUD at http://localhost:8000/hud ...
    start "" "http://localhost:8000/hud"
    "%PYTHON_EXE%" main.py serve
    pause
    exit /b 0
)

if "%MODE%"=="3" (
    echo [START] Booting Desktop Arc Reactor HUD Window...
    "%PYTHON_EXE%" -c "from interface.hud_app import JarvisHUDApp; app = JarvisHUDApp(); app.root.mainloop()"
    pause
    exit /b 0
)

if "%MODE%"=="4" (
    echo [START] Running Comprehensive Multi-Subsystem Diagnostics...
    "%PYTHON_EXE%" -c "import asyncio; from core.integration.system_diagnostics import system_diagnostics; rep = asyncio.run(system_diagnostics.run_full_diagnostics()); print(rep.format_terminal_summary())"
    pause
    exit /b 0
)

if "%MODE%"=="5" (
    echo [START] Running Autonomous Sentinel Daemon in continuous monitoring mode...
    "%PYTHON_EXE%" -c "import asyncio; from core.integration.master_runtime import master_runtime; from core.sentinel import sentinel_daemon; async def run(): await master_runtime.boot(); print('Sentinel active. Press Ctrl+C to halt.'); await asyncio.Event().wait(); asyncio.run(run())"
    pause
    exit /b 0
)

if "%MODE%"=="6" (
    echo Goodbye, sir.
    exit /b 0
)

echo [ERROR] Invalid selection.
pause
exit /b 1
