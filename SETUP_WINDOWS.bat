@echo off
setlocal
cd /d "%~dp0"

echo [1/3] Checking Python...
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found. Install Python 3.10+ and run this file again.
    pause
    exit /b 1
)

echo [2/3] Creating Python virtual environment...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Cannot create .venv.
        pause
        exit /b 1
    )
)

echo [3/3] Installing Python packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example. Edit .env if llama-server.exe or MODEL_PATH is different.
)

echo.
echo Setup completed.
pause
