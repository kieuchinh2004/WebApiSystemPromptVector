@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment was not found. Running SETUP_WINDOWS.bat first...
    call "%~dp0SETUP_WINDOWS.bat"
)

".venv\Scripts\python.exe" main.py
pause
