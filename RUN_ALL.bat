@echo off
cd /d "%~dp0"

if not exist ".env" (
    copy ".env.example" ".env" >nul
)

if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment was not found. Running setup first...
    call "%~dp0SETUP_WINDOWS.bat"
)

start "APC v4 Vector Classifier API" cmd /k "cd /d ""%~dp0"" && "".venv\Scripts\python.exe"" main.py"
timeout /t 5 /nobreak >nul
start "" "http://127.0.0.1:8000/docs"
