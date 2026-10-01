@echo off
REM EYE FOR AI - one-click launcher for Windows.
REM First run: creates .venv and installs libraries (5-10 minutes). Later runs start the app directly.
cd /d "%~dp0"
title EYE FOR AI

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo and tick "Add Python to PATH" during installation.
    goto :error
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :error
)

if not exist ".venv\installed.flag" (
    echo [2/3] Installing libraries - first time only, takes 5-10 minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
    echo ok> ".venv\installed.flag"
)

REM Skip Streamlit's first-run e-mail prompt, which would otherwise wait for input.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo [3/3] Starting EYE FOR AI - the browser opens at http://localhost:8501
echo Close this window or press Ctrl+C to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
exit /b 0

:error
echo.
echo Something went wrong. Take a screenshot of this window and share it.
pause
exit /b 1
