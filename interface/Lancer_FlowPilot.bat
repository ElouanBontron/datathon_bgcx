@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)
".venv\Scripts\python.exe" -c "import streamlit,pandas,plotly" >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installation impossible. Verifiez votre connexion internet.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.address localhost
if errorlevel 1 pause
