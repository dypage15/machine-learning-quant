@echo off
REM PCA MNQ ML System — Quick Launcher
REM Double-click this file to install deps and open the dashboard.

cd /d "%~dp0"

echo.
echo ============================================================
echo  PCA MNQ  ML Intelligence System
echo ============================================================
echo.

REM Install / update dependencies
echo [1/2] Installing Python dependencies ...
pip install -r requirements.txt --quiet

echo.
echo [2/2] Launching Streamlit dashboard ...
echo       Open your browser to http://localhost:8501
echo.
echo       To run the nightly pipeline:
echo         python run_nightly.py --skip-fetch
echo.

streamlit run ml_system/dashboard.py

pause
