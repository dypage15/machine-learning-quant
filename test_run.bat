@echo off
chcp 65001 > nul
cd /d C:\Users\dypag\Claude\Projects\TRADING
echo Running run_nightly.py --skip-fetch ...
echo.
python run_nightly.py --skip-fetch
echo.
echo === DONE (exit code: %errorlevel%) ===
pause
