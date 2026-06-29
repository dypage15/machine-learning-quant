@echo off
chcp 65001 > nul
cd /d C:\Users\dypag\Claude\Projects\TRADING
echo Importing bars.json into database...
echo.
python run_nightly.py --import-json bars.json
echo.
echo === DONE (exit code: %errorlevel%) ===
pause
