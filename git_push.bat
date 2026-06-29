@echo off
chcp 65001 > nul
cd /d C:\Users\dypag\Claude\Projects\TRADING

echo === Git Setup and Push ===
echo.

:: Init repo if not already
if not exist ".git" (
    echo Initializing git repository...
    git init
    git branch -m main
) else (
    echo Git repo already initialized.
)

:: Configure user
git config user.email "thepage8171@gmail.com"
git config user.name "Dylan"

:: Add remote if not set
git remote get-url origin > nul 2>&1
if errorlevel 1 (
    echo Adding remote origin...
    git remote add origin https://github.com/dypage15/machine-learning-quant.git
) else (
    echo Remote already set.
    git remote set-url origin https://github.com/dypage15/machine-learning-quant.git
)

:: Create .gitignore
echo Creating .gitignore...
(
echo __pycache__/
echo *.pyc
echo *.pyo
echo .env
echo bars.json
echo last_run.txt
echo last_predictions.json
echo *.db
echo *.log
echo .DS_Store
) > .gitignore

:: Stage and commit
echo.
echo Staging all files...
git add -A
git status

echo.
echo Committing...
git commit -m "PCA MNQ ML system: pipeline, data import, Lorentzian+LSTM models"

echo.
echo Pushing to GitHub...
git push -u origin main

echo.
echo === DONE (exit code: %errorlevel%) ===
pause
