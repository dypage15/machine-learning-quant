@echo off
copy "%USERPROFILE%\.pca_mnq_ml\logs\nightly_20260629_081245.txt" "%~dp0last_run.txt" > nul
copy "%USERPROFILE%\.pca_mnq_ml\next_session_predictions.json" "%~dp0last_predictions.json" > nul 2>&1
echo Done.
