@echo off
cd /d C:\Users\dypag\Claude\Projects\TRADING
git log --oneline > git_output.txt 2>&1
git status >> git_output.txt 2>&1
git remote -v >> git_output.txt 2>&1
echo Done.
