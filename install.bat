@echo off
chcp 65001 > nul
echo [Notice] Starting dependency installation using Poetry...
poetry install
echo [Success] Installation complete.
pause
