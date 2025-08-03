@echo off
chcp 65001 > nul
echo [Notice] Starting model training...
poetry run python src/train.py
echo [Success] Training complete.
pause
