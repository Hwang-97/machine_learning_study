@echo off
chcp 65001 > nul
echo [Notice] Starting prediction...
poetry run python src/predict.py
echo [Success] Prediction complete.
pause
