#!/bin/bash
echo "모델 학습을 시작합니다..."
poetry run python src/train.py
echo "학습이 완료되었습니다."
