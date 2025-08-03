# 머신러닝 학습 예제: 붓꽃(Iris) 품종 예측

이 프로젝트는 머신러닝의 기본적인 워크플로우를 이해하기 위한 예제입니다.
붓꽃(Iris) 데이터셋을 사용하여 품종을 예측하는 간단한 분류(Classification) 모델을 만듭니다.

## 📖 기본 개념

### 머신러닝 워크플로우
1.  **데이터 준비**: 모델을 학습시킬 데이터를 수집하고 정제합니다. (`data/iris.csv`)
2.  **모델 학습**: 준비된 데이터로 모델을 학습(Training)시킵니다. 이 과정에서 모델은 데이터의 패턴을 학습합니다. (`src/train.py`)
3.  **모델 평가**: 학습된 모델이 얼마나 정확한지 테스트 데이터로 평가합니다.
4.  **예측**: 학습된 모델을 사용하여 새로운 데이터의 결과를 예측(Prediction)합니다. (`src/predict.py`)

### 붓꽃 데이터셋 (Iris Dataset)
-   **sepal_length**: 꽃받침 길이
-   **sepal_width**: 꽃받침 너비
-   **petal_length**: 꽃잎 길이
-   **petal_width**: 꽃잎 너비
-   **species**: 품종 (`setosa`, `versicolor`, `virginica` 중 하나)

모델은 4가지 특성(길이, 너비)을 기반으로 3가지 품종 중 하나를 맞추는 것을 목표로 합니다.

---

## 📂 프로젝트 구조

```
.
├── data/
│   └── iris.csv          # 붓꽃 샘플 데이터
├── models/
│   └── iris_model.pkl    # 학습된 모델이 저장되는 폴더
├── src/
│   ├── train.py          # 모델을 학습시키는 스크립트
│   └── predict.py        # 학습된 모델로 예측하는 스크립트
├── pyproject.toml        # Poetry 의존성 및 프로젝트 설정 파일
├── install.bat           # (Windows) 의존성 설치 스크립트
├── train.bat             # (Windows) 모델 학습 스크립트
├── predict.bat           # (Windows) 모델 예측 스크립트
├── install.sh            # (Linux/macOS) 의존성 설치 스크립트
├── train.sh              # (Linux/macOS) 모델 학습 스크립트
├── predict.sh            # (Linux/macOS) 모델 예측 스크립트
└── README.md             # 프로젝트 설명서
```

---

## 🚀 사용법

Windows와 Linux/macOS 환경에서 쉽게 실행할 수 있는 스크립트를 제공합니다.
[Poetry 설치](https://python-poetry.org/docs/#installation)가 먼저 필요합니다.

### 1. 환경 설정
-   **Windows**: `install.bat` 파일을 더블 클릭하여 실행하세요.
-   **Linux/macOS**: 터미널에서 `./install.sh` 를 실행하세요. (실행 권한이 없다면 `chmod +x *.sh` 명령으로 권한을 부여하세요.)

### 2. 모델 학습
-   **Windows**: `train.bat` 파일을 더블 클릭하여 실행하세요.
-   **Linux/macOS**: 터미널에서 `./train.sh` 를 실행하세요.

### 3. 새로운 데이터 예측
-   **Windows**: `predict.bat` 파일을 더블 클릭하여 실행하세요.
-   **Linux/macOS**: 터미널에서 `./predict.sh` 를 실행하세요.

---

이제 이 프로젝트를 기반으로 새로운 데이터를 `data/iris.csv`에 추가하거나, `src/train.py`의 모델을 다른 모델(예: `RandomForestClassifier`, `SVC`)로 변경해보며 머신러닝을 학습할 수 있습니다.
