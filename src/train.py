# --- 라이브러리 임포트 ---
# 컴퓨터에게 필요한 도구들을 가져오라고 지시하는 부분입니다.

import pandas as pd  # pandas는 엑셀 시트처럼 생긴 데이터를 다루기 쉽게 해주는 도구입니다. (예: CSV 파일 읽기)
from sklearn.model_selection import train_test_split  # 데이터를 학습용과 테스트용으로 나눠주는 도구입니다.
from sklearn.linear_model import LogisticRegression  # 여러 종류의 머신러닝 모델 중 '로지스틱 회귀'라는 분류 모델입니다.
from sklearn.metrics import accuracy_score  # 모델이 얼마나 정답을 잘 맞췄는지 '정확도'를 계산해주는 도구입니다.
import joblib  # 학습이 끝난 모델을 파일로 저장하거나 불러올 때 사용하는 도구입니다.
import os  # 운영체제(예: Windows)와 상호작용하여 폴더를 만드는 등의 작업을 도와주는 도구입니다.

# --- 경로 설정 ---
# 데이터 파일과 모델 파일을 어디에 저장할지 미리 정해두는 부분입니다.

DATA_PATH = 'data/iris.csv'  # 우리가 사용할 붓꽃 데이터 파일의 위치입니다.
MODEL_PATH = 'models/iris_model.pkl'  # 학습이 완료된 모델을 저장할 파일의 위치와 이름입니다.

# --- 폴더 생성 ---
# 모델을 저장할 'models' 폴더가 없다면, 자동으로 폴더를 생성합니다.
# exist_ok=True는 폴더가 이미 있어도 에러를 발생시키지 말라는 의미입니다.
os.makedirs('models', exist_ok=True)

# --- 1. 데이터 준비 (Data Preparation) ---
print("1. 데이터를 로드합니다...")
# pandas(pd)를 사용해서 CSV 파일을 읽어와 iris_df라는 변수에 저장합니다.
# 이 변수에는 붓꽃 데이터가 표(DataFrame) 형태로 들어갑니다.
iris_df = pd.read_csv(DATA_PATH)

print("데이터의 처음 5줄:")
print(iris_df.head()) # .head()는 데이터가 어떻게 생겼는지 처음 5줄만 보여주는 기능입니다.

# --- 특성(X)과 레이블(y) 분리 ---
# 머신러닝 모델에게 "이게 문제(X)고, 이게 정답(y)이야"라고 알려주는 과정입니다.
#
# - 특성 (Features, X): '문제'에 해당하며, 예측의 기반이 되는 데이터입니다.
#   여기서는 꽃받침 길이/너비, 꽃잎 길이/너비 4가지가 특성입니다.
#
# - 레이블 (Label, y): '정답'에 해당하며, 모델이 맞춰야 하는 결과입니다.
#   여기서는 붓꽃의 품종('species')이 레이블입니다.

X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]  # 4개의 특성 열을 선택하여 X에 저장
y = iris_df['species']  # 'species' 열을 선택하여 y에 저장

print("\n문제(특성, X) 데이터와 정답(레이블, y) 데이터로 분리했습니다.")

# --- 2. 데이터 분할 (Data Splitting) ---
# "공부할 문제집(학습용 데이터)"과 "실전 모의고사(테스트용 데이터)"로 나누는 과정입니다.
# 모델이 학습 데이터만 너무 외워서, 처음 보는 데이터에 대해서는 예측을 못하는 '과적합(Overfitting)'을 방지하기 위함입니다.
#
# - X_train, y_train: 모델을 학습시킬 '문제집'과 '정답지' (전체의 80%)
# - X_test, y_test: 모델의 성능을 평가할 '모의고사' 문제와 정답 (전체의 20%)
#
# - test_size=0.2: 전체 데이터 중 20%를 테스트용으로 사용하겠다는 의미입니다. (80%는 학습용)
# - random_state=42: 데이터를 나눌 때 무작위로 섞는데, 이 값을 고정하면 언제 실행해도 항상 똑같은 방식으로 데이터가 나뉩니다.
#   이것은 실험 결과를 재현하기 위해 매우 중요합니다. (숫자 42는 관례적으로 많이 사용됩니다.)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n2. 총 {len(X)}개의 샘플 중 {len(X_train)}개를 학습용으로, {len(X_test)}개를 테스트용으로 분리합니다.")

# --- 3. 모델 학습 (Model Training) ---
print("\n3. 모델을 학습시킵니다...")
# '로지스틱 회귀'라는 분류 모델을 사용하겠습니다.
# 이 모델은 각 데이터가 특정 카테고리(여기서는 붓꽃 품종)에 속할 확률을 계산하여 분류합니다.
#
# - LogisticRegression(): 모델 객체를 생성합니다.
# - max_iter=200: 모델이 정답을 찾아가는 과정(최적화)을 최대 200번 반복하라는 의미입니다.
#   데이터가 복잡할 때 이 값을 늘려주면 학습이 더 잘 될 수 있습니다. 데이터가 작아서 기본값으로도 충분하지만 명시적으로 작성했습니다.
model = LogisticRegression(max_iter=200)

# .fit() 함수는 모델에게 학습용 데이터(X_train)와 정답(y_train)을 주고 "이제 공부해!"라고 지시하는 것과 같습니다.
# 이 과정을 통해 모델은 특성(X)과 레이블(y) 사이의 관계, 즉 패턴을 학습합니다.
model.fit(X_train, y_train)

print("학습이 완료되었습니다.")

# --- 4. 모델 평가 (Model Evaluation) ---
print("\n4. 학습된 모델을 평가합니다...")
# 학습이 끝난 모델에게 한 번도 본 적 없는 테스트 문제(X_test)를 주고 정답을 예측하게 합니다.
y_pred = model.predict(X_test)

# accuracy_score를 사용해 실제 정답(y_test)과 모델이 예측한 정답(y_pred)을 비교하여 정확도를 계산합니다.
# 정확도 = (정답을 맞춘 개수) / (전체 테스트 데이터 개수)
accuracy = accuracy_score(y_test, y_pred)

print(f"테스트 데이터에 대한 모델의 정확도: {accuracy:.4f}") # 소수점 4자리까지 정확도를 출력합니다.

# --- 5. 모델 저장 (Model Saving) ---
print(f"\n5. 학습된 모델을 '{MODEL_PATH}' 경로에 저장합니다.")
# joblib.dump()를 사용해 학습이 완료된 모델 객체(model)를 파일(MODEL_PATH)로 저장합니다.
# 이렇게 저장해두면, 나중에 예측이 필요할 때 다시 학습할 필요 없이 이 파일만 불러와서 바로 사용할 수 있습니다.
joblib.dump(model, MODEL_PATH)

print("\n모든 과정이 완료되었습니다.")