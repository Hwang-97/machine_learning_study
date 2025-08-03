# --- 라이브러리 임포트 ---
# 컴퓨터에게 필요한 도구들을 가져오라고 지시하는 부분입니다.

import joblib  # 학습이 끝난 모델을 파일에서 불러올 때 사용하는 도구입니다.
import pandas as pd  # pandas는 엑셀 시트처럼 생긴 데이터를 다루기 쉽게 해주는 도구입니다.

# --- 경로 설정 ---
# 우리가 저장해둔 모델 파일이 어디에 있는지 알려줍니다.
MODEL_PATH = 'models/iris_model.pkl'

# --- 1. 모델 불러오기 (Model Loading) ---
print(f"1. '{MODEL_PATH}' 경로에서 학습된 모델을 불러옵니다.")

# 저장된 모델 파일을 불러오기 위해 try-except 구문을 사용합니다.
# 이것은 혹시 파일이 없을 경우 발생할 수 있는 에러를 방지하기 위함입니다.
try:
    # joblib.load() 함수를 사용해 파일에 저장되어 있던 모델을 불러와 model 변수에 할당합니다.
    # 이제 'model' 변수는 train.py에서 학습했던 모든 내용을 기억하고 있는 상태입니다.
    model = joblib.load(MODEL_PATH)
    print("모델을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    # 만약 'models/iris_model.pkl' 파일이 없다면 이 메시지를 출력하고 프로그램을 종료합니다.
    print(f"에러: '{MODEL_PATH}' 경로에 모델 파일이 없습니다.")
    print("먼저 'python src/train.py' 또는 'train.bat'/'train.sh'를 실행하여 모델을 학습시켜주세요.")
    exit() # 프로그램 종료

# --- 2. 새로운 데이터 준비 (New Data Preparation) ---
# 이제, 학습된 모델에게 한 번도 보여준 적 없는 새로운 붓꽃 데이터를 주고 품종을 예측해달라고 요청할 것입니다.
#
# - pd.DataFrame({...}) : pandas의 DataFrame 형식으로 새로운 데이터를 만듭니다.
#   모델을 학습시킬 때 사용했던 데이터와 똑같은 구조(동일한 열 이름)를 가져야 합니다.
#   (sepal_length, sepal_width, petal_length, petal_width)
new_data = pd.DataFrame({
    'sepal_length': [5.1, 6.5, 6.0],
    'sepal_width': [3.5, 3.0, 2.2],
    'petal_length': [1.4, 5.2, 5.0],
    'petal_width': [0.2, 2.0, 1.5]
})

print("\n2. 예측할 새로운 데이터:")
print(new_data)

# --- 3. 예측 수행 (Prediction) ---
# .predict() 함수를 사용하여 모델에게 새로운 데이터(new_data)의 품종이 무엇일지 물어봅니다.
# 모델은 내부적으로 학습된 지식을 바탕으로 각 데이터의 품종을 예측하여 결과를 반환합니다.
predictions = model.predict(new_data)

# --- 4. 예측 결과 출력 ---
print("\n3. 모델의 예측 결과:")
# for문을 사용해 각 데이터에 대한 예측 결과를 하나씩 출력합니다.
# enumerate는 순서(i)와 값(prediction)을 함께 제공해줍니다. (i는 0부터 시작)
for i, prediction in enumerate(predictions):
    print(f"데이터 {i+1} -> 예측된 품종: {prediction}")

# --- (선택 사항) 각 품종별 예측 확률 보기 ---
# .predict()가 가장 가능성 높은 '정답' 하나만 알려주는 반면,
# .predict_proba()는 각 품종('setosa', 'versicolor', 'virginica')에 대해
# 모델이 얼마나 확신하는지를 '확률'로 보여줍니다.
if hasattr(model, "predict_proba"): # 모델이 predict_proba 기능을 지원하는지 확인
    probabilities = model.predict_proba(new_data)

    print("\n4. 각 품종별 예측 확률 (모델의 확신도):")
    for i, prob in enumerate(probabilities):
        # model.classes_ 에는 품종 이름 ['setosa', 'versicolor', 'virginica'] 가 순서대로 들어있습니다.
        # 이 이름과 위에서 계산된 확률(prob)을 짝지어 더 보기 좋게 출력합니다.
        # 예: {'setosa': 0.95, 'versicolor': 0.05, 'virginica': 0.0}
        class_probabilities = {model.classes_[j]: f"{p:.2%}" for j, p in enumerate(prob)}
        print(f"데이터 {i+1} -> {class_probabilities}")

print("\n모든 과정이 완료되었습니다.")