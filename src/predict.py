# --- 라이브러리 임포트 ---
import joblib
import pandas as pd
from konlpy.tag import Okt

# train.py에서 모델(파이프라인)을 만들 때 사용했던 것과 동일한 토크나이저 함수를 정의해야 합니다.
# 저장된 모델(pipeline) 객체를 불러올 때, 이 함수를 찾기 때문입니다.
okt = Okt()
def tokenizer(text):
    return okt.morphs(text, stem=True)

# --- 경로 및 상수 설정 ---
MODEL_PATH = 'models/accident_model.pkl'
DATA_PATH = 'data/accident_data.csv'

# --- 1. 모델 및 데이터 불러오기 ---
print(f"1. '{MODEL_PATH}' 경로에서 학습된 파이프라인을 불러옵니다.")
try:
    # train.py에서 저장했던 파이프라인 객체 전체를 불러옵니다.
    # 이 파이프라인은 텍스트 전처리(토큰화, 벡터화)와 모델을 모두 포함하고 있습니다.
    pipeline = joblib.load(MODEL_PATH)
    print("모델을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"에러: '{MODEL_PATH}' 경로에 모델 파일이 없습니다.")
    print("먼저 'train.bat' 또는 'train.sh'를 실행하여 모델을 학습시켜주세요.")
    exit()

# 대응 방안(action) 정보를 불러오기 위해 원본 데이터를 로드합니다.
# 나중에 예측된 '사고 유형'에 맞는 '대응 방안'을 찾아주기 위함입니다.
try:
    df = pd.read_csv(DATA_PATH)
    # 'accident_type'을 key로, 'action'을 value로 하는 딕셔너리를 만들어 둡니다.
    # drop_duplicates()를 사용해 각 사고 유형별로 하나의 대응 방안만 남깁니다.
    action_map = df.drop_duplicates(subset=['accident_type']).set_index('accident_type')['action'].to_dict()
except FileNotFoundError:
    print(f"에러: '{DATA_PATH}' 경로에 데이터 파일이 없습니다.")
    exit()


# --- 2. 예측할 새로운 텍스트 데이터 ---
# 사용자가 입력했다고 가정한 여러 예시 문장들입니다.
# 이 문장들은 학습 데이터에 없던 새로운 문장들입니다.
new_texts = [
    "오늘 새벽에 눈이 많이 올 것으로 예상됩니다.",
    "음주운전 차량이 사고를 냈어요.",
    "차가 미끄러져서 가드레일을 들이받았어요.",
    "앞차가 갑자기 서는 바람에 뒤에서 박았습니다."
]

print("\n--- 새로운 텍스트에 대한 사고 유형 및 대응 방안 예측 ---")

# --- 3. 예측 수행 및 결과 출력 ---
for text in new_texts:
    print(f"\n[입력 문장]: {text}")

    # pipeline.predict()에 새로운 텍스트를 리스트 형태로 넣어 예측을 요청합니다.
    # 파이프라인 내부에서 자동으로 아래 과정이 수행됩니다.
    # 1. KoNLPy 형태소 분석기로 텍스트를 토큰화 (예: '눈', '많이', '오다')
    # 2. 학습 때 사용된 TF-IDF 기준으로 텍스트를 숫자 벡터로 변환
    # 3. 로지스틱 회귀 모델이 숫자 벡터를 기반으로 사고 유형 예측
    predicted_type = pipeline.predict([text])[0] # 결과는 리스트 형태이므로 첫 번째 요소를 가져옵니다.

    # 위에서 만들어둔 action_map 딕셔너리를 사용해, 예측된 사고 유형에 맞는 대응 방안을 찾습니다.
    predicted_action = action_map.get(predicted_type, "일치하는 대응 방안 정보가 없습니다.")

    print(f"  > 예측된 사고 유형: {predicted_type}")
    print(f"  > 추천 대응 방안: {predicted_action}")

print("\n\n모든 예측 과정이 완료되었습니다.")
