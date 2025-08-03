# --- 라이브러리 임포트 ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os
from konlpy.tag import Okt  # 한국어 자연어 처리를 위한 형태소 분석기

# --- 경로 및 상수 설정 ---
DATA_PATH = 'data/accident_data.csv'
MODEL_PATH = 'models/accident_model.pkl'
os.makedirs('models', exist_ok=True)

# --- 1. 데이터 로드 ---
print("1. 데이터를 로드합니다...")
df = pd.read_csv(DATA_PATH)

# 데이터 확인: 레이블(accident_type)이 몇 종류고, 각 종류별로 데이터가 몇 개씩 있는지 확인합니다.
# 데이터가 불균형할 경우, 모델 성능에 영향을 줄 수 있습니다.
print("\n[데이터 요약]")
print(df['accident_type'].value_counts())

# --- 2. 데이터 전처리 및 모델 학습 준비 ---
# 머신러닝 모델은 텍스트(한글)를 직접 이해할 수 없습니다.
# 따라서, 텍스트를 숫자 형태의 벡터(Vector)로 변환해주는 과정이 반드시 필요합니다.
# 이 과정을 '피처 엔지니어링(Feature Engineering)' 또는 '벡터화(Vectorization)'라고 부릅니다.

# 단계 2-1: 텍스트를 의미있는 작은 단위로 쪼개기 (토큰화, Tokenization)
# - "아버지가방에들어가신다" -> "아버지", "가방", "에", "들어가신다"
# - 그냥 띄어쓰기로만 나누면 "아버지가방에들어가신다" 같은 경우 의미가 왜곡됩니다.
# - 한국어는 조사가 발달했기 때문에, 의미의 최소 단위인 '형태소'로 분석하는 것이 효과적입니다.
# - 여기서는 KoNLPy의 Okt 형태소 분석기를 사용합니다.
okt = Okt()

def tokenizer(text):
    # okt.morphs(text): 텍스트를 형태소 단위로 나눔 (예: ['폭설', '로', '인해', '도로', '가', '마비'])
    # stem=True: '마비되다', '마비되고' 등을 원형인 '마비되다'로 통일해주는 '어간 추출' 기능입니다.
    # 이를 통해 단어의 종류를 줄여 모델의 계산 효율성을 높입니다.
    return okt.morphs(text, stem=True)

# 단계 2-2: 쪼갠 단어들을 숫자 벡터로 변환하기 (벡터화, Vectorization)
# - TfidfVectorizer: 단어의 중요도를 계산하여 벡터로 만드는 가장 대중적인 방법입니다.
#   - TF(Term Frequency): 특정 문서에서 특정 단어가 얼마나 자주 등장하는가? (단순 빈도)
#   - IDF(Inverse Document Frequency): 특정 단어가 여러 문서에 걸쳐 얼마나 흔하게 등장하는가?
#     ("그리고", "하지만" 처럼 모든 문서에 자주 나오는 단어는 중요도를 낮추고,
#      "폭설", "타이어" 처럼 특정 문서에만 나오는 단어의 중요도를 높입니다.)
# - tokenizer=tokenizer: 위에서 정의한 우리만의 한국어 토크나이저를 사용하라고 지정합니다.
# - ngram_range=(1, 2): 단어를 1개씩(예: '폭설') 볼 뿐만 아니라, 2개씩 묶어서(예: '폭설', '도로')도 함께 보도록 합니다.
#   '중앙선'과 '침범'이 따로 있을 때보다 '중앙선 침범'이 함께 있을 때 더 강한 의미를 가지기 때문입니다.
# - min_df=3, max_df=0.9: 단어가 너무 드물게(3번 미만) 또는 너무 흔하게(전체 문서의 90% 이상) 나타나면,
#   분류에 별 도움이 안 될 가능성이 높으므로 무시하라는 옵션입니다. (현재 데이터는 작아서 적용 안함)
tfidf_vect = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))

# 단계 2-3: 모델 정의
# 로지스틱 회귀는 여러 클래스(사고 유형) 중 하나를 예측하는 데 효과적인 기본 모델입니다.
# class_weight='balanced' 옵션은 데이터가 불균형할 때, 적은 수의 클래스에 더 높은 가중치를 부여하여
# 모델이 소수 클래스를 더 잘 학습하도록 돕는 중요한 기능입니다.
lr_clf = LogisticRegression(random_state=42, class_weight='balanced')

# --- 3. 파이프라인(Pipeline) 설계 ---
# '파이프라인'은 위에서 정의한 여러 단계(토큰화, 벡터화, 모델 학습)를 하나로 묶어주는 конвейер입니다.
# 장점:
# 1. 코드 간소화: 전처리부터 학습까지 한 줄의 코드로 실행할 수 있습니다.
# 2. 실수 방지: 학습 데이터와 테스트 데이터에 동일한 전처리 과정을 일관되게 적용해주므로 실수를 줄입니다.
# 3. 재사용성: 이 파이프라인 객체 하나만 저장하면, 나중에 예측할 때도 똑같은 전처리 과정을 거쳐 예측할 수 있습니다.
pipeline = Pipeline([
    ('tfidf_vect', tfidf_vect), # 'tfidf_vect'라는 이름으로 TF-IDF 벡터화 단계를 추가
    ('lr_clf', lr_clf)         # 'lr_clf'라는 이름으로 로지스틱 회귀 모델 단계를 추가
])

# --- 4. 데이터 분할 및 모델 학습 ---
print("\n2. 데이터를 학습용과 테스트용으로 분리합니다...")
# 특성(X)은 사고 내용 텍스트, 레이블(y)은 사고 유형입니다.
X = df['text']
y = df['accident_type']

# 데이터를 학습용(70%)과 테스트용(30%)으로 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# stratify=y: 원본 데이터의 레이블 비율을 학습/테스트 데이터에도 동일하게 유지해줍니다.
# 예를 들어, 원본에 '자연재해'가 20% 있다면, 테스트 데이터에도 '자연재해'가 약 20% 있도록 나눠줍니다.
# 데이터가 불균형할 때 매우 중요한 옵션입니다.

print("\n3. 모델(파이프라인)을 학습시킵니다...")
# pipeline.fit()을 호출하면, 데이터가 파이프라인을 따라 순서대로 처리됩니다.
# 1) X_train 데이터가 tfidf_vect로 들어가 숫자 벡터로 변환됩니다.
# 2) 변환된 숫자 벡터와 y_train 정답지가 lr_clf 모델에 들어가 학습이 진행됩니다.
pipeline.fit(X_train, y_train)
print("학습이 완료되었습니다.")

# --- 5. 모델 평가 ---
print("\n4. 학습된 모델을 평가합니다...")
# pipeline.predict()를 호출하면, 테스트 데이터(X_test)가 파이프라인을 따라 순서대로 처리됩니다.
# 1) X_test 데이터가 학습 때와 '똑같은' 방식의 tfidf_vect로 들어가 숫자 벡터로 변환됩니다.
# 2) 변환된 벡터를 lr_clf 모델이 예측하여 결과를 반환합니다.
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 데이터에 대한 모델의 정확도: {accuracy:.4f}")

# --- 6. 모델 저장 ---
print(f"\n5. 학습된 파이프라인을 '{MODEL_PATH}' 경로에 저장합니다.")
# 모델 객체(lr_clf)만 저장하는 것이 아니라,
# 전처리(tfidf_vect) 과정까지 모두 포함된 파이프라인(pipeline) 전체를 저장해야 합니다.
# 그래야 나중에 새로운 텍스트가 들어왔을 때 동일한 전처리 후 예측을 할 수 있습니다.
joblib.dump(pipeline, MODEL_PATH)
print("\n모든 과정이 완료되었습니다.")
