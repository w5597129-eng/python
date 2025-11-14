"""
Isolation Forest 예측 스크립트
- 학습된 모델로 새로운 데이터 예측
- 이상 점수 및 레이블 반환
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# =====================================
# 설정
# =====================================

# 입력
INPUT_FEATURES = 'data/processed/1107_features.csv'  # 예측할 데이터

# 모델 경로
MODEL_PATH = 'models/isolation_forest.pkl'
SCALER_PATH = 'models/scaler.pkl'

# 출력
OUTPUT_DIR = 'results/isolation_forest'
OUTPUT_FILE = 'predictions.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# 1. 모델 및 스케일러 로드
# =====================================
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# =====================================
# 2. 데이터 로드
# =====================================
df = pd.read_csv(INPUT_FEATURES)

# 메타데이터 분리
metadata_cols = ['window_id', 'start_time', 'end_time', 'n_samples']
feature_cols = [col for col in df.columns if col not in metadata_cols]

X = df[feature_cols].values

# =====================================
# 3. 정규화
# =====================================
X_scaled = scaler.transform(X)

# =====================================
# 4. 예측
# =====================================

# 이상 점수
anomaly_scores = model.score_samples(X_scaled)

# 예측 (-1: 이상, 1: 정상)
predictions = model.predict(X_scaled)

# 통계
n_normal = np.sum(predictions == 1)
n_anomaly = np.sum(predictions == -1)

# =====================================
# 5. 결과 저장
# =====================================

# 결과 DataFrame 생성
df_results = df.copy()
df_results['anomaly_score'] = anomaly_scores
df_results['prediction'] = predictions
df_results['prediction_label'] = ['Normal' if p == 1 else 'Anomaly' for p in predictions]
df_results['prediction_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 저장
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
df_results.to_csv(output_path, index=False)

# =====================================
# 6. 이상 샘플 확인
# =====================================
if n_anomaly > 0:    
    # 이상으로 판별된 샘플
    anomaly_indices = np.where(predictions == -1)[0]
    
    # 점수 기준 상위 10개 (가장 이상한 것)
    top_k = min(10, n_anomaly)
    top_anomaly_indices = anomaly_indices[np.argsort(anomaly_scores[anomaly_indices])[:top_k]]
    
    for i, idx in enumerate(top_anomaly_indices, 1):
        score = anomaly_scores[idx]
        window_id = df_results.iloc[idx]['window_id']
        start_time = df_results.iloc[idx]['start_time']