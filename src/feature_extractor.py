"""
Feature Extraction from Real Sensor Data (InfluxDB Format)
Version 8: 실측 데이터 기반, 상태 클래스 분류 없음
"""

import os
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from datetime import datetime

# =====================================
# 설정
# =====================================

# InfluxDB CSV 형식: _field로 구분되는 센서 타입
SENSOR_FIELDS = [
    'fields_pressure_hpa',
    'fields_accel_x', 'fields_accel_y', 'fields_accel_z',  # 가속도계 3축
    'fields_gyro_x', 'fields_gyro_y', 'fields_gyro_z'      # 자이로스코프 3축
]

# 주파수 도메인 사용 여부
# - True: 시간 도메인(5개) + 주파수 도메인(6개) = 11개 특징/축
# - False: 시간 도메인(5개) = 5개 특징/축
USE_FREQUENCY_DOMAIN = True

# 윈도우 설정
WINDOW_SIZE = 5.0  # 초 단위 (1개 데이터셋 = 5초)
WINDOW_OVERLAP = 0.0  # 초 단위 (겹침, 0이면 겹침 없음)

# 출력 디렉토리
OUTPUT_DIR = 'data/processed'
RAW_DATA_DIR = 'data/raw'

# 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# InfluxDB CSV 읽기 함수
# =====================================

def read_influxdb_csv(file_path):
    """
    InfluxDB CSV 형식 읽기 (다중 device_id 지원)
    
    파일 내에 여러 device_id가 있을 경우, 각 device_id마다 헤더가 반복됨
    이 함수는 모든 device_id의 데이터를 통합하여 반환
    
    InfluxDB 출력 형식:
    - 1~3행: 메타데이터 (#group, #datatype, #default)
    - 4행: 컬럼 헤더
    - 5행~: 실제 데이터 (Long format)
    - device_id가 변경되면 헤더 재등장
    
    Parameters:
    - file_path: CSV 파일 경로
    
    Returns:
    - DataFrame with columns: _time, field1, field2, ...
    - sampling_rate: 추정 샘플링 주파수 (Hz)
    """
    # 파일을 읽어서 헤더 위치 찾기
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 헤더가 등장하는 라인 번호 찾기 (#group으로 시작하는 줄)
    header_lines = []
    for i, line in enumerate(lines):
        if line.startswith('#group'):
            header_lines.append(i)
    
    # 각 device의 데이터를 읽어서 통합
    all_dataframes = []
    
    for idx, header_start in enumerate(header_lines):
        # 다음 헤더까지 또는 파일 끝까지
        if idx + 1 < len(header_lines):
            next_header = header_lines[idx + 1]
            section_lines = lines[header_start:next_header]
        else:
            section_lines = lines[header_start:]
        
        # 임시 파일로 저장하지 않고 StringIO 사용
        from io import StringIO
        section_text = ''.join(section_lines)
        
        # pandas로 읽기 (메타데이터 3줄 스킵)
        df_section = pd.read_csv(StringIO(section_text), skiprows=3)
        
        # 필요한 컬럼만 선택
        df_section = df_section[['_time', '_value', '_field']].copy()
        
        all_dataframes.append(df_section)
    
    # 모든 데이터 통합
    df = pd.concat(all_dataframes, ignore_index=True)
    
    # Long format → Wide format (각 field가 컬럼이 됨)
    df_pivot = df.pivot_table(index='_time', columns='_field', values='_value', aggfunc='first')
    df_pivot = df_pivot.reset_index()
    
    # 시간을 datetime으로 변환
    df_pivot['_time'] = pd.to_datetime(df_pivot['_time'])
    
    # 시간 정렬
    df_pivot = df_pivot.sort_values('_time').reset_index(drop=True)
    
    # 상대 시간 계산 (초 단위)
    df_pivot['Time(s)'] = (df_pivot['_time'] - df_pivot['_time'].iloc[0]).dt.total_seconds()
    
    # 샘플링 레이트 추정
    # 공식: (샘플 개수 - 1) / (전체 시간)
    # 이유:
    # 1. 1초 간격의 2개 샘플 → (2-1)/(1-0) = 1 Hz (올바름)
    # 2. 결측값이 있어도 전체 시간 범위를 기준으로 하므로 정확함
    # 3. 평균 간격의 역수보다 간단하고 안정적
    n_samples = len(df_pivot)
    total_time = df_pivot['Time(s)'].iloc[-1] - df_pivot['Time(s)'].iloc[0]
    sampling_rate = (n_samples - 1) / total_time if total_time > 0 else 1.0
    
    return df_pivot, sampling_rate

# =====================================
# 특징 추출 함수
# =====================================

def extract_features(signal, sampling_rate, use_freq_domain=USE_FREQUENCY_DOMAIN):
    """
    시간 도메인과 주파수 도메인 특징을 추출
    
    Parameters:
    - signal: 시간 도메인 신호 (1D numpy array)
    - sampling_rate: 샘플링 주파수 (Hz)
    - use_freq_domain: 주파수 도메인 특징 사용 여부
    
    Returns:
    - features: 추출된 특징 리스트
      * 시간 도메인 (5개): STD, Peak-to-Peak, Crest Factor, Impulse Factor, Mean
      * 주파수 도메인 (6개, 선택적): Dominant Freq, Spectral Centroid, Energy, Kurtosis, Skewness, Std
    
    Note: RMS 대신 STD를 사용하여 절대값 영향 제거 (압력 센서 등 오프셋이 큰 센서 고려)
    """
    N = len(signal)
    
    # ===== Time Domain Features =====
    abs_signal = np.abs(signal)
    max_val = np.max(abs_signal)
    abs_mean = np.mean(abs_signal)
    
    # Standard Deviation (변동성, 절대값 영향 없음)
    std = np.std(signal)
    
    # Peak-to-Peak
    peak_to_peak = np.ptp(signal)
    
    # Crest Factor (최대값 / RMS)
    rms = np.sqrt(np.mean(signal ** 2))
    crest_factor = max_val / rms if rms > 0 else 0
    
    # Impulse Factor (최대값 / 평균 절댓값)
    impulse_factor = max_val / abs_mean if abs_mean > 0 else 0
    
    # Mean (DC 성분, 압력 등의 기준값)
    mean_val = np.mean(signal)
    
    time_features = [std, peak_to_peak, crest_factor, impulse_factor, mean_val]
    
    # 주파수 도메인을 사용하지 않으면 시간 도메인만 반환
    if not use_freq_domain:
        return time_features
    
    # ===== Frequency Domain Features =====
    # DC 성분 제거
    signal_centered = signal - np.mean(signal)
    
    # RFFT 계산 (실수 신호 → 양의 주파수만)
    spectrum = np.abs(rfft(signal_centered))
    freqs = rfftfreq(N, 1/sampling_rate)
    
    # Dominant frequency (최대 진폭의 주파수)
    dominant_freq = freqs[np.argmax(spectrum)] if len(spectrum) > 0 else 0
    
    # Spectral centroid (주파수 무게중심)
    spectral_sum = np.sum(spectrum)
    spectral_centroid = np.sum(freqs * spectrum) / spectral_sum if spectral_sum > 0 else 0
    
    # Spectral energy (총 에너지)
    spectral_energy = np.sum(spectrum ** 2)
    
    # Spectral statistics
    spectral_kurt = kurtosis(spectrum, fisher=False) if len(spectrum) > 1 else 0
    spectral_skewness = skew(spectrum) if len(spectrum) > 1 else 0
    spectral_std = np.std(spectrum)
    
    freq_features = [
        dominant_freq,
        spectral_centroid,
        spectral_energy,
        spectral_kurt,
        spectral_skewness,
        spectral_std
    ]
    
    return time_features + freq_features

# =====================================
# 단일 파일 처리 함수 (윈도우 기반)
# =====================================

def process_single_file_windowed(file_path, window_size=WINDOW_SIZE, window_overlap=WINDOW_OVERLAP,
                                  fields=None, use_freq_domain=USE_FREQUENCY_DOMAIN):
    """
    InfluxDB CSV 파일을 읽어 윈도우 단위로 특징 추출
    
    Parameters:
    - file_path: CSV 파일 경로
    - window_size: 윈도우 크기 (초)
    - window_overlap: 윈도우 겹침 (초)
    - fields: 추출할 필드 리스트 (None이면 사용 가능한 모든 필드)
    - use_freq_domain: 주파수 도메인 특징 사용 여부
    
    Returns:
    - feature_matrix: (n_windows, n_features) 형태의 특징 행렬
    - field_names: 실제로 사용된 필드 이름 리스트
    - window_info: 각 윈도우의 정보 (시작 시각, 끝 시각, 샘플 개수)
    - sampling_rate: 샘플링 주파수
    """
    # 데이터 로드
    df, sampling_rate = read_influxdb_csv(file_path)
    
    # 사용 가능한 필드 확인
    available_fields = [col for col in df.columns if col not in ['_time', 'Time(s)']]
    
    # 지정된 필드가 없으면 모든 필드 사용
    if fields is None:
        fields = available_fields
    else:
        # 지정된 필드 중 실제로 존재하는 것만 사용
        fields = [f for f in fields if f in available_fields]
    
    # 윈도우 슬라이딩 설정
    window_step = window_size - window_overlap  # 윈도우 이동 간격
    total_time = df['Time(s)'].iloc[-1] - df['Time(s)'].iloc[0]
    
    # 윈도우 시작 시각 계산
    window_starts = []
    current_time = df['Time(s)'].iloc[0]
    while current_time + window_size <= df['Time(s)'].iloc[-1]:
        window_starts.append(current_time)
        current_time += window_step
    
    if len(window_starts) == 0:
        return None, fields, None, sampling_rate
    
    # 각 윈도우에 대해 특징 추출
    feature_list = []
    window_info_list = []
    feature_per_field = 11 if use_freq_domain else 5
    
    for window_start in window_starts:
        window_end = window_start + window_size
        
        # 현재 윈도우에 해당하는 데이터 추출
        mask = (df['Time(s)'] >= window_start) & (df['Time(s)'] < window_end)
        df_window = df[mask]
        
        if len(df_window) < 2:  # 최소 2개 샘플 필요
            continue
        
        # 각 필드에 대해 특징 추출
        window_features = []
        for field in fields:
            if field in df_window.columns:
                signal = df_window[field].dropna().values
                if len(signal) >= 2:
                    window_features.extend(extract_features(signal, sampling_rate, use_freq_domain))
                else:
                    # 신호가 너무 짧으면 0으로 채움
                    window_features.extend([0] * feature_per_field)
            else:
                # 필드가 없으면 0으로 채움
                window_features.extend([0] * feature_per_field)
        
        feature_list.append(window_features)
        window_info_list.append({
            'start_time': window_start,
            'end_time': window_end,
            'n_samples': len(df_window)
        })
    
    if len(feature_list) == 0:
        return None, fields, None, sampling_rate
    
    # NumPy 배열로 변환
    feature_matrix = np.array(feature_list)
    
    return feature_matrix, fields, window_info_list, sampling_rate

# =====================================
# 메인 실행 예시
# =====================================

if __name__ == "__main__":    
    # 테스트 파일 경로
    test_file = "data/raw/1107_data.csv"
    
    if os.path.exists(test_file):
        # 1. 윈도우 기반 처리 (주요 방식)
        
        feature_matrix, field_names, window_info, sampling_rate = process_single_file_windowed(
            test_file,
            window_size=WINDOW_SIZE,
            window_overlap=WINDOW_OVERLAP,
            fields=SENSOR_FIELDS,
            use_freq_domain=USE_FREQUENCY_DOMAIN
        )
        
        if feature_matrix is not None:            
            # 특징 이름 생성
            feature_names_list = []
            for field in field_names:
                for feat in ['std', 'ptp', 'crest', 'impulse', 'mean']:
                    feature_names_list.append(f"{field}_{feat}")
                
                if USE_FREQUENCY_DOMAIN:
                    for feat in ['dominant_freq', 'spectral_centroid', 'spectral_energy',
                                 'spectral_kurt', 'spectral_skew', 'spectral_std']:
                        feature_names_list.append(f"{field}_{feat}")
            
            # DataFrame으로 저장
            df_result = pd.DataFrame(feature_matrix, columns=feature_names_list)
            
            # 윈도우 메타데이터 추가
            df_result['window_id'] = range(len(feature_matrix))
            df_result['start_time'] = [info['start_time'] for info in window_info]
            df_result['end_time'] = [info['end_time'] for info in window_info]
            df_result['n_samples'] = [info['n_samples'] for info in window_info]
            
            output_file = os.path.join(OUTPUT_DIR, "1107_features.csv")
            df_result.to_csv(output_file, index=False)