#!/usr/bin/env python3
"""
Feature Extraction from Sensor Data (InfluxDB Format)
Version 15 (Fixed):
- Handles multi-file CSV format
- Configurable sensor field selection
- Window-based feature extraction only
- FIXED: Multi-sensor synchronization using outer join
- FIXED: Prevent NaN [*_SpectralKurtosis, *_SpectralSkewness]

Author: WISE Team, Project MOBY
Date: 2025-11-19
"""

import os
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =====================================
# 설정
# =====================================

# 추출할 센서 필드 지정
# 이 필드들만 특징 추출에 사용됨
SENSOR_FIELDS = [
    'fields_pressure_hpa',
    'fields_accel_x', 'fields_accel_y', 'fields_accel_z',
    'fields_gyro_x', 'fields_gyro_y', 'fields_gyro_z'
]

# 주파수 도메인 사용 여부
USE_FREQUENCY_DOMAIN = True

# 윈도우 설정
WINDOW_SIZE = 5.0  # 초 단위
WINDOW_OVERLAP = 0.0  # 초 단위 (겹침, 0이면 겹침 없음)

# 출력 디렉토리
OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# CSV 읽기 함수
# =====================================

def read_influxdb_csv(file_path: str) -> Tuple[pd.DataFrame, float]:
    """
    InfluxDB CSV 파일 읽기 (혼합 타입 처리)
    
    Parameters:
    - file_path: CSV 파일 경로
    
    Returns:
    - DataFrame: 시계열 데이터 (wide format)
    - sampling_rate: 추정 샘플링 주파수 (Hz)
    """
    # Check for multiple header sections
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header_lines = [i for i, line in enumerate(lines) if line.startswith('#group')]
    
    if len(header_lines) > 1:
        # Multiple headers - read each section
        print(f"  File has {len(header_lines)} header sections")
        all_dfs = []
        
        for idx, header_start in enumerate(header_lines):
            if idx + 1 < len(header_lines):
                section_end = header_lines[idx + 1]
            else:
                section_end = len(lines)
            
            from io import StringIO
            section_text = ''.join(lines[header_start:section_end])
            
            try:
                df_section = pd.read_csv(StringIO(section_text), skiprows=3, 
                                        dtype={'_time': str, '_value': float, '_field': str})
                all_dfs.append(df_section)
            except Exception as e:
                continue
        
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            raise ValueError("Could not read any valid sections")
    else:
        # Single header
        df = pd.read_csv(file_path, skiprows=3, 
                         dtype={'_time': str, '_field': str},
                         low_memory=False)
    
    # Clean data
    df = df[df['_time'].notna() & (df['_time'] != '_time')]
    df = df[df['_field'].notna() & (df['_field'] != '_field')]
    df['_value'] = pd.to_numeric(df['_value'], errors='coerce')
    df = df[df['_value'].notna()]
    
    # Pivot to wide format
    df_pivot = df.pivot_table(
        index='_time',
        columns='_field',
        values='_value',
        aggfunc='first'
    ).reset_index()
    
    # Parse time
    df_pivot['_time'] = pd.to_datetime(df_pivot['_time'], format='mixed', utc=True)
    df_pivot = df_pivot.sort_values('_time').reset_index(drop=True)
    
    # Calculate relative time
    df_pivot['Time(s)'] = (df_pivot['_time'] - df_pivot['_time'].iloc[0]).dt.total_seconds()
    
    # Estimate sampling rate
    n_samples = len(df_pivot)
    total_time = df_pivot['Time(s)'].iloc[-1] - df_pivot['Time(s)'].iloc[0]
    sampling_rate = (n_samples - 1) / total_time if total_time > 0 else 1.0
    
    return df_pivot, sampling_rate

# =====================================
# 특징 추출 함수
# =====================================

def extract_features(signal: np.ndarray, sampling_rate: float,
                     use_freq_domain: bool = USE_FREQUENCY_DOMAIN) -> List[float]:
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
    
    Note:
    - NaN 방지 처리 포함:
      * SpectralKurtosis: NaN → 3.0 (정규분포의 kurtosis)
      * SpectralSkewness: NaN → 0.0 (대칭 분포의 skewness)
    """
    if len(signal) < 2:
        feature_count = 11 if use_freq_domain else 5
        return [0.0] * feature_count
    
    N = len(signal)
    
    # ===== Time Domain Features =====
    abs_signal = np.abs(signal)
    max_val = np.max(abs_signal)
    abs_mean = np.mean(abs_signal)
    
    # Standard Deviation (변동성)
    std = np.std(signal)
    
    # Peak-to-Peak
    peak_to_peak = np.ptp(signal)
    
    # Crest Factor
    rms = np.sqrt(np.mean(signal ** 2))
    crest_factor = max_val / rms if rms > 0 else 0
    
    # Impulse Factor
    impulse_factor = max_val / abs_mean if abs_mean > 0 else 0
    
    # Mean (DC 성분)
    mean_val = np.mean(signal)
    
    time_features = [std, peak_to_peak, crest_factor, impulse_factor, mean_val]
    
    if not use_freq_domain:
        return time_features
    
    # ===== Frequency Domain Features =====
    # DC 성분 제거
    signal_centered = signal - np.mean(signal)
    
    # RFFT 계산
    spectrum = np.abs(rfft(signal_centered))
    freqs = rfftfreq(N, 1/sampling_rate)
    
    # Dominant frequency
    dominant_freq = freqs[np.argmax(spectrum)] if len(spectrum) > 0 else 0
    
    # Spectral centroid
    spectral_sum = np.sum(spectrum)
    spectral_centroid = np.sum(freqs * spectrum) / spectral_sum if spectral_sum > 0 else 0
    
    # Spectral energy
    spectral_energy = np.sum(spectrum ** 2)
    
    # Spectral statistics
    # NaN 방지: 스펙트럼이 너무 짧거나 균일하면 기본값 사용
    if len(spectrum) > 1:
        spectral_kurt = kurtosis(spectrum, fisher=False)
        spectral_skewness = skew(spectrum)
        
        # NaN 체크 및 기본값 설정
        spectral_kurt = 3.0 if np.isnan(spectral_kurt) else spectral_kurt
        spectral_skewness = 0.0 if np.isnan(spectral_skewness) else spectral_skewness
    else:
        spectral_kurt = 3.0  # 정규분포의 kurtosis
        spectral_skewness = 0.0  # 대칭 분포의 skewness
    
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
# 다중 센서 파일 처리 (동기화 포함)
# =====================================

def process_multi_sensor_files(file_dict: Dict[str, str],
                                resample_rate: str = '100ms',
                                window_size: float = WINDOW_SIZE,
                                window_overlap: float = WINDOW_OVERLAP,
                                fields: List[str] = None) -> pd.DataFrame:
    """
    여러 센서 파일을 동기화하여 특징 추출
    
    수정사항 (V14):
    - Outer join 방식으로 센서 길이 불일치 문제 해결
    - 각 센서를 독립적으로 리샘플링한 후 병합
    
    Parameters:
    - file_dict: {sensor_type: file_path} 딕셔너리
    - resample_rate: 동기화 시 리샘플링 주기
    - window_size: 윈도우 크기 (초)
    - window_overlap: 윈도우 겹침 (초)
    - fields: 추출할 필드 리스트 (None이면 SENSOR_FIELDS 사용)
    
    Returns:
    - 특징 DataFrame
    """
    if fields is None:
        fields = SENSOR_FIELDS
    
    print("\n=== Multi-Sensor Processing ===")
    print(f"Target fields: {fields}")
    
    # 1. 각 센서 파일 독립적으로 읽고 리샘플링
    resampled_dfs = []
    sensor_info = []
    
    for sensor_type, file_path in file_dict.items():
        if os.path.exists(file_path):
            try:
                df, sr = read_influxdb_csv(file_path)
                sensor_info.append(f"{sensor_type}: {len(df)} samples @ {sr:.2f} Hz")
                
                # 인덱스 설정
                df_indexed = df.set_index('_time')
                
                # 해당 센서의 필드들 선택
                sensor_fields = [col for col in df_indexed.columns if col in fields]
                
                if sensor_fields:
                    # 리샘플링
                    df_resampled = df_indexed[sensor_fields].resample(resample_rate).mean()
                    resampled_dfs.append(df_resampled)
                    
            except Exception as e:
                print(f"  Error reading {sensor_type}: {e}")
    
    # 센서 정보 출력
    for info in sensor_info:
        print(f"  {info}")
    
    if not resampled_dfs:
        return pd.DataFrame()
    
    # 2. Outer join으로 병합
    print(f"\nSynchronizing at {resample_rate}...")
    
    # 첫 번째 DataFrame부터 시작
    merged_df = resampled_dfs[0].copy()
    
    # 나머지 DataFrame들을 outer join
    for df in resampled_dfs[1:]:
        merged_df = merged_df.join(df, how='outer')
    
    # 3. NaN 보간
    merged_df = merged_df.ffill().bfill()
    
    # 4. 상대 시간 추가
    merged_df = merged_df.reset_index()
    merged_df['Time(s)'] = (merged_df['_time'] - merged_df['_time'].iloc[0]).dt.total_seconds()
    
    print(f"Synchronized: {len(merged_df)} samples")
    
    # NaN 체크 (디버깅용)
    nan_counts = merged_df[[col for col in fields if col in merged_df.columns]].isna().sum()
    if nan_counts.any():
        print(f"Warning: NaN values remaining after interpolation:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"  {col}: {nan_counts[col]} NaN values")
    
    # 5. 윈도우 기반 특징 추출
    window_step = window_size - window_overlap
    
    # 샘플링 레이트 추정 (리샘플링 후)
    n_samples = len(merged_df)
    total_time = merged_df['Time(s)'].iloc[-1] - merged_df['Time(s)'].iloc[0]
    effective_sr = (n_samples - 1) / total_time if total_time > 0 else 1.0
    
    print(f"\nExtracting features with {window_size}s windows (overlap: {window_overlap}s)...")
    print(f"Effective sampling rate after resampling: {effective_sr:.2f} Hz")
    
    # 특징 추출
    features_list = []
    
    start_time = merged_df['Time(s)'].iloc[0]
    end_time = merged_df['Time(s)'].iloc[-1]
    
    current_time = start_time
    window_count = 0
    
    while current_time + window_size <= end_time:
        window_end = current_time + window_size
        
        # 윈도우 데이터 추출
        window_mask = (merged_df['Time(s)'] >= current_time) & (merged_df['Time(s)'] < window_end)
        window_data = merged_df[window_mask]
        
        if len(window_data) < 2:
            current_time += window_step
            continue
        
        # 각 필드별로 특징 추출
        window_features = {
            'window_id': window_count,
            'start_time': current_time,
            'end_time': window_end,
        }
        
        for field in fields:
            if field in window_data.columns:
                signal = window_data[field].values
                
                # NaN 제거
                signal = signal[~np.isnan(signal)]
                
                if len(signal) > 0:
                    features = extract_features(signal, effective_sr, USE_FREQUENCY_DOMAIN)
                    
                    # 특징 이름 생성
                    if USE_FREQUENCY_DOMAIN:
                        feature_names = ['STD', 'PeakToPeak', 'CrestFactor', 'ImpulseFactor', 'Mean',
                                       'DominantFreq', 'SpectralCentroid', 'SpectralEnergy', 
                                       'SpectralKurtosis', 'SpectralSkewness', 'SpectralStd']
                    else:
                        feature_names = ['STD', 'PeakToPeak', 'CrestFactor', 'ImpulseFactor', 'Mean']
                    
                    for feat_name, feat_val in zip(feature_names, features):
                        window_features[f"{field}_{feat_name}"] = feat_val
        
        features_list.append(window_features)
        window_count += 1
        current_time += window_step
    
    print(f"Extracted features from {window_count} windows")
    
    return pd.DataFrame(features_list)
# =====================================
# 메인 실행 예시
# =====================================

if __name__ == "__main__":    
    # 다중 센서 파일 처리
    print("\n" + "=" * 60)
    print("Multi-sensor files processing")
    print("=" * 60)
    
    sensor_files = {
        'accel_gyro': 'data/raw/1118 sensor_data/1118_accel_gyro.csv',
        'pressure': 'data/raw/1118 sensor_data/1118_pressure.csv',
    }
    
    # 모든 파일이 존재하는지 확인
    valid_files = {k: v for k, v in sensor_files.items() if os.path.exists(v)}
    
    if valid_files:
        features = process_multi_sensor_files(
            valid_files,
            resample_rate='78.125ms',  # ~12.8Hz
            window_size=WINDOW_SIZE,
            window_overlap=WINDOW_OVERLAP,
            fields=SENSOR_FIELDS  # 필드 세트 사용
        )
        
        if not features.empty:
            output_path = os.path.join(OUTPUT_DIR, "1118_features_fluctuating.csv")
            features.to_csv(output_path, index=False)
            print(f"\nSaved to: {output_path}")