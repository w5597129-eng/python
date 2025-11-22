# MOBY Edge Sensor Publisher

간단 소개
- 이 저장소는 라즈베리파이 기반 엣지 노드용 센서 수집 및 MQTT 퍼블리셔 스크립트 모음입니다.

핵심 파일/위치
- `src/sensor_final.py` : 통합 센서 퍼블리셔 및 터미널 출력(하드웨어 필요).
- `src/inference_interface.py` : 센서 퍼블리셔와 워커가 공유하는 메시지/토픽 스키마(라이브러리 모듈, 실행 불필요).
- `src/inference_worker.py` : MQTT로 윈도우를 구독해 특징 추출 → 스케일링 → 모델 추론을 수행하고 결과를 퍼블리시하는 워커.
- `src/feature_extractor.py` : 실제 특징 추출 구현(V17, PCA + 벡터 스칼라화). 멀티센서 파일 처리 및 윈도우 기반 피처 추출 제공.
- `feature_extractor.py` : 루트에 둔 경량 호환 래퍼로, `src.feature_extractor`를 재수출하여 `import feature_extractor` 호출을 지원함.
- `src/__init__.py` : `src`를 패키지로 만들어 임포트 호환성을 보장합니다.
- `models/` : 학습된 모델·스케일러 파일 (예: `isolation_forest.joblib`, `scaler_if.joblib`).
- `scripts/resave_models.py` : 기존 모델/스케일러를 로드해 joblib로 재저장하는 편리한 스크립트.

빠른 시작
1. (권장) 가상환경 생성 및 활성화:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. 필수 패키지 설치:
```bash
pip install -r requirements.txt
# 하드웨어 전용 의존성만 설치하려면:
pip install -r requirements-hw.txt
```
3. 항상 리포지토리 루트에서 실행하세요 (`/home/wise/python`). 상대경로로 모델/스크립트 파일을 찾습니다.

실행 예시
- 센서 퍼블리셔(하드웨어 필요 — 보통 `sudo`):
```bash
cd /home/wise/python
sudo python src/sensor_final.py
```
- 인퍼런스 워커(센서와 별개로 실행 — 다른 터미널에서):
```bash
cd /home/wise/python
python src/inference_worker.py
```
- 모델 재저장(권한 문제 없이 파이썬으로 실행):
```bash
cd /home/wise/python
python scripts/resave_models.py
```
참고: `scripts/resave_models.py`는 실행 권한이 없어도 `python scripts/resave_models.py`로 실행 가능합니다. 실행 권한을 주고 싶다면 `chmod +x scripts/resave_models.py` 후 `./scripts/resave_models.py`로 실행할 수 있습니다.

테스트용 로컬 MQTT 브로커
- 로컬 브로커가 필요하면 `mosquitto`를 설치하고 실행하세요:
```bash
sudo apt update
sudo apt install -y mosquitto
sudo systemctl enable --now mosquitto
```
- 토픽 수신 확인 예시:
```bash
# 윈도우 토픽(워커가 구독하는 토픽)
mosquitto_sub -h localhost -t "factory/inference/windows/#" -v

# 워커가 발행하는 결과 토픽 확인
mosquitto_sub -h localhost -t "factory/inference/results/#" -v
```

**최근 변경 및 주의사항**
- `feature_extractor` 개선: 특징 추출기가 `src/feature_extractor.py`의 V17 구현(PC A + 벡터 스칼라화, FFT 재사용)을 기준으로 갱신되었습니다. 이를 위해 루트에 경량 래퍼 `feature_extractor.py`를 추가했고 `src`를 패키지로 만들었습니다.
- `src/inference_worker.py`는 가능하면 `feature_extractor.extract_features_v17`를 사용해 센서-레벨(예: accel 3축, gyro 3축, pressure) 특징을 직접 추출하도록 변경되었습니다. 추출기가 사용 불가할 경우 기존의 1D 신호 기반 `extract_features` 폴백을 사용합니다.
- ONNX 관련: `ONNXMLPWrapper`는 기본적으로 CPU 실행 프로바이더(`CPUExecutionProvider`)를 사용합니다. GPU를 사용하려면 `onnxruntime-gpu` 설치 또는 `providers` 명시를 해주세요.
- 모델 호환성 주의: V17 피처 집합(특성 이름/순서/길이)이 기존 모델의 입력 차원과 다를 수 있습니다. 저장된 PyTorch 체크포인트(`.pth/.pt`)는 이 저장소의 기본 워커에서 자동 변환되지 않으므로, ONNX로 변환하거나 모델을 재학습/재저장하는 것을 권장합니다. `scripts/resave_models.py`로 스케일러를 재저장해 보고 모델 입력 차원을 확인하세요.

검증/실행 팁
- 가상환경 활성화(권장):
```bash
python3 -m venv .venv
source .venv/bin/activate
```
- 의존성 설치:
```bash
pip install -r requirements.txt
# 하드웨어 전용 의존성만 설치하려면:
pip install -r requirements-hw.txt
```
- `motor` 스크립트 동작 확인 (IR 이벤트가 있을 때 터미널에 publish 로그가 찍힙니다):
```bash
source .venv/bin/activate
python motor_slow.py
# 또는
python motor.py
```
- 로컬 MQTT 브로커가 없다면 `mosquitto`를 설치해서 테스트하면 편합니다(위의 토픽 수신 예시 참고).

문제 해결
- ONNX Runtime이 GPU를 탐지하며 나오는 경고를 제거하기 위해 워커는 기본적으로 CPU 실행 프로바이더를 사용합니다. GPU 가속을 원하면 `onnxruntime-gpu`를 설치하거나 `ONNXMLPWrapper(..., providers=["CUDAExecutionProvider"])`처럼 명시적으로 providers를 지정하세요.
- 하드웨어가 없는 로컬 개발 환경에서는 센서 드라이버 import가 실패할 수 있습니다. 이 경우 예외 분기나 도커/모킹을 이용해 테스트하세요.

모델/스케일러 파일
- 워커는 기본적으로 `models/isolation_forest.joblib` 및 `models/scaler_if.joblib` 같은 파일을 찾습니다. 상대 경로는 리포지토리 루트를 기준으로 해석되므로 `cd /home/wise/python`에서 실행하세요.
- 모델이 현재 환경에서 로드되지 않으면 `scripts/resave_models.py`로 재저장하거나, 필요한 라이브러리(scikit-learn 등)를 venv에 맞게 설치해야 합니다.

동작 요약
- `src/sensor_final.py`는 센서 데이터를 읽어 `factory/sensor/<type>` 토픽으로 퍼블리시하고, MPU6050 윈도우가 채워지면 `factory/inference/windows/<sensor_type>`에 윈도우 메시지를 보냅니다.
- `src/inference_worker.py`는 해당 윈도우를 구독해 특징 추출 → 스케일링 → 모델 추론을 수행하고 결과를 `factory/inference/results/<sensor_type>` (또는 모델별 토픽)로 퍼블리시합니다.
- `src/inference_interface.py`는 `WindowMessage` / `InferenceResultMessage` 등의 스키마를 정의하는 라이브러리 모듈로 두 스크립트가 import해서 사용합니다. 직접 실행할 필요는 없습니다.

권장 실행 방식
- 센서(퍼블리셔)와 워커를 각각 별도 터미널(또는 systemd 서비스)로 띄워 두 프로세스가 동일한 MQTT 브로커를 통해 통신하도록 하세요.
- 로컬 테스트 시 하드웨어가 없으면 센서 읽기 부분 일부가 실패할 수 있으니, 모킹 또는 `mosquitto`로 직접 테스트 메시지를 publish 하여 워커 동작을 확인하세요.

부가 정보
- `paho-mqtt`의 콜백 API 버전 관련 DeprecationWarning은 동작에는 영향이 크지 않습니다. 필요시 `_make_mqtt_client()`에서 경고를 억제하거나 `paho-mqtt`를 업그레이드 하세요.
- 권한 문제(허가 거부)는 스크립트를 직접 실행하려 할 때 발생할 수 있습니다. 권한 대신 `python scripts/...` 형태로 실행하는 것을 권장합니다.

문의/다음 단계
- README에 추가할 내용(예: systemd 서비스 예시, 더미 데이터 주입 스크립트, Dockerfile)이 있으면 알려주세요.

---
파일은 현재 리포지토리 상태에 맞추어 업데이트되었습니다.
