## 목적
이 저장소는 라즈베리파이 기반의 센서 수집/퍼블리셔 스크립트 모음입니다. AI 코파일럿(또는 다른 자동화 에이전트)이 빠르게 작업을 시작할 수 있도록, 아키텍처와 실행/디버깅 요령, 코드 관례를 간결하게 정리합니다.

## 한눈에 보는 구조
- 각 파일은 독립 실행형 Python 스크립트입니다 (예: `sensor_final.py`, `data_collector.py`, `sensor.py`).
- 주요 역할
  - 센서 읽기 및 MQTT 퍼블리시: `data_collector.py`, `sensor_final.py`
  - 센서별 샘플/드라이버: `DHT11.py`, `bmp180.py`, `bmp280.py`, `accel_gyro.py` 계열
  - MQTT 샘플 퍼블리셔/구독자: `dht11_mqtt.py`, `IR_mqtt.py`, `IR_mqtt_pub.py`

## 핵심 패턴 / 규칙 (프로젝트 특이)
- 하드코드된 구성: 브로커 주소, MQTT 토픽, I2C 주소, 핀(BCM) 등이 스크립트 상단에 직접 정의되어 있습니다. 수정 시 해당 파일의 상단 섹션을 찾아 변경하세요.
- 하드웨어 종속성에 대해 try/except로 안전하게 처리: BMP 센서(`Adafruit_BMP`) 같은 모듈은 없을 수 있으므로 코드에서 옵셔널로 처리됩니다. 에이전트는 import 실패를 예상하고 대체 흐름(로그, 모형 데이터) 제안을 할 것.
- 실행 방식: 각 파일은 `if __name__ == "__main__": main()` 패턴으로 실행됩니다. 즉 단일 파일 단위로 실행/디버깅하면 됩니다.
- 시간/타임스탬프: `now_ns()` 유틸을 사용해 나노초 타임스탬프를 만듭니다. MQTT 페이로드는 JSON 구조(예: `{"sensor_type":"dht11","fields":{...},"timestamp_ns":...}`)를 사용합니다.
- 루프 제어: 전역 `stop_flag` + signal(SIGINT/SIGTERM) 처리로 안전 종료를 구현합니다.

## 런/디버그 요령 (핵심 명령)
- 라즈베리파이에서 실행 (GPIO/I2C 접근 필요)
  - 권한 요구: GPIO/I2C 접근을 위해 보통 `sudo`로 실행하거나 적절한 그룹 설정 필요.
  - 예: `sudo python3 sensor_final.py` 또는 `python3 data_collector.py`
- 로컬(하드웨어 없음) 개발
  - 하드웨어 관련 import가 실패할 수 있으므로, 에이전트는 모킹(mock) 또는 예외 분기를 권장합니다.

## 필수/권장 의존성
- 코드에서 사용되는 패키지(예시):
  - `adafruit_dht`, `board`, `busio`, `adafruit_ads1x15`, `smbus2`, `paho-mqtt`, `Adafruit-BMP`(옵션), `RPi.GPIO` (팬 제어)
- 설치 예시(라즈베리파이 / Python3):
  - `pip3 install adafruit-circuitpython-dht adafruit-circuitpython-ads1x15 smbus2 paho-mqtt Adafruit-BMP RPi.GPIO`
  - (주의) 일부 Adafruit 라이브러리는 `blinka`가 필요합니다.

## 통신·통합 포인트
- MQTT 브로커: 기본 `BROKER = "localhost", PORT = 1883`. 코드 내 토픽은 `factory/sensor/<type>` 형태로 고정되어 있습니다.
- I2C/ADC/IMU
  - ADS1115: I2C 주소 0x48, 채널 0/1을 진동/소리 입력으로 사용합니다.
  - MPU6050: WHO_AM_I 레지스터(0x75)를 통해 0x68/0x69 중 활성 주소를 판별합니다.
  - BMP085/BMP180: optional; 모듈 없이는 비활성화됩니다.
- GPIO 팬 제어: TB6612 드라이버(A 채널)를 사용합니다. 핀은 BCM 기준(예: AIN1=27, AIN2=22, PWMA=18, STBY=23).

## 예시 페이로드 (sensor_final/data_collector)
```
{
  "sensor_type": "dht11",
  "sensor_model": "DHT11",
  "fields": { "temperature_c": 24.1, "humidity_percent": 56.3 },
  "timestamp_ns": 1710000000000000000
}
```

## 에이전트 작동 시 우선순위 체크리스트
1. 하드웨어 접근 필요 여부 확인(RPi, I2C, GPIO). 로컬에서 작업 시 모킹 제안.
2. 의존성 설치 및 `BROKER`/토픽 확인.
3. 결과는 MQTT 토픽으로 퍼블리시됨 — 브로커가 로컬에 없다면 테스트용 브로커(`mosquitto`) 권장.
4. 변경 시 스크립트 상단의 상수를 수정하고, 안전 종료 패턴(stop_flag, signal)을 유지.

## 참고 파일
- 실행/종합: `sensor_final.py`, `data_collector.py`
- 드라이버/샘플: `DHT11.py`, `bmp180.py`, `bmp280.py`, `accel_gyro.py`(또는 `accel_gyro` 관련 파일)
- MQTT 샘플: `dht11_mqtt.py`, `IR_mqtt.py`

피드백: 이 파일로 충분한가요? 실행 환경(예: 사용하는 Raspbian 버전, 파이썬 버전, 테스트 브로커 유무)이나 추가로 문서화하길 원하는 패턴을 알려주시면 바로 반영하겠습니다.
