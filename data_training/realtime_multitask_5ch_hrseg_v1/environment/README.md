# 학습 환경 기록

## 목적

실시간 5채널 학생 모델의 학습 환경을 다른 실행 결과와 구분하고, 패키지·CUDA·GPU 차이로 인한 재현성 문제를 추적합니다.

## 환경 범위

- 격리된 Python 3.11 환경
- PyTorch, ONNX와 ONNX Runtime
- Paddle 기반 HrSegNet 교사 변환 도구
- CUDA 런타임과 GPU 정보

## 기록 흐름과 산출물

환경을 구성한 뒤 해석된 패키지 버전과 GPU 정보를 기록합니다. `requirements.txt`는 설치 범위이고, 실제 실행에 사용한 `pip-freeze.txt`와 `environment.json`이 학습 run의 재현 기록입니다. Paddle 교사 변환 환경과 PyTorch 학생 학습 환경은 별도 snapshot으로 보존합니다.

## 현재 상태와 한계

저장된 환경 정보는 당시 실행 조건을 설명하지만 다른 GPU나 운영체제에서 동일 결과를 보장하지 않습니다. 환경 기록만으로 데이터 승인, 정확도 검증 또는 모델 배포가 승인되는 것은 아닙니다.
