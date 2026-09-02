# YOLO26n 상단 이물질 기준 모델

## 목적

TOP 카메라 영상에서 레일 위 이물질 후보를 박스로 검출하기 위한 단일 클래스 YOLO 기준 모델을 준비합니다.

## 데이터와 모델

- 데이터 출처: Roboflow 프로젝트 `-ohs3h/2-iemaw/1`
- 모델: YOLO26n
- 클래스: `obstacle`
- 기준 학습 입력: 640×640
- 출력: 위치, confidence와 class를 포함한 검출 boxes
- 학습기: [Ultralytics](https://github.com/ultralytics/ultralytics)

API 자격증명은 데이터 다운로드에만 사용하며 코드·설정·로그에 저장하지 않습니다.

## 학습·평가 흐름

Roboflow 원본 YAML을 보존하고 로컬 데이터 위치만 분리합니다. 이미지·라벨 대응과 split을 감사한 뒤 GPU smoke test, 학습과 validation을 수행합니다. validation 최적 checkpoint를 ONNX로 변환하고 대상 Jetson에서 TensorRT I/O와 실제 카메라 동작을 확인합니다.

## 산출물

학습 checkpoint, validation 결과, 데이터 감사 기록과 단일 클래스 ONNX를 생성합니다.

## 현재 상태와 한계

이 폴더는 최초 기준 모델 계보입니다. 현재 상단 이물질 런타임은 정상 배경 오탐을 추가 학습한 [hard-negative 후속 모델](../YOLO26_hard_negative_1280x720/README.md)을 사용합니다. 고정 시연 환경 밖의 조명·레일·배경 정확도는 별도 검증이 필요합니다.
