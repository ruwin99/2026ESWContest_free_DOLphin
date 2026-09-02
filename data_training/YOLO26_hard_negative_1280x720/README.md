# YOLO26n 이물질 hard-negative 학습

## 목적

정상 레일과 주변 구조물을 이물질로 잘못 검출하는 문제를 줄이기 위해 기존 YOLO26n 이물질 모델의 분류 출력을 제한적으로 미세조정한 프로젝트입니다.

## 데이터와 모델

- 모델: YOLO26n, 단일 클래스 `obstacle`
- 카메라 원본: 1280×720
- 학습 입력: 긴 변 1280, 일반적인 16:9 직사각형 배치에서는 stride 정렬된 1280×736
- 데이터: 기존 양성 자료와 여러 촬영 세션의 정상 hard-negative
- 분리 원칙: 연속 프레임을 촬영 그룹으로 묶고 개발 세트와 최종 test를 분리
- 미세조정 범위: 분류 출력만 학습하고 backbone, BatchNorm 통계와 box 회귀는 고정

기준 데이터는 Roboflow 프로젝트 `-ohs3h/2-iemaw/1`에서 출발했고 학습기는 [Ultralytics](https://github.com/ultralytics/ultralytics)를 사용했습니다.

## 학습·평가 흐름

정상 사진의 빈 라벨과 중복을 감사하고 촬영 그룹 단위로 분리합니다. 양성 validation과 hard-negative 개발 세트에서 후보를 선택한 뒤, 학습과 임계값 조정에 사용하지 않은 최종 test를 한 번 평가합니다. 마지막으로 고정 입력 ONNX와 원 모델의 출력을 비교합니다.

## 산출물

고정 설정, 학습 checkpoint, validation·locked test 보고서와 ONNX를 생성합니다. 실제 학습 이력과 자산 해시는 [TRAINING_RECORD.md](TRAINING_RECORD.md)에 보존했습니다.

## 현재 상태와 한계

현재 상단 이물질 런타임 모델의 hard-negative 계보입니다. 고정 카메라 시연 환경을 중심으로 개선한 후보이므로 새로운 레일, 조명과 배경에서 동일한 오탐 억제를 보장하지 않습니다. 모델 바이너리와 촬영 데이터는 공개본에 포함하지 않습니다.
