# AI 학습·평가 코드

## 목적

레일 표면의 녹·크랙 후보와 상단 이물질을 분석하는 모델의 데이터 준비, 학습, 평가와 ONNX 변환 과정을 보존합니다. 원본 데이터, 촬영 사진, 가중치와 실행 결과는 공개본에 포함하지 않습니다.

## 모델 계보

| 분석 대상 | 프로젝트 | 현재 상태 |
| --- | --- | --- |
| 실시간 녹 | `vt_kd/`, `realtime_w1280/` | 현재 런타임 계보 |
| 캡처 녹 | `capture_1280x720/` | 정상 임무 기준 계보 |
| 캡처 녹 hard-negative | `capture_rust_teacher_hardneg_v1/` | 수동 캡처 시험 후보 |
| 상단 이물질 | `yolo26_obstacle/`, `YOLO26_hard_negative_1280x720/` | 현재 이물질 계보 |
| BGCrack | `steelcrack/`, `realtime_w1280/` | 과거 크랙 계보 |
| HrSegNet 교사 변환 | `realtime_multitask_5ch_hrseg_v1/` | 현재 실시간 크랙 런타임의 변환 계보 |
| 5채널·경량 듀얼헤드 학생 | `realtime_multitask_5ch_hrseg_v1/`, `realtime_light_dualhead_96_v1/` | 연구 코드, 미배포 |

배포 상태의 기준은 [모델 상태표](../docs/MODEL_STATUS.md)이며 프로젝트별 입력·출력과 출처는 각 폴더 README에 정리했습니다.

## 학습·평가 흐름

1. [제3자 고지](THIRD_PARTY_NOTICES.md)의 공식 출처와 이용조건을 확인합니다.
2. 시편·촬영 세션·연속 프레임 단위로 개발, 검증, 잠금 시험을 분리합니다.
3. 데이터와 라벨을 감사한 뒤 고정된 config와 seed로 학습합니다.
4. validation에서 모델과 후처리를 선택하고, 선택이 끝난 뒤 locked/sealed test를 평가합니다.
5. 원 프레임워크와 ONNX Runtime 출력을 비교한 뒤 대상 Jetson에서 TensorRT plan을 검증합니다.

## 산출물

학습 checkpoint, 평가 JSON/CSV, 데이터·환경 감사 기록, ONNX와 무결성 해시를 생성합니다. 공개 저장소에는 소스·설정·고지만 남기며 바이너리와 개인정보·비밀정보는 제외합니다.

## 현재 상태와 한계

현재 런타임 모델과 연구 후보를 구분한 공개 스냅샷입니다. hard negative는 관측한 정상 배경의 오탐을 줄일 수 있지만 모든 재질·색·조명에 대한 일반화를 보장하지 않습니다. 세부 제출 범위는 [대회 공개용 가이드](COMPETITION_CODE_GUIDE.md)를 따릅니다.
