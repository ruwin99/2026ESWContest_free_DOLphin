# 대회 발표·공개용 학습 코드 가이드

이 문서는 심사위원에게 “어떤 코드를 실제로 사용했고, 어떤 코드는 실험 단계인지”를 명확히 설명하기 위한 기준 문서다. 성능 수치는 저장된 평가 결과로 확인되는 값만 사용한다.

## 1. 먼저 보여줄 코드 흐름

### 이물질 탐지: YOLO26n hard-negative 미세조정

대표 폴더: `YOLO26_hard_negative_1280x720`

1. `tools/prepare_hardneg.py`: 이미지·라벨 형식, SHA 중복, 촬영 그룹을 감사한다.
2. `tools/prepare_all_hardneg.py`: 정상 배경을 빈 라벨 hard negative로 추가하되 검증·최종 시험 세트를 분리한다.
3. `tools/preflight.py`: CUDA, 패키지 버전, baseline SHA, 클래스 계약을 학습 전에 확인한다.
4. `tools/train_cls_output_safe.py`: 분류 출력층만 학습하고 박스 회귀·BatchNorm 통계를 고정한다.
5. `tools/evaluate_hardneg_dev.py`: 개발용 정상 영상에서 오탐을 비교한다.
6. `tools/evaluate_locked_final.py`: 후보와 임계값을 고정한 뒤 봉인 최종 음성 세트를 한 번만 평가한다.
7. `tools/export_final_onnx.py`: 고정 입력 ONNX로 변환하고 PyTorch–ONNX Runtime 동등성을 확인한다.

발표 핵심: “정상 레일을 학습 데이터에 추가해 오탐을 줄였고, 최종 시험 영상을 학습·튜닝에서 분리했다.”

### 녹 판단: 캡처 teacher hard-negative 후보

대표 폴더: `capture_rust_teacher_hardneg_v1`

- `tools/prepare_manifests.py`, `tools/audit.py`: 공개 녹 데이터와 정상 레일 음성 데이터의 출처·split·SHA를 확인한다.
- `tools/train.py`: DeepLabV3+ ResNet-101 teacher의 제한된 부분을 미세조정한다.
- `tools/evaluate.py`, `tools/evaluate_sealed.py`: 검증과 봉인 평가를 분리한다.
- `tools/export_onnx.py`: raw logits ONNX와 입출력 계약을 기록한다.

정확한 상태: v6 Stage A는 수동 `--capture-test`에만 시험 통합됐지만 독립 positive sealed test가 없어 정상 UART 임무 기준으로는 `ACCURACY_NOT_FINAL / NOT_FOR_UART / NOT_DEPLOYED`다.

### 크랙·경량 모델 개발

대표 폴더: `realtime_multitask_5ch_hrseg_v1`, `realtime_light_dualhead_96_v1`

- 공식 HrSegNet-B32 체크포인트를 변환하고 Paddle–ONNX 수치 동등성을 확인했다.
- CrackSeg9k 공개 데이터로 크랙 bootstrap 학습 경로와 누수 감사 도구를 구현했다.
- MobileNetV2 기반 경량 듀얼헤드의 학습·평가·ONNX 변환 코드를 구현했다.
- 실제 카메라 크랙 양성 GT와 독립 sealed test가 부족하므로 경량 듀얼헤드는 개발 단계이며 최종 배포 모델로 설명하지 않는다.

중요: HrSegNet-B32 자체를 팀이 처음부터 학습했다고 말하지 않는다. 공식 모델을 출처와 함께 변환·검증해 사용했고, 팀이 수행한 부분은 카메라 입력 계약, 후처리, 경량 학생 학습 경로, GPU 실행 최적화다.

## 2. 모델 상태표

| 모델 경로 | 코드 근거가 확인되는 상태 | 발표에서 사용 가능한 표현 | 금지 표현 |
|---|---|---|---|
| YOLO26n hard-negative | 실제 최종 학습·고정 임계값 평가·ONNX 변환 기록 존재 | 자체 데이터 추가 미세조정, 최종 음성 세트 분리 | 모든 환경 정확도 검증 완료 |
| 녹 ResNet101–DeepLabV3+ v6 | 시연 정상 환경 hard-negative 후보 | 시연 후보 선정, 정상 배경 오탐 개선 | 최종 현장 모델, UART 배포 완료 |
| HrSegNet-B32 | 공식 checkpoint 변환·동등성 확인 | 공식 모델 기반 크랙 영역 분석 | 자체 원천 학습 모델 |
| 5채널 MobileNetV2 학생 | 공개 데이터 bootstrap 및 demo 후보 기록 | 경량화 학습·검증 경로 구현 | 실제 레일 크랙 정확도 확정 |
| Light DualHead 96 | Phase A 개발 코드·시험 계약 | 후속 경량화 실험 | 최종 제어 모델, KD 완료 확정 |

## 3. 심사위원에게 보여줄 파일 7개

1. `README_KR.md`: 전체 모델 변천사와 현재 상태
2. `YOLO26_hard_negative_1280x720/TRAINING_RECORD.md`: 실제 학습 설정과 결과 해시
3. `YOLO26_hard_negative_1280x720/tools/prepare_hardneg.py`: 데이터 누수 방지
4. `YOLO26_hard_negative_1280x720/tools/train_cls_output_safe.py`: 제한적 미세조정
5. `capture_rust_teacher_hardneg_v1/tools/train.py`: 녹 teacher 학습
6. `realtime_multitask_5ch_hrseg_v1/tools/convert_hrseg_teacher.py`: 공식 크랙 모델 변환·parity
7. `realtime_light_dualhead_96_v1/tests/test_split_leakage.py`: split 누수 검사

## 4. 재현 절차

1. Python·CUDA·PyTorch·Ultralytics 버전을 각 프로젝트 requirements와 environment 기록에 맞춘다.
2. `RAIL_ROBOT_ROOT`를 설정한다.
3. 외부 데이터와 checkpoint를 공식 출처에서 직접 받아 SHA를 확인한다.
4. 데이터 감사와 preflight를 먼저 실행한다.
5. 고정 seed와 config로 학습한다.
6. validation으로 checkpoint와 후처리 임계값을 선택한다.
7. 선택이 끝난 뒤에만 locked test를 평가한다.
8. ONNX export 후 PyTorch–ONNX Runtime parity를 검사한다.

데이터·가중치·로그가 저장소에서 제외되어 있으므로 완전 재학습에는 별도 다운로드가 필요하다. 각 결과 JSON과 checkpoint SHA를 함께 공개하면 코드와 발표 수치의 추적성이 높아진다.

## 5. 발표 슬라이드에 넣을 설명

> 이물질 모델은 정상 레일 영상을 hard negative로 추가해 오탐을 줄였으며, 검증과 최종 시험 영상을 분리하였다. 녹·크랙은 픽셀 단위 세그멘테이션을 사용하고, 공개 모델과 데이터의 출처를 기록한 뒤 입력 형식과 ONNX 변환 결과를 검증하였다. 경량 모델은 별도 개발 단계로 관리하며 독립 시험이 끝나지 않은 결과는 현장 정확도로 주장하지 않는다.

## 6. GitHub 공개 전 확인

- `PUBLIC_RELEASE_MANIFEST.txt`의 포함·제외 기준을 적용한다.
- `THIRD_PARTY_NOTICES.md`의 `확인 필요` 항목을 해결한다.
- API 키, 사용자 이름이 포함된 절대경로, 데이터 원본, 재배포 제한 checkpoint를 제외한다.
- 실제 보고서 수치와 연결되는 평가 JSON 및 SHA만 선별해 추가한다.
- `verify_public_release.py`를 실행한다.
