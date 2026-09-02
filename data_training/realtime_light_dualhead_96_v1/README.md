# Light DualHead 96 학습 준비 상태

이 폴더는 `REALTIME_RUST_CRACK_LIGHT_DUALHEAD_96_TRAINING_AGENT_HANDOFF.md`의 기술 사양을 구현한 독립 학습 작업공간입니다. 현재 사용 승인 범위는 **PHASE_A_DEVELOPMENT_ONLY**입니다. 결과는 항상 `ACCURACY_NOT_FINAL / NOT_FOR_UART`이며, Phase B sealed test 전에는 최종 정확도나 배포를 주장할 수 없습니다.

## 구현된 고정 계약

- 입력: RGB + ImageNet 정규화, FP32 `[1,3,240,1280]`, 패딩/letterbox 없음
- 학생: MobileNetV2 OS8 + LR-ASPP 96 + shared DW/PW + 녹 4채널/crack 1채널
- 출력: activation 없는 `multitask_logits [1,5,240,1280]`
- crack loss/KD 유효 영역: `rows 112:240`만
- 녹 교사와 HrSeg 교사의 서로 다른 전처리 및 FP32 cache
- 4단계 동결, 모든 BatchNorm running statistics 동결, 3 seeds `17/29/43`
- manifest SHA/원본/print/placement/session cross-split 누수 감사
- FP32 ONNX opset 17 export, PyTorch↔ORT parity, complexity 보고서
- GPU용 rust argmax/crack 후보 수 prefilter wrapper(연결성분·제어는 포함하지 않음)
- UART/actuator 사용은 모든 산출물에서 금지

## 현재 승인 상태

1. 녹 및 HrSeg 교사는 계획서의 정확한 SHA/크기/I/O 계약을 통과해야 합니다.
2. 녹 rubric은 사용자가 승인한 `Normal=0, Fair=1, Poor=2, Severe=3, ambiguous=255`로 고정됩니다.
3. Phase A는 공개 CrackSeg9k V4와 기존 정상 레일 음성 데이터 학습, ONNX export, 속도 시험까지 승인되었습니다.
4. 독립 sealed manifest commitment와 실제 인쇄물 촬영 평가는 Phase B로 미뤄졌습니다.
5. 최종 정확도·UART·actuator 승인은 계속 금지됩니다.

`*.REQUIRED.*` 파일은 형식 예시일 뿐 승인 자료가 아닙니다. 이름만 바꿔 사용하면 안 됩니다.

## 지금 실행해 볼 수 있는 명령

```powershell
$ROOT = $env:RAIL_ROBOT_ROOT

# 현재 차단 사유를 보고서로 저장(실패 종료하지 않음)
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$ROOT\data_training\realtime_light_dualhead_96_v1\run_light_dualhead96.ps1" `
  -Task audit -AllowBlocked

# RTX 5070 Ti 전체 shape forward/backward 확인
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$ROOT\data_training\realtime_light_dualhead_96_v1\run_light_dualhead96.ps1" `
  -Task preflight

# 구조/loss/교사 계약 단위 테스트
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$ROOT\data_training\realtime_light_dualhead_96_v1\run_light_dualhead96.ps1" `
  -Task test
```

## Phase A 실행 순서

1. `-Task prepare`로 공개 crack/정상 음성 Phase A manifest를 만듭니다.
2. `-Task audit`으로 교사·rubric·데이터 SHA 및 split 누수를 확인합니다.
3. `-Task cache-smoke`로 두 샘플의 실제 GPU 교사 추론을 확인합니다.
4. `-Task phase-a-smoke`로 공개 crack+정상 레일 실제 batch의 forward/backward를 확인합니다.
5. `-Task cache-train`, `-Task cache-validation`으로 전체 교사 cache를 만듭니다.
6. 각 seed를 `-Task train -Seed 17 -Stage all` 형태로 실행합니다.
7. Phase A validation은 개발 지표만 기록하고 `ACCURACY_NOT_FINAL`로 유지합니다.
8. 승인된 후보를 FP32 ONNX로 export하고 로컬/Jetson 속도를 시험합니다.

전체 teacher cache는 약 19.5 GiB이며 OneDrive 동기화 대상이 아닌 `C:\rail_robot_cache\realtime_light_dualhead_96_v1`에 저장됩니다.

장시간 학습 명령은 audit가 `ready_for_training: true`가 되기 전에는 실행되지 않습니다. Phase B에서는 실제 인쇄물 데이터와 독립 sealed commitment가 별도로 필요합니다.
