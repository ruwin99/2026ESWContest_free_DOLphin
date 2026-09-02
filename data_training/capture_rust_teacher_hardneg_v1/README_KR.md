# 캡처 녹 Teacher Hard-negative 파인튜닝 준비본

이 폴더는 기존 캡처 녹 teacher를 보존한 채 별도 후보를 만드는 작업 공간이다. 생성된 v6 후보는 이후 수동 `--capture-test`에만 시험 통합됐으며 정상 UART 임무 기준으로는 계속 `candidate / NOT_FOR_UART / NOT_DEPLOYED`다. 이 학습 폴더 자체에는 Jetson 배포·TensorRT·UART 작업을 포함하지 않는다.

## 고정된 모델 계약

- DeepLabV3+ ResNet-101, output stride 8, 4 classes, 58,749,604 parameters
- 입력: OpenCV BGR FP32 `0..255`, 정적 `1x3x720x1280`
- 출력: activation 전 raw logits FP32 `1x4x720x1280`
- 클래스: `Good, Fair, Poor, Severe`; ignore index `255`
- 원 학습기: Adam `1e-4`; Stage A classifier/decoder `1e-5`; 조건부 Stage B layer4 `1e-6`
- 모든 BatchNorm affine 및 running statistics는 두 단계 모두 고정

## 현재 상태

구조와 strict checkpoint load는 검증되었다. 사용자가 2026-08-20에 후보 사진 1,462장 전체를 녹이 없는 `approved_good`으로 승인해 class 0 zero-mask를 생성했다. 현재 manifest는 train 1,557장(VT positive 316 + hard-negative 1,241), validation 301장(VT positive 80 + hard-negative 221)이다. 독립 sampled overlay QA와 sealed test는 아직 없으므로 `lock`, 실제 data smoke, 본 학습은 fail-closed로 중단된다.

이후 사용자가 대표 overlay 8개 group과 같은 시연 도메인의 세션 간 유사성을 단독 승인했다. 독립 두 번째 검토자는 없었다는 사실을 QA metadata에 기록했다. 현재 유일한 학습 차단 항목은 새로운 sealed/locked holdout 사진과 commitment가 없다는 점이다.

기존 캡처 녹 ONNX 본체가 원래 출력 폴더에서 누락되어 있었으므로 기존 경로를 덮어쓰지 않고 `baselines/corrosion-capture-r101-os8-w1280-h720-fp32.onnx`로 재생성했다. SHA-256은 기존 metadata에 기록된 `38db0d0afe7ef1e808e77858dabd3a75dd5ecd6df40bd2bcfcda98f728bab9e8`과 정확히 일치한다.

## 순서

```powershell
$ROOT = $env:RAIL_ROBOT_ROOT
$RUNNER = "$ROOT\data_training\capture_rust_teacher_hardneg_v1\run_capture_rust_hardneg.ps1"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task inventory
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task smoke-structure
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task audit
```

`manifests/hard_negative_review.csv`에서 각 사진을 실제로 검토한다. 전체가 녹이 없는 Good이고 가려져 추정할 영역도 없을 때만 `label_status=approved_good`으로 바꾼다. 실제 녹·불확실 영역이 있으면 수동 class-index mask(`0/1/2/3/255`)와 valid mask를 만든 뒤 `approved_masked`로 기록한다. `split`, 촬영 metadata, `group_id`, 서로 다른 `labeler/reviewer`도 채워야 한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task build-manifests
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task audit
```

독립 사진과 mask로 `sealed_test.csv`를 만든 후 `sealed_test_commitment.yaml`의 TBD를 채운다. 감사 통과 후에만 manifest를 잠근다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task lock
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task smoke-data -Seed 42
```

그 다음 Stage A를 seed 42/43/44로 각각 새 run에서 실행한다. Stage B는 Stage A validation gate를 통과한 `best.pt`에만 별도 run으로 실행한다. 같은 run을 다른 설정으로 단순 resume하지 않는다.

## 금지 사항

- 폴더 이름만 보고 zero mask를 자동 승인하지 않는다.
- 연속 프레임을 무작위로 train/validation/sealed에 흩뜨리지 않는다.
- sealed test를 모델 선택·threshold 조정에 사용하지 않는다.
- 기존 캡처 ONNX 또는 teacher state_dict를 덮어쓰지 않는다.
- validation 통과 전 ONNX export를 하지 않는다.
