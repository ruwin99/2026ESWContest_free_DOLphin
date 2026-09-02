# MobileNetV2 + DeepLabV3+ 부식 모델 학습 실행 안내

이 폴더는 `MOBILENETV2_DEEPLABV3PLUS_STUDENT_TRAINING_GUIDE.md`의 구현본이다.
기존 YOLO 가상환경과 분리되어 있으므로 기존 오염·레일 결함 모델에는 영향을 주지 않는다.

## 현재 준비된 것

- Virginia Tech 공식 데이터 ZIP 및 압축 해제본
- Train 396쌍 / 잠긴 Test 44쌍 무결성 검사
- seed 42 고정 분할: train 316 / validation 80
- Virginia Tech 공식 교사 모델 ZIP과 체크포인트 4개
- 공식 코드 commit `f3d098b09784ac7e78b160906952e7bc79940fb1`
- 최신 torchvision 호환 import 수정
- PyTorch 공식 MobileNetV2 ImageNet 가중치 로컬 캐시
- 데이터 감사, 교사 변환, 지도학습/KD, 평가, ONNX 내보내기 스크립트

## 1. 전용 환경 만들기

프로젝트 루트가 어디로 옮겨져도 `$ROOT` 한 줄만 바꾼다.

```powershell
$ROOT = "<repo-root>"
& "$ROOT\data_training\vt_kd\setup_vt_kd.ps1"
```

정상 환경 기준은 Python 3.12, PyTorch 2.12.1, torchvision 0.27.1,
CUDA 13.0 wheel, RTX 5070 Ti capability 12.0과 `sm_120`이다.

## 2. 데이터 분할 다시 검증하기

이미 생성되어 있지만, 다른 PC로 옮긴 뒤에는 다시 실행한다. 같은 원본과 seed라면
CSV SHA-256도 같아야 한다.

```powershell
& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" -Task audit
```

예상 결과:

```text
source Train=396, train=316, val=80, locked Test=44
split SHA-256=49cc832f1f7894a16fec67dd735d24ce0249cbd94361cc27730788fcd04aae7a
```

## 3. 교사 체크포인트 변환

공식 교사 `.pt`는 구형 전체 Python 객체 pickle이다. `weights_only=False`로 열면
코드가 실행될 수 있으므로, 공식 ZIP 및 체크포인트 SHA-256이 맞는 경우에만 한 번 변환한다.

검증된 weighted-CE 체크포인트 SHA-256:

```text
9110b28a7d027679076f831d2987d34be3be47883a9d724456810452a32e1dd5
```

위 위험을 이해하고 파일 출처·해시를 직접 확인한 뒤 실행한다.

```powershell
& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" `
  -Task convert-teacher `
  -AcknowledgeLegacyPickleRisk
```

변환 후에는 `teacher_converted`의 안전한 state-dict bundle만 사용한다.

## 4. smoke test

먼저 지도학습 한 batch를 확인한다.

```powershell
& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" `
  -Task smoke-supervised `
  -Batch 8 `
  -Workers 0
```

교사 변환이 끝났으면 KD도 한 batch 확인한다.

```powershell
& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" `
  -Task smoke-kd `
  -Batch 8 `
  -Workers 0
```

확인할 값은 학생 파라미터 5,221,348개, `[N,4,512,512]` 로짓, target 0~3,
finite CE/KD/total loss, 교사 gradient 없음이다.

## 5. 2 epoch 실행 확인

```powershell
& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" `
  -Task train-supervised -Epochs 2 -Workers 0 -RunName smoke-supervised

& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" `
  -Task train-kd -Epochs 2 -Workers 0 -RunName smoke-kd
```

`outputs/training/vt_kd/<run-name>/`에 `metrics.csv`, TensorBoard 로그,
`weights/best.pt`, `weights/last.pt`가 생성된다.

## 6. 본 학습

지도학습 기준선을 먼저 실행한다.

```powershell
& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" `
  -Task train-supervised `
  -Seed 42 `
  -RunName mnv2-os8-supervised-seed42
```

그 다음 같은 분할·시드·학습 예산으로 KD 학생을 실행한다.

```powershell
& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" `
  -Task train-kd `
  -Seed 42 `
  -RunName mnv2-os8-kd-t4-a05-seed42
```

중단은 학습 PowerShell 창에서 `Ctrl+C`를 한 번 누르고 저장이 끝날 때까지 기다린다.
재개 예시:

```powershell
& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" `
  -Task train-kd `
  -Resume "$ROOT\outputs\training\vt_kd\mnv2-os8-kd-t4-a05-seed42\weights\last.pt"
```

## 7. 평가와 ONNX

Validation은 필요할 때 반복 평가할 수 있다.

```powershell
$BEST = "$ROOT\outputs\training\vt_kd\mnv2-os8-kd-t4-a05-seed42\weights\best.pt"
& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" -Task evaluate-val -Checkpoint $BEST
```

공개 Test 44장은 구조·하이퍼파라미터·세 시드를 모두 고정한 뒤 한 번만 실행한다.
같은 출력 폴더의 Test 재실행은 기본적으로 차단된다.

```powershell
& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" -Task evaluate-test -Checkpoint $BEST
```

ONNX 내보내기:

```powershell
& "$ROOT\data_training\vt_kd\run_vt_kd.ps1" -Task export-onnx -Checkpoint $BEST
```

TensorRT 엔진은 Windows에서 만들지 않고 ONNX를 Jetson으로 옮긴 뒤 실제 Jetson에서 만든다.

## 중요한 해석 제한

공개 교사는 원래 Train 전체와 공개 Test를 모델 선택에 사용했을 가능성이 있다.
이 교사로 증류한 결과는 `teacher-contaminated transfer benchmark`이며 독립적인 일반화
성능으로 주장하지 않는다. 또한 이 모델은 표면 부식 상태를 분할할 뿐 강재 두께,
구조 강도, 잔존 수명 또는 운전 가능 여부를 판정하지 않는다.
