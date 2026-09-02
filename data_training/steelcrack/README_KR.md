# Steelcrack + BGCrack 로컬 학습 사용법

확인일: 2026-08-09  
프로젝트: `<repo-root>`

## 준비 결과

공식 Steelcrack 데이터셋과 공식 BGCrack 코드를 내려받아 RTX 5070 Ti의 Windows 네이티브 단일 GPU 환경에서 학습할 수 있도록 준비했다.

공식 학습 진입점은 GPU가 한 장이어도 Windows에서 지원되지 않는 NCCL/DDP를 강제로 초기화한다. 현재 PC에는 WSL2도 설치되어 있지 않으므로, 공식 모델 구조·전처리·다섯 가지 손실식은 유지하면서 NCCL/DDP만 제거한 로컬 학습 래퍼를 사용한다. 공식 clone은 수정하지 않았다.

현재 검증 상태:

- 공식 데이터 ZIP 크기와 SHA-256 일치
- Train 3,300 / Validation 525 / Test 530 공식 분할 유지
- 전체 13,065개 이미지·mask·edge 파일 검사 완료
- 모든 파일 RGB 또는 이진 label, 512×512
- 손상·대응 누락·빈 mask·split 간 완전 중복 0개
- Python 3.11.14, PyTorch 2.12.1+cu130
- RTX 5070 Ti, CUDA 13.0, `sm_120` 확인
- BGCrack FP32 배치 9 실제 forward/backward와 Validation 추론 통과
- FP32 스모크 최대 GPU 메모리 약 10.29 GiB
- 공식 모델의 FFT가 FP16 `ComplexHalf`에서 불안정하므로 학습·평가는 FP32가 기본

데이터 출처는 [Civil-dataset 공식 저장소](https://github.com/hzlbbfrog/Civil-dataset), 모델 코드는 [BGCrack 공식 저장소](https://github.com/hzlbbfrog/BGCrack)이며 정확한 해시와 commit은 `asset_manifest.json`에 기록되어 있다.

이 공개 제출본에는 원본 데이터, checkpoint와 제3자 BGCrack 소스 복사본을 넣지 않았다. 재현할 때는 `asset_manifest.json`에 기록된 출처에서 Steelcrack 데이터를 받아 `data/Steelcrack`에 두고, 다음처럼 고정 commit을 준비한다.

```powershell
git clone https://github.com/hzlbbfrog/BGCrack.git official_bgcrack
git -C official_bgcrack checkout 207929839318267e7f47c91f1b177f321777f732
```

그 후 `setup_steelcrack.ps1`로 전용 환경을 만든다. 다운로드한 자산의 라이선스와 SHA-256은 학습 전에 직접 대조해야 한다.

## 바로 학습하는 명령

PowerShell의 현재 위치가 `C:\Windows\System32`여도 실행되도록 절대 경로를 사용한다. 아래 명령은 총 70 epoch, 공식 learning rate 0.006, batch 9, seed 42 설정이다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\steelcrack\run_steelcrack.ps1" -Task train -Epochs 70 -Batch 9 -Workers 4 -LearningRate 0.006 -Seed 42 -RunName bgcrack-seed42
```

최초 실행 환경에서는 먼저 다음 명령으로 전용 환경을 준비한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\steelcrack\setup_steelcrack.ps1"
```

## 학습 진행과 결과 파일

학습 중 터미널에 epoch, batch, loss가 출력된다. 공식 정책과 동일하게 Validation은 전체 epoch의 60% 지점부터 시작하므로 70 epoch 설정에서는 Epoch 42부터 검증한다. 따라서 `best.pth`가 초반에 없는 것은 정상이다.

결과 폴더:

```text
<repo-root>\outputs\training\steelcrack\bgcrack-seed42\
├─ metrics.csv
├─ run_config.json
├─ run_summary.json
└─ weights\
   ├─ best.pth
   └─ last_resume.pt
```

- `best.pth`: Validation soft-Dice가 가장 높았던 최종 선택 모델. 추론·평가·향후 변환에 사용한다.
- `last_resume.pt`: 마지막 epoch의 모델·optimizer·난수 상태가 든 학습 재개 전용 파일이다.

## 중단과 재개

학습을 중단할 때는 해당 PowerShell 창에서 `Ctrl+C`를 한 번 누른다. 저장은 epoch가 끝날 때마다 이루어지므로 현재 epoch의 중간 진행분은 다시 실행된다.

재개 명령:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\steelcrack\run_steelcrack.ps1" -Task train -Epochs 70 -Batch 9 -Workers 4 -LearningRate 0.006 -Seed 42 -Resume "<repo-root>\outputs\training\steelcrack\bgcrack-seed42\weights\last_resume.pt"
```

`-Epochs 70`은 추가 70회가 아니라 총 목표 70 epoch이다. 기존 결과 폴더에 `-Resume` 없이 같은 `RunName`으로 다시 시작하면 덮어쓰기 방지 오류가 발생한다.

## 검증과 최종 Test 평가

학습이 끝난 뒤 `best.pth`를 Validation에서 다시 평가할 수 있다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\steelcrack\run_steelcrack.ps1" -Task evaluate-val -Checkpoint "<repo-root>\outputs\training\steelcrack\bgcrack-seed42\weights\best.pth"
```

Test는 학습 조건이나 epoch를 고르는 데 사용하지 않는다. Validation으로 모델 선택을 끝낸 뒤 마지막에 한 번만 실행한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\steelcrack\run_steelcrack.ps1" -Task evaluate-test -Checkpoint "<repo-root>\outputs\training\steelcrack\bgcrack-seed42\weights\best.pth"
```

동일 run의 Test 결과가 이미 있으면 재실행을 차단한다. 정말 반복할 의도가 있을 때만 `-AllowTestRerun`을 추가한다.

## 점검 명령

GPU 환경 확인:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\steelcrack\run_steelcrack.ps1" -Task verify
```

데이터 전수 감사 재실행:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\steelcrack\run_steelcrack.ps1" -Task audit
```

학습 1 batch 스모크 테스트:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\steelcrack\run_steelcrack.ps1" -Task smoke -Batch 9 -Workers 4
```

## 경고의 의미

이 프로젝트는 FP32가 기본이며 일반 학습 명령에 별도 정밀도 옵션을 붙일 필요가 없다. 공식 모델의 FFT 구간을 AMP/FP16으로 실행하면 `ComplexHalf support is experimental` 경고와 함께 장시간 학습 중 NaN이 발생할 수 있으므로 `-Amp`는 사용하지 않는다. `nn.functional.upsample is deprecated`는 공식 코드의 구형 API 안내이며 현재 결과를 중단시키는 오류가 아니다.

OOM이 발생하면 다른 옵션은 그대로 두고 `-Batch 4`, 그래도 부족하면 `-Batch 2`로 낮춘다. 현재 PC에서는 FP32 batch 9가 약 10.29 GiB로 통과했다.

## 배포 시 주의

`best.pth`는 공식 BGCrack PyTorch 가중치다. 기존 YOLO `.pt`나 TensorRT `.plan`과 바꿔 끼울 수 없고, 확장자만 변경해서도 안 된다. BGCrack은 균열 확률·edge·gradient 세 출력을 반환하며 최종 균열 출력도 이미 sigmoid가 적용된 확률이다. 아래 전용 exporter는 Jetson 배포에 필요한 균열 확률 출력만 남기고 PyTorch와 ONNX의 수치 동등성을 검사한다.

## ONNX 내보내기

Jetson 배포용 ONNX는 균열 확률맵 하나만 출력한다. 공식 모델의 복소 FFT 연산은 같은 결과를 내는 실수 `MatMul` 연산으로 변환되므로 ONNX 그래프에는 `DFT` 연산이 없다. 입력과 출력 크기는 고정이다.

- 입력 `images`: FP32 RGB, `1x3x512x512`, 픽셀을 `[-1, 1]`로 정규화
- 출력 `crack_probability`: FP32, `1x1x512x512`, 값 범위 `0..1`
- 이진 균열 마스크 기준 임계값: `0.5`

```powershell
$ROOT = "<repo-root>"
$BEST = "$ROOT\outputs\training\steelcrack\bgcrack-seed42\weights\best.pth"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ROOT\data_training\steelcrack\run_steelcrack.ps1" -Task export-onnx -Checkpoint $BEST
```

기본 출력은 `output\models\steelcrack\bgcrack-steelcrack-512-fp32.onnx`이다. 함께 생성되는 `.metadata.json`에는 전처리와 검증 결과가, `.sha256.txt`에는 파일 무결성 확인용 해시가 들어 있다. TensorRT 엔진은 RTX 학습 PC에서 만들지 말고 대상 Jetson에서 해당 장치의 JetPack/TensorRT 버전으로 생성한다.
