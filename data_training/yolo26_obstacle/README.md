# YOLO26n 이물질 탐지 로컬 학습

이 폴더는 Roboflow `-ohs3h/2-iemaw/1`의 YOLO26 객체 탐지 데이터를 RTX 5070 Ti에서 학습하기 위한 실행 환경이다.

## 현재 PC 기준 기본값

- GPU: RTX 5070 Ti 16GB
- CPU: Ryzen 7 7800X3D, 8코어/16스레드
- RAM: 32GB
- 입력: 640×640
- 기본 물리 배치: 16
- DataLoader workers: 8
- AMP: 활성화
- 데이터/이미지 디스크 캐시: `C:\rail_robot_cache\yolo26_obstacle`
- 결과: `outputs\training\yolo26_obstacle`

Roboflow가 내보낸 원본 `data.yaml`은 보존한다. 실행기는 실제 폴더 구조를 확인한 뒤 경로가 수정된 `data.local.yaml`을 자동 생성하고 학습에 사용한다.

Batch 16을 기본으로 사용한다. 데이터가 준비되면 `benchmark`가 Batch 8과 16을 각각 2%/1 epoch로 시험한다. CUDA OOM이 발생하면 Batch 8로 낮춘다.

Windows에서 시스템 RAM 사용률이 높으면 `cache=ram`을 사용하지 않는다. 데이터는 `cache=disk`로 유지하고 `Workers 2`로 낮춘 뒤, 남는 VRAM은 `Batch 32`로 활용한다. YOLO26의 기본 `nbs=64`에서는 Batch 16×누적 4와 Batch 32×누적 2가 같은 유효 배치 64를 유지한다.

## 실행 순서

```powershell
$ROOT = "<repo-root>"
$RUNNER = "$ROOT\data_training\yolo26_obstacle\run_yolo26.ps1"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task setup
$env:ROBOFLOW_API_KEY = "본인의_API_키"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task download
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task benchmark
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task train -Epochs 50 -Batch 16 -Workers 8
```

API 키는 환경변수로만 입력하며 파일에 저장하지 않는다.

## 중단 후 재개

```powershell
$LAST = "$ROOT\outputs\training\yolo26_obstacle\waste_yolo26n_640_b16_seed42\weights\last.pt"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task resume -Checkpoint $LAST
```

## 학습 후 validation

```powershell
$BEST = "$ROOT\outputs\training\yolo26_obstacle\waste_yolo26n_640_b16_seed42\weights\best.pt"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task val -Checkpoint $BEST -Batch 16 -Workers 8
```

최종 후보는 `best.pt`이며 `last.pt`는 재개용이다. TensorRT 엔진은 Jetson에서 해당 장치의 JetPack/TensorRT 환경으로 생성한다.
