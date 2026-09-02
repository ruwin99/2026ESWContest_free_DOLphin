# YOLO26 장애물 hard-negative 학습 준비 (1280×720 카메라 기준)

## 권장 학습 명령

PowerShell을 어느 폴더에서 열어도 아래 명령을 그대로 실행할 수 있다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$env:RAIL_ROBOT_ROOT\data_training\yolo26_obstacle_hardneg_v1\run_hardneg.ps1" `
  -Task train `
  -Batch 8 `
  -Workers 2
```

현재 `-Task train`은 사용자가 지정한 v2~v7 JPG 1,245장과 기존 v8 학습분 110장을 모두
사용한다. v8의 나머지 55장은 독립 개발 평가용으로 유지한다. 전체 학습 세트는 기존 양성
3,390장과 hard negative 1,355장, 합계 4,745장이다.

안전 학습은 기존 모델의 BatchNorm 통계와 검출/박스 회귀 계층을 고정하고, 장애물 분류
최종 출력 12개 텐서(총 390개 파라미터)만 3 epoch 미세조정한다. 학습 종료 후 고정 텐서가
하나라도 달라졌으면 실패 처리하고 `reports\all1410_train_frozen_audit.json`에 기록한다.

이 폴더는 기존 Roboflow `-ohs3h/2-iemaw/1` 양성 데이터와 `for model test v8` 정상 카메라 프레임을 결합해 YOLO26n을 로컬 RTX 5070 Ti에서 파인튜닝하기 위한 격리된 작업 영역이다.

## 해상도 계약

- 카메라 및 hard negative 원본: 1280×720
- 학습: `imgsz=1280`, `rect=True`
- 16:9 입력은 YOLO stride 정렬에 따라 보통 1280×736 텐서로 최소 패딩된다.
- predict/export/운영 검증: `(height,width)=(720,1280)`, 고정 입력이면 `rect=False`
- Roboflow 양성 데이터는 이미 640×640 Fit으로 export되어 있다. 1280 학습은 잃어버린 원본 디테일을 복구하지 못한다.

Ultralytics train은 직사각형 `(h,w)`가 아니라 단일 정수 `imgsz`만 받기 때문에 위처럼 구성한다. `rect=True`가 batch 내 최소 패딩을 적용한다.

## 안전 원칙

- 원본 데이터셋, 기존 run, 기존 모델을 수정하거나 덮어쓰지 않는다.
- hard negative는 사람이 contact sheet 전체를 확인한 뒤에만 빈 라벨로 승인한다.
- v8 그룹 1·2(110장)만 train에 추가하고 그룹 3(55장)은 개발용 음성 평가로 격리한다.
- 기존 Roboflow valid/test split은 변경하지 않는다.
- 누락된 baseline `best.pt`를 `last.pt`나 임의 epoch 파일로 대체하지 않는다.
- locked test는 candidate, confidence, IoU를 고정한 뒤 한 번만 평가한다.

## 봉인된 최종 test

무버전 `for model test` 폴더는 사용자가 지정한 최종 test 전용 데이터다. 현재 train 목록과
manifest에는 이 폴더의 파일이 0장 포함되어 있다. 현재 학습이 끝나고 candidate checkpoint와
confidence/IoU를 고정하기 전에는 파일을 열거나 평가하지 않는다. 최종 평가 결과를 보고 같은
test에 맞춰 재학습하거나 checkpoint를 다시 선택하지 않는다. 계약은
`manifests\final_test_policy.yaml`에 기록되어 있다.

## 현재 필요한 baseline

다음 파일을 복원해야 smoke/train이 열린다.

```text
models/baseline/best.pt
size: 5,387,269 bytes
sha256: 333c49d129698e2ed781a6862b12224ea3eafe973b582322bffc6ed8573087bd
```

## 현재 데이터 차단 상태

자동 감사에서 원본 Roboflow train/valid/test 사이의 exact SHA 중복이 발견되면 준비기는 `BLOCKED_DATASET_SPLIT_LEAKAGE`로 중단한다. 이 경우 원본은 그대로 보존하고, 최소한 test > val > train 우선순위로 중복 사본을 제외한 새 manifest를 별도 승인한 뒤 사용해야 한다. 단순히 경고를 무시한 기존 split 평가는 허용하지 않는다.

## 실행 순서

```powershell
$ROOT = $env:RAIL_ROBOT_ROOT
$RUNNER = "$ROOT\data_training\yolo26_obstacle_hardneg_v1\run_hardneg.ps1"

# 자동 데이터/라벨/SHA/누수 감사 및 contact sheet 생성
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task audit

# contact_sheets의 모든 페이지를 사람이 확인한 뒤에만 실행
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task approve-reviewed

# 원본 split을 보존한 train manifest와 빈 라벨 데이터 생성
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task prepare

# 현재 export처럼 exact SHA split 누수가 있을 때 명시적으로 교정 정책 승인
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER `
  -Task prepare-dedup -AcceptExactDedupPolicy

# baseline SHA, CUDA, Ultralytics, 데이터 계약 확인
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task preflight

# 2% / 1 epoch smoke
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task smoke

# 본 학습 (기본 batch 8)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task train -Batch 8 -Workers 2
```

RTX 5070 Ti 16GB에서 1280 rectangular batch 8을 1차 고정값으로 사용한다. smoke 또는 본 학습에서 CUDA OOM이 나면 실패 run과 로그를 보존하고 baseline `best.pt`에서 batch 4로 새 이름의 실행을 시작한다.

