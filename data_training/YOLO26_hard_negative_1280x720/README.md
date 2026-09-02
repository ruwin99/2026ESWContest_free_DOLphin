# YOLO26 hard-negative 학습 코드 정리본

이 폴더는 1280×720 철도 카메라용 장애물 검출 모델의 hard-negative 미세조정에 실제로 사용한 코드를 모아 둔 보관본이다. 데이터 이미지, 가중치, 학습 결과는 용량과 중복 문제 때문에 포함하지 않았으며 기존 위치를 그대로 사용한다.

## 최종 학습 요약

- 모델: YOLO26n, 단일 클래스 `obstacle`
- 카메라 원본: 1280×720
- 학습 설정: `imgsz=1280`, `rect=True`
- 실제 stride 정렬 입력: 일반적인 16:9 배치에서 1280×736
- 데이터: 기존 양성 3,390장 + hard negative 1,355장 = 학습 4,745장
- hard negative 원본: `for model test v2`~`v7` 1,245장 + v8 학습 그룹 110장
- v8 개발 평가 그룹: 55장
- 최종 test: 무버전 `for model test` 66장, 학습·튜닝에 사용하지 않고 1회만 평가
- 최종 미세조정: 3 epochs, batch 8, workers 2, AdamW, lr0 0.00002
- 변경 허용 범위: 분류 출력 12개 텐서, 총 390개 파라미터
- BatchNorm 통계와 박스 회귀 및 나머지 파라미터: 고정

자세한 결과와 해시는 [TRAINING_RECORD.md](TRAINING_RECORD.md)에 기록되어 있다.

## 폴더 구조

```text
YOLO26_hard_negative_1280x720/
├─ README.md                         현재 문서
├─ TRAINING_RECORD.md                실제 최종 학습·평가·모델 기록
├─ FINAL_TRAIN_COMMAND.ps1           최종 학습 명령 재실행용 진입점
├─ requirements-used.txt             당시 핵심 패키지 버전
├─ run_hardneg.ps1                   전체 작업 PowerShell 러너 원본
├─ REFERENCE_README_KR.md            당시 프로젝트 설명 원본
├─ configs/
│  └─ hardneg_1280x720_v1.yaml       데이터·학습 계약
├─ tools/
│  ├─ prepare_hardneg.py             v8 감사/분할/빈 라벨 준비
│  ├─ prepare_all_hardneg.py         v2~v7 및 v8 학습분 통합 준비
│  ├─ preflight.py                   CUDA·모델 SHA·데이터 사전 검사
│  ├─ train_cls_output_safe.py       분류 출력만 미세조정하고 동결 감사
│  ├─ compare_fixed_threshold_val.py 양성 검증 고정 threshold 비교
│  ├─ evaluate_hardneg_dev.py        v8 개발 음성 세트 FP 비교
│  ├─ evaluate_locked_final.py       봉인 최종 test 1회 평가
│  └─ export_final_onnx.py           최종 ONNX 내보내기 및 parity 검사
├─ tests/
│  └─ test_prepare_hardneg.py        데이터 준비 코드 테스트
└─ reference/
   ├─ dedup_policy_approval.yaml     exact-SHA 중복 제거 정책
   └─ final_test_policy.yaml         최종 test 봉인 및 결과 기록
```

## 실제 최종 학습 명령

PowerShell 위치와 관계없이 다음 명령으로 실행했다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$env:RAIL_ROBOT_ROOT\data_training\yolo26_obstacle_hardneg_v1\run_hardneg.ps1" `
  -Task train `
  -Batch 8 `
  -Workers 2
```

이 정리 폴더에서는 다음 진입점을 사용할 수 있다. 실제 원본 프로젝트 러너를 호출하므로 코드·데이터 경로가 달라지는 일을 막는다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\FINAL_TRAIN_COMMAND.ps1" `
  -ProjectRoot $env:RAIL_ROBOT_ROOT
```

기존과 같은 이름의 학습 결과 폴더가 이미 있으므로 그대로 재실행하면 덮어쓰기 방지 로직에 의해 중단되는 것이 정상이다. 새 학습을 할 때는 원본 러너의 run 이름을 새 버전으로 변경해야 한다.

## 전체 준비 순서

```powershell
$RUNNER = ".\run_hardneg.ps1"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task audit
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task approve-reviewed
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task prepare-dedup -AcceptExactDedupPolicy
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task prepare-all
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task preflight
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUNNER -Task train -Batch 8 -Workers 2
```

## 중요한 운영 주의사항

- `for model test`는 봉인된 최종 test이므로 학습, validation, confidence 조정, checkpoint 선택에 다시 사용하지 않는다.
- 카메라 입력은 1280×720이다. ONNX 입력은 위·아래 8픽셀씩 값 114로 패딩한 FP32 RGB `[1,3,736,1280]`이다.
- Ultralytics 설정 폴더가 `C:\Windows\System32\Ultralytics`로 잡혀 권한 오류가 나는 문제를 막기 위해 러너가 `YOLO_CONFIG_DIR`을 프로젝트 내부로 지정한다.
- 원본 데이터, baseline, 생성된 dataset manifest 및 결과 보고서는 `data_training\yolo26_obstacle_hardneg_v1`에 보존되어 있다.

## 코드 보관 기준

이 폴더의 원본 스크립트·설정·정책 파일은 2026-09-02 기준 실제 프로젝트 파일과 텍스트 내용이 동일하다. 보관 과정에서 줄바꿈만 LF 형식으로 정규화되었다.
