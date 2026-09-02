# 학습·평가 코드

로컬 NVIDIA GPU PC에서 사용한 데이터 준비, 학습, 평가와 ONNX export 코드입니다. 데이터셋, 촬영 사진, 가중치, ONNX, TensorRT plan, 가상환경과 실행 결과는 포함하지 않습니다.

## 필수 경로

| 목적 | 경로 | 상태 |
| --- | --- | --- |
| 실시간 녹 | `vt_kd/`, `realtime_w1280/` | 현재 런타임 계보 |
| 캡처 녹·크랙 | `capture_1280x720/` | 기준 export |
| 캡처 녹 hard-negative | `capture_rust_teacher_hardneg_v1/` | `--capture-test` 후보 |
| 상단 이물질 | `yolo26_obstacle/`, `YOLO26_hard_negative_1280x720/` | 현재 이물질 계보 |
| 초기 BGCrack | `steelcrack/` | 과거 계보, 외부 코드는 고정 commit으로 준비 |
| HrSegNet·5채널·경량 듀얼헤드 | `realtime_multitask_5ch_hrseg_v1/`, `realtime_light_dualhead_96_v1/` | 개발·연구, 미배포 |

배포 상태는 [대회 공개용 가이드](COMPETITION_CODE_GUIDE.md)와 [모델 상태표](../docs/MODEL_STATUS.md)가 기준입니다. 프로젝트별 데이터 배치, requirements, config, seed와 명령은 각 폴더 README를 확인하십시오.

## 재현과 검증

1. [제3자 고지](THIRD_PARTY_NOTICES.md)의 공식 출처에서 데이터와 checkpoint를 받고 라이선스·commit·SHA-256을 확인합니다.
2. 프로젝트별 가상환경을 만든 뒤 데이터 audit와 split leakage 검사를 먼저 실행합니다.
3. 고정 config/seed로 학습하고 validation에서 모델과 후처리를 선택합니다.
4. 선택 후 locked/sealed test를 평가하고 원 프레임워크와 ONNX Runtime 출력을 비교합니다.
5. TensorRT plan은 사용할 Jetson에서 생성해 I/O·NaN·실제 지연을 다시 검증합니다.

```powershell
$env:RAIL_ROBOT_ROOT = (Resolve-Path .).Path
python .\data_training\verify_public_release.py
```

## 주의

- 캡처 녹 hard-negative v6는 `--capture-test` 후보이며 정상 UART 임무에는 미배포입니다.
- HrSegNet-B32는 공식 checkpoint 변환본이고, 5채널 학생과 경량 듀얼헤드는 연구 코드입니다.
- hard negative가 모든 재질·색·조명의 오탐을 해결하지는 않습니다.
- 같은 시편·연속 프레임을 개발·검증·locked/sealed test 사이에 나누지 않으며 임계값은 잠금 시험 전에 고정합니다.
- 원본 데이터, 모델 바이너리, 가상환경, cache, outputs/runs, 비밀정보와 재배포 권한이 불명확한 자산은 공개본에서 제외합니다.
