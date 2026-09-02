# Jetson 실행 코드

이 디렉터리는 두 대의 USB 카메라 영상에서 레일 표면의 녹·균열 후보와 상단 이물질을 분석하고, STM32와 통신해 축소 테스트베드의 주행·클리너·워터펌프를 제어합니다. 카메라 결과는 표면 이상 후보 선별용이며 구조 안전 판정이 아닙니다.

## 시작하기

Jetson Orin Nano에서 다음을 실행합니다.

```bash
cd ~/Desktop/rail_robot/code
chmod +x scripts/*.sh
./scripts/requirement.sh
```

`requirement.sh`는 JetPack용 TensorRT, CUDA Python, OpenCV, UART와 XLSX 의존성을 확인합니다. 모델은 자동으로 내려받지 않습니다. TensorRT plan은 대상 Jetson과 같은 JetPack/TensorRT 환경에서 생성해야 합니다.

## 실행 모드

| 명령 | 용도 | UART |
| --- | --- | --- |
| `./scripts/run.sh --training` | `S` 키로 원본 학습 사진 저장 | 사용 안 함 |
| `./scripts/run.sh --capture-test` | 캡처용 녹·크랙 모델 수동 확인 | 사용 안 함 |
| `./scripts/run.sh --realtime-test` | 단일 카메라 실시간 모델·FPS·제어 판단 확인 | 기본 모의 출력 |
| `./scripts/dual_camera_test.sh` | SIDE/TOP 실시간 통합 시험 | 기본 모의 출력 |
| `./scripts/dual_camera_test.sh --headless --timing` | 화면 없이 5초 간격 지연 통계 출력 | 사용 안 함 |
| `./scripts/dual_camera_test.sh --uart` | 두 카메라와 실제 액추에이터 시험 | 사용 |
| `./scripts/run.sh --no-uart` | 전체 엔진·카메라와 수동 캡처·보고 확인 | 사용 안 함 |
| `./scripts/run.sh` | 캡처→청소→재검사 정상 임무 | 사용 |
| `./scripts/dc_test.sh` | FRONT/SIDE cleaner·pump 개별 UART 시험 | 사용 |

`--capture-test`와 `--training`은 GUI가 필요합니다. `S`로 저장하고 `Q` 또는 `Esc`로 종료합니다. 학습 사진은 `for model/`에 저장되며 Git에서 제외됩니다.

UART를 쓰기 전에는 반드시 `--no-uart` 또는 기본 `dual_camera_test.sh`로 카메라, 엔진 경로, SHA-256, I/O shape와 제어 로그를 확인하십시오. 정상 임무는 프로그램이 준비된 뒤 터미널에 `start`를 입력해야 시작합니다.

## 필수 장치와 경로

| 역할 | 기본값 |
| --- | --- |
| SIDE 카메라 | `/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0` |
| TOP 카메라 | `/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0` |
| STM32 UART | `/dev/ttyACM0`, 115200 baud, 8-N-1 |
| TensorRT plan | `${HOME}/models/...` 아래 역할별 승인 파일 |
| 캡처 결과·XLSX | `outputs/captures/`, `outputs/` |
| 대시보드 JSON·PNG | `outputs/dashboard/runs/`, `outputs/dashboard/media/` |

같은 Arducam 장치가 동일 USB serial을 보고하므로 카메라는 `/dev/v4l/by-id`가 아니라 위 USB 포트 경로로 구분합니다. 두 경로가 없거나 같은 video node를 가리키면 UART를 열기 전에 종료합니다.

현재 활성 모델의 정확한 파일명, 입력 전처리, shape와 배포 상태는 [모델 상태표](../docs/MODEL_STATUS.md)가 기준입니다. `scripts/run.sh`는 승인된 실시간 녹, HrSegNet 크랙, 상단 이물질 plan과 SHA를 고정합니다. 정상 임무의 캡처 녹 plan은 팀이 승인한 SHA를 `RAIL_ROBOT_CAPTURE_RUST_ENGINE_SHA256`으로 명시해야 UART 실행이 열립니다. ONNX·plan 파일은 저장소에 포함하지 않습니다.

## 현재 처리와 제어 기준

- SIDE 캡처: 전체 1280×720 녹 모델과 HrSegNet 크랙 모델
- TOP 캡처: 전체 1280×720 HrSegNet 크랙 모델
- SIDE 실시간: `y=0:240` 녹, `y=112:240` 크랙 ROI
- TOP 실시간: `y=0:240` 이물질, `y=112:240` 크랙 ROI
- 실시간 모델별 최근 4개 결과와 freshness를 확인하며, 결과 부족·stale·오류에서는 해당 출력을 OFF로 둡니다.
- 균열 mask가 ROI의 0.05%를 넘으면 해당 역할의 cleaner와 pump를 모두 OFF합니다.

| 역할 | 균열이 안전할 때 판단 | cleaner 명령 | pump |
| --- | --- | --- | --- |
| FRONT | 이물질 없음 | `PWM_33_3` | ON |
| FRONT | 최근 4회 중 이물질 있음 | `PWM_55_6` | ON |
| SIDE | 녹 0단계 | `PWM_33_3` | ON |
| SIDE | 녹 1단계 | `PWM_55_6` | OFF |
| SIDE | 녹 2·3단계 | OFF | OFF |

UART 프로토콜도 실제 출력에 맞춰 `PWM_33_3`과 `PWM_55_6`을 사용합니다. STM32 TIM1 ARR 3599와 compare 1200/2000의 실제 cleaner duty는 각각 `1200/3600 = 33.3%`, `2000/3600 = 55.6%`입니다.

## 출력

초기 검사와 재검사의 SIDE/TOP 원본·분석 이미지를 저장하고, 사진별 녹 등급·부식률·균열 후보 비율을 XLSX에 기록합니다. 대시보드용 JSON/PNG 내보내기는 보조 기능이며 실패해도 기존 JPEG·XLSX와 안전 제어는 유지됩니다. 온라인 DB 업로드는 이 코드에 포함되지 않습니다.

## 검증

하드웨어가 없는 PC에서는 저장소 루트에서 문법과 공개본 검사를 실행할 수 있습니다.

```powershell
python -c "from pathlib import Path; [compile(p.read_text(encoding='utf-8-sig'), str(p), 'exec') for p in Path('code').rglob('*.py')]"
python .\data_training\verify_public_release.py
```

OpenCV·NumPy·openpyxl이 준비된 환경에서는 회귀 테스트도 실행합니다.

```powershell
python -m unittest discover -s code/tests -p "test_*.py"
```

Jetson에서는 각 plan의 SHA, TensorRT I/O, finite/NaN, 워밍업, 실제 카메라 지연을 확인합니다. STM32는 최신 펌웨어를 업로드한 뒤 모터드라이버 전원을 끄거나 바퀴를 띄운 축소 테스트베드에서 먼저 시험합니다.

## 주의

- 실제 가동 크레인에서는 사용하지 마십시오. 시험 조건은 [안전 및 한계](../docs/SAFETY_AND_LIMITATIONS.md)를 따릅니다.
- 현재 UART 프로토콜에는 명령별 ACK, sequence, CRC와 주행 모터용 원격 STOP이 없습니다.
- `--no-crack`과 모델 경로 덮어쓰기는 안전 인터록을 약화할 수 있으므로 정상 임무에 사용하지 않습니다.
- 모델의 녹·균열·이물질 결과는 조명, 레일 재질, 각도와 학습 분포 밖 배경에서 오탐·미탐이 생길 수 있습니다.
- 엔진·환경변수의 상세 계약이 필요하면 `scripts/run.sh --help`, 소스와 [루트 README](../README.md)를 확인하십시오.
