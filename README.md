# DOLphin

카메라와 Jetson으로 레일 표면의 녹·크랙 후보와 이물질을 찾고, STM32로 주행 모터·클리너·워터펌프를 제어하는 축소 테스트베드용 로봇입니다.

> 비전 결과는 표면 이상 후보 선별용입니다. 구조 강도, 균열 깊이, 잔존 수명이나 실제 설비의 운행 안전을 판정하지 않습니다.

## 한눈에 보기

```text
SIDE/TOP 카메라 → Jetson(TensorRT) → UART → STM32F103 → 모터·클리너·펌프
                              └─ JSON·PNG·XLSX → 로컬 대시보드
```

| 경로 | 내용 |
| --- | --- |
| [`code`](code/README.md) | Jetson 실행 코드, 스크립트, 테스트 |
| [`stm_code`](stm_code/README.md) | STM32CubeIDE 펌웨어 |
| [`data_training`](data_training/README_KR.md) | 학습·평가·ONNX 변환 코드 |
| [`dashboard`](dashboard/README.md) | 검사 결과 조회 화면 |
| [`docs`](docs/MODEL_STATUS.md) | 모델 상태와 안전 한계 |

데이터셋, 촬영 영상, checkpoint, ONNX와 TensorRT plan은 용량·라이선스·장치 호환성 때문에 포함하지 않습니다.

## Jetson 실행

```bash
cd code
chmod +x scripts/*.sh
./scripts/requirement.sh
./scripts/run.sh --realtime-test
```

| 명령 | 용도 | 실제 UART 제어 |
| --- | --- | --- |
| `./scripts/run.sh --training` | `S` 키로 학습 사진 저장 | 없음 |
| `./scripts/run.sh --capture-test` | 캡처 녹·크랙 모델 확인 | 없음 |
| `./scripts/run.sh --realtime-test` | 실시간 ROI·FPS·판단 확인 | 기본 모의 출력 |
| `./scripts/run.sh` | 전체 촬영·청소·재검사 임무 | 사용 |

정상 임무 전에 `--no-uart` 또는 `--realtime-test`로 카메라, I/O shape와 모델 SHA를 확인해야 합니다. 정상 임무용 캡처 녹 plan은 검증한 SHA를 `RAIL_ROBOT_CAPTURE_RUST_ENGINE_SHA256`으로 직접 지정해야 UART 실행이 열립니다.

## 기준 모델

| 역할 | 기준 |
| --- | --- |
| 실시간 녹 | MobileNetV2–DeepLabV3+, 1280×240, TensorRT FP16 내부 연산 |
| 캡처 녹 | ResNet101–DeepLabV3+, 1280×720 |
| 실시간·캡처 크랙 | HrSegNet-B32 |
| 상단 이물질 | YOLO26n hard-negative |
| 경량 듀얼헤드 | 연구 후보, 현재 런타임 미사용 |

정확한 전처리, plan 이름과 승인 상태는 [모델 상태표](docs/MODEL_STATUS.md)에 있습니다. TensorRT plan은 사용할 Jetson에서 같은 TensorRT 버전으로 생성해야 합니다.

## 클리너 출력

STM32 TIM1은 `ARR=3599`이므로 한 주기는 3600 count입니다.

| compare | 실제 PWM duty | 사용 단계 |
| ---: | ---: | --- |
| 1200 | 1200 / 3600 = **33.3%** | 기본 클리너 출력 |
| 2000 | 2000 / 3600 = **55.6%** | 녹 1단계 출력 |
| 0 | **0%** | 녹 2·3단계 또는 크랙 기준 초과 시 정지 |

워터펌프는 녹이 감지되거나 크랙 점유율이 제어 기준을 넘으면 정지합니다. 세부 판단은 최근 모델별 4회 결과를 사용하며, 결과 부족·오류·stale 상태에서는 안전 정지를 선택합니다.

## PC 검증

```powershell
python -m pip install -r .\code\requirements-test.txt
python -m unittest discover -s code/tests -p "test_*.py"
python .\data_training\verify_public_release.py

cd dashboard
npm ci
npm run typecheck
npm test
```

STM32는 `stm_code/code/encoder/encoder.ioc`를 STM32CubeIDE에서 clean build합니다. Jetson·카메라·UART·모터·펌프 통합시험은 축소 테스트베드에서 별도로 수행해야 합니다.

## 안전과 라이선스

- 실제 가동 설비 시험은 시설관리자의 승인, 운행 중지, 위험성평가, 보호구와 물리 비상정지 절차가 있을 때만 수행합니다.
- 현재 프로토콜에는 명령별 ACK, version, sequence와 CRC가 없습니다.
- 자세한 제한은 [안전 문서](docs/SAFETY_AND_LIMITATIONS.md)를 확인하십시오.
- 팀 코드와 제3자 구성요소의 이용조건은 [LICENSE](LICENSE)와 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)에 정리했습니다.
