# 모델 상태와 배포 계약

이 문서는 저장소에 포함하지 않은 모델 바이너리의 역할과 현재 검증 범위를 기록합니다. TensorRT plan은 Jetson/TensorRT/GPU에 종속되므로 다른 장치에서 복사해 쓰지 않고 대상 Jetson에서 다시 생성합니다.

## 활성 런타임 계약

| 역할 | 기준 파일명 | I/O 요약 | 상태 |
| --- | --- | --- | --- |
| 캡처 녹 기준 | `corrosion-capture-r101-os8-w1280-h720-fp32.plan` | OpenCV BGR `0..255` → FP32 `[1,3,720,1280]` → 4-class logits | 정상 임무 기준 경로 |
| 캡처 녹 hard-negative | `capture-rust-r101-os8-hardneg-v6-w1280-h720-fp32-notf32.plan` | OpenCV BGR `0..255` → FP32 `[1,3,720,1280]` → 4-class logits | `--capture-test`에만 통합된 후보; 정상 UART 임무에는 미배포, 독립 positive sealed test 미완료 |
| 실시간 녹 | `realtime-rust-mnv2-os8-w1280-h240-fp16.plan` | BGR→RGB, `/255`, ImageNet 정규화 → FP32 `[1,3,240,1280]` → 4-class logits | 실시간 기준; plan 내부 FP16, CUDA argmax 사용 |
| 캡처 크랙 | `hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32.plan` | BGR→RGB, `x/127.5-1` → FP32 `[1,3,720,1280]` → crack score/mask | 공식 HrSegNet 기반 후보 |
| 실시간 크랙 | `hrsegnet-b32-crackseg9k-realtime-w1280-h128-fp32-notf32.plan` | BGR→RGB, `x/127.5-1` → FP32 `[1,3,128,1280]` → crack score/mask | 공식 HrSegNet 기반 후보 |
| 상단 이물질 | `obstacle-yolo26n-hardneg-all1410-roi-y0-240-w1280-h240-int-h256-fp32-notf32-trt10.3.plan` | 1280×240 ROI를 RGB `/255`, 상하 8픽셀 패딩 → FP32 `[1,3,256,1280]` → boxes/classes | hard-negative 후보 |

실시간 녹, 캡처·실시간 HrSeg와 이물질 모델의 SHA-256 승인값은 `code/scripts/run.sh`에 고정되어 시작 시 실제 파일과 비교합니다. 정상 임무의 캡처 녹 기준 모델은 경로만 기본 제공되며 승인 SHA를 임의로 넣지 않았습니다. 먼저 `--no-uart`에서 실제 파일과 I/O를 확인하고, 승인한 SHA를 `RAIL_ROBOT_CAPTURE_RUST_ENGINE_SHA256`으로 명시해야 UART 임무가 열립니다. 승인 경로나 SHA를 바꾸는 일은 새 모델의 I/O·수치 동등성·실제 카메라 회귀시험을 통과한 뒤에만 수행해야 합니다.

## 연구 후보

`realtime_light_dualhead_96_v1`은 MobileNetV2 OS8 encoder, LR-ASPP 계열 96채널 decoder와 녹/크랙 출력 head를 함께 시험한 개발 코드입니다. 실제 카메라 hard negative와 독립 sealed test가 충분하지 않아 현재 기준 런타임 모델로 사용하지 않습니다.

`realtime_multitask_5ch_hrseg_v1`은 HrSegNet 교사 변환, 공개 CrackSeg9k bootstrap, 5채널 학생 실험을 재현하기 위한 코드입니다. 폴더가 존재한다는 사실은 최종 배포 승인을 뜻하지 않습니다.

## 바이너리 준비 원칙

1. 각 `data_training/<project>/README*`의 공식 출처에서 데이터와 checkpoint를 준비합니다.
2. 원본 SHA와 라이선스를 확인합니다.
3. 고정 config/seed로 학습·평가하고 validation에서 후보와 임계값을 선택합니다.
4. 후보 선택 이후에만 locked/sealed test를 평가합니다.
5. PyTorch/Paddle과 ONNX Runtime의 출력 동등성을 확인합니다.
6. 대상 Jetson에서 `trtexec`로 plan을 만들고 I/O, finite/NaN, 실제 카메라 지연과 오탐을 확인합니다.

저장소에는 데이터, checkpoint, ONNX, plan을 넣지 않습니다. 재배포 권한이 확인된 모델도 필요하면 GitHub Release/LFS 등 별도 배포 방식을 사용하고, 이 소스 제출 커밋과 해시로 연결해야 합니다.
