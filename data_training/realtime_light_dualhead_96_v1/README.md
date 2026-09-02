# 경량 녹·크랙 듀얼헤드 96

## 목적

실시간 녹과 크랙을 한 번의 backbone 추론으로 동시에 처리해 두 모델을 번갈아 실행하는 구조를 줄일 수 있는지 검토한 연구 프로젝트입니다.

## 데이터와 모델

- 입력: RGB, ImageNet 정규화, FP32 `[1,3,240,1280]`
- 구조: MobileNetV2 OS8 encoder + LR-ASPP 96채널 shared decoder
- 출력: `Good, Fair, Poor, Severe, Crack` raw logits `[1,5,240,1280]`
- 크랙 유효 영역: 입력의 rows `112:240`
- 교사: 기존 실시간 녹 모델과 공식 HrSegNet-B32
- 개발 데이터: [CrackSeg9k V4](https://doi.org/10.7910/DVN/EGIEBY)와 팀 정상 레일 hard-negative

HrSegNet 출처는 [공식 저장소](https://github.com/CHDyshli/HrSegNet4CrackSegmentation)입니다.

## 학습·평가 흐름

두 교사의 FP32 출력을 고정 cache하고, 녹 head·크랙 head·shared decoder·encoder 순으로 단계적으로 학습합니다. 촬영 세션과 원본 자료 단위의 split 누수를 감사하고 여러 seed의 개발 결과를 비교합니다. PyTorch와 ONNX Runtime의 raw logits 동등성과 모델 복잡도를 확인합니다.

## 산출물

5채널 checkpoint와 FP32 ONNX, 평가·누수 감사·복잡도 보고서, GPU 후처리 연구용 wrapper를 생성합니다.

## 현재 상태와 한계

Phase A 개발과 export만 승인된 연구 후보입니다. 실제 인쇄물·카메라 양성 자료의 독립 sealed test가 없어 `ACCURACY_NOT_FINAL / NOT_FOR_UART`이며 현재 시연 런타임과 액추에이터 제어에는 사용하지 않습니다.
