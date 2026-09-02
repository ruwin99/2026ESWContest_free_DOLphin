# 캡처 녹 hard-negative 미세조정

## 목적

고정 카메라에서 정상 레일이 녹으로 흔들려 보이던 오탐을 줄이되 기존 캡처 녹 기준 모델을 보존한 별도 후보를 만듭니다.

## 데이터와 모델

- 모델: DeepLabV3+ ResNet101, output stride 8
- 입력: OpenCV BGR FP32 `[1,3,720,1280]`
- 출력: `Good, Fair, Poor, Severe` raw logits `[1,4,720,1280]`
- 양성 자료: Virginia Tech 4등급 부식 데이터
- 음성 자료: 팀 카메라로 촬영하고 사람이 `Good`으로 검토한 정상 레일 사진
- 라벨 예외: 불확실 영역은 ignore index `255`

공개 양성 자료와 기준 교사는 [Virginia Tech CSSD](https://data.lib.vt.edu/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663)에서 출발했습니다.

## 학습·평가 흐름

연속 프레임을 촬영 그룹 단위로 분리하고 정상 zero-mask를 사람 검토 후 승인합니다. BatchNorm을 고정한 채 classifier·decoder부터 미세조정하고, 필요할 때만 backbone 상단을 제한적으로 엽니다. validation을 통과한 후보만 ONNX로 내보내며 독립 sealed test는 모델 선택에 사용하지 않습니다.

## 산출물

검토 기록, split 감사 결과, 단계별 checkpoint, 평가 보고서와 1280×720 ONNX 후보를 생성합니다. 기존 캡처 기준 모델은 덮어쓰지 않습니다.

## 현재 상태와 한계

v6 후보는 수동 `--capture-test`에만 통합됐습니다. 정상 UART 임무에는 배포하지 않았고, 실제 녹 양성을 포함한 독립 sealed test와 두 번째 독립 라벨 검토가 충분하지 않아 최종 정확도를 주장하지 않습니다.
