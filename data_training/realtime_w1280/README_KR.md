# 실시간 1280폭 ONNX 기준선

## 목적

기존 checkpoint를 실시간 고정 ROI 크기에 맞는 ONNX로 변환하고 원 모델과의 수치 동등성을 확인합니다. 새 학습을 수행하는 프로젝트는 아닙니다.

## 데이터와 모델

| 대상 | 모델 | 입력 | 출력 |
| --- | --- | --- | --- |
| 녹 4등급 | MobileNetV2–DeepLabV3+ OS8 | RGB·ImageNet 정규화 `[1,3,240,1280]` | logits `[1,4,240,1280]` |
| 크랙 후보 | BGCrack V1 | RGB `[-1,1]` `[1,3,128,1280]` | 확률맵 `[1,1,128,1280]` |

녹 모델은 [Virginia Tech 부식 데이터](https://data.lib.vt.edu/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663) 계보이고, 크랙 모델은 [BGCrack](https://github.com/hzlbbfrog/BGCrack) 계보입니다.

## 학습·평가 흐름

검증된 checkpoint를 고정 shape로 내보내고 finite/NaN, 정적 I/O와 ONNX Runtime parity를 검사합니다. resize나 letterbox를 모델 앞에 추가하지 않습니다. TensorRT plan은 대상 Jetson에서 만들고 FP32 ONNX 기준과 다시 비교합니다.

## 산출물

고정 입력의 녹·크랙 ONNX, 전처리 metadata와 무결성 해시를 생성합니다. 캡처용 모델과 기존 512×512 산출물은 덮어쓰지 않습니다.

## 현재 상태와 한계

녹 ONNX는 현재 실시간 녹 런타임 계보입니다. 이 폴더의 BGCrack 실시간 export는 과거 개발 경로이며 현재 크랙 기준은 HrSegNet-B32입니다. 모델의 실제 레일 정확도는 별도 카메라 검증이 필요합니다.
