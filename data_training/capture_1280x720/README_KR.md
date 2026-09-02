# 1280×720 캡처 모델 기준선

## 목적

정지 캡처 전체 프레임을 녹·크랙 모델에 입력하기 위한 고정 shape ONNX 기준선과 회귀 검증 절차를 보존합니다.

## 데이터와 모델

| 대상 | 모델 | 입력 | 출력 |
| --- | --- | --- | --- |
| 녹 4등급 | DeepLabV3+ ResNet101, OS8 | BGR FP32 `[1,3,720,1280]` | 4-class logits `[1,4,720,1280]` |
| 크랙 후보 | BGCrack V1 | RGB FP32 `[-1,1]`, 외부 `[1,3,720,1280]` | 확률맵 `[1,1,720,1280]` |

녹은 Virginia Tech의 [부식 데이터](https://data.lib.vt.edu/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663)와 [교사 모델](https://data.lib.vt.edu/articles/code/Trained_Model_for_the_Semantic_Segmentation_of_Corrosion_Condition_States/16628668)을, 크랙은 [Civil-dataset](https://github.com/hzlbbfrog/Civil-dataset)과 [BGCrack](https://github.com/hzlbbfrog/BGCrack)을 출발점으로 사용했습니다.

## 학습·평가 흐름

기존 checkpoint를 1280×720 고정 입력으로 재구성하고 finite/NaN과 ONNX Runtime 동등성을 확인합니다. 공개 512×512 자료는 늘이지 않고 큰 canvas 중앙에 배치해 원본 영역만 회귀 평가합니다. 실촬영 라벨이 준비된 경우에만 별도 파인튜닝을 수행합니다.

## 산출물

녹·크랙 고정 shape ONNX, 전처리 metadata, 무결성 해시와 공개 데이터 회귀 보고서를 생성합니다. TensorRT plan은 실제 Jetson에서 별도로 만듭니다.

## 현재 상태와 한계

녹 ResNet101은 정상 임무의 캡처 기준 계보입니다. 이 폴더의 BGCrack 캡처 경로는 과거 개발 기준이며 현재 런타임 크랙 기준은 HrSegNet-B32입니다. 공개 자료 shape 회귀가 잠정 허용 기준을 통과하지 못했으므로 실제 레일 카메라 정확도를 검증한 결과로 해석하지 않습니다.
