# 실시간 녹 4등급 경량 학생 모델

## 목적

무거운 부식 분할 교사의 지식을 MobileNetV2 학생으로 전달해 Jetson 실시간 녹 분석에 사용할 4등급 모델을 학습합니다.

## 데이터와 모델

- 데이터: Virginia Tech CSSD 512×512 이미지와 4등급 mask
- 클래스: `Good, Fair, Poor, Severe`
- 개발 분리: train 316 / validation 80 / locked test 44
- 교사: DeepLabV3+ ResNet101
- 학생: MobileNetV2–DeepLabV3+ OS8
- 학생 입력: 512×512 RGB, ImageNet 정규화
- 학생 출력: 4-class raw logits

공식 출처는 [부식 상태 데이터셋](https://data.lib.vt.edu/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663)과 [공개 교사 모델](https://data.lib.vt.edu/articles/code/Trained_Model_for_the_Semantic_Segmentation_of_Corrosion_Condition_States/16628668)입니다.

## 학습·평가 흐름

데이터 무결성과 고정 split을 감사하고, 출처와 해시가 확인된 legacy 교사 checkpoint를 안전한 state-dict로 변환합니다. 같은 분할과 학습 예산에서 지도학습 기준선과 지식증류 학생을 비교합니다. validation으로 모델을 선택한 뒤 locked test를 한 번 평가하고 ONNX parity를 확인합니다.

## 산출물

학생 checkpoint, 학습·평가 지표, TensorBoard 기록, 고정 split 감사 자료와 4-class logits ONNX를 생성합니다. TensorRT plan은 대상 Jetson에서 별도로 검증합니다.

## 현재 상태와 한계

현재 실시간 녹 런타임의 기준 계보입니다. 공개 교사가 원래 공개 test를 모델 선택에 사용했을 가능성이 있어 증류 결과를 완전히 독립적인 일반화 성능으로 주장하지 않습니다. 표면 상태를 분할할 뿐 강재 두께·구조 강도·잔존 수명을 판정하지 않습니다.
