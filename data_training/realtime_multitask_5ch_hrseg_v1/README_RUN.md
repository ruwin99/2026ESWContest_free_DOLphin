# 실시간 녹·크랙 5채널 학생 모델

## 목적

기존 실시간 녹 학생에 크랙 head를 추가해 1280×240 한 프레임에서 두 표면 후보를 함께 출력하는 개발 모델을 연구했습니다.

## 데이터와 모델

- 입력: RGB, ImageNet 정규화, FP32 `[1,3,240,1280]`
- 학생: MobileNetV2–DeepLabV3+ OS8 듀얼헤드
- 출력: `Good, Fair, Poor, Severe, Crack` raw logits `[1,5,240,1280]`
- 크랙 유효 영역: rows `112:240`
- 녹 교사: 현재 실시간 녹 모델
- 크랙 교사: 공식 HrSegNet-B32 변환본
- 공개 양성: [CrackSeg9k V4](https://doi.org/10.7910/DVN/EGIEBY)의 원본 crop을 연결한 개발용 panel

공식 출처는 [HrSegNet](https://github.com/CHDyshli/HrSegNet4CrackSegmentation)과 [CrackSeg9k](https://github.com/Dhananjay42/crackseg9k)입니다.

## 학습·평가 흐름

HrSegNet의 Paddle checkpoint를 고정 shape ONNX로 변환해 수치 동등성을 확인합니다. 공개 크랙 자료로 crack head를 bootstrap하고, 카메라 원본 GT가 준비된 뒤에만 제한적 joint fine-tune을 진행하도록 설계했습니다. 데이터 출처·촬영 그룹·teacher cache의 해시가 달라지면 재개를 막습니다.

## 산출물

공식 교사 변환본, teacher cache, 단계별 checkpoint, 평가 보고서와 5채널 raw-logit ONNX 후보를 생성합니다. 구조 확인용 ONNX와 정확도 후보는 구분합니다.

## 현재 상태와 한계

공개 CrackSeg9k 기반 bootstrap과 제한된 정상 시연 사진 검사는 완료했지만, 실제 카메라 크랙 양성 GT와 독립 locked test가 없습니다. 개발용 demo 후보일 뿐 최종 배포 승인이 아니며 현재 기준 런타임에는 사용하지 않습니다.
