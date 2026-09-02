# Steelcrack·BGCrack 학습 기준선

## 목적

공개 Steelcrack 데이터와 BGCrack 모델을 Windows 단일 GPU 환경에서 학습·평가하고 ONNX로 변환하는 과거 크랙 기준선을 보존합니다.

## 데이터와 모델

- 데이터: 512×512 RGB 이미지, 이진 mask와 edge
- 공식 split: Train 3,300 / Validation 525 / Test 530
- 모델: BGCrack V1
- 학습 정밀도: FP32
- ONNX 입력: RGB FP32 `[-1,1]`, `[1,3,512,512]`
- ONNX 출력: sigmoid가 적용된 크랙 확률맵 `[1,1,512,512]`

데이터 출처는 [Civil-dataset](https://github.com/hzlbbfrog/Civil-dataset), 모델 코드는 [BGCrack](https://github.com/hzlbbfrog/BGCrack)이며 공개본에는 원본 데이터와 제3자 소스를 복사하지 않습니다.

## 학습·평가 흐름

이미지·mask·edge 대응, 손상 파일과 split 중복을 감사합니다. 공식 모델 구조와 손실은 유지하고 Windows에서 불필요한 NCCL/DDP 초기화만 제거한 단일 GPU wrapper로 학습합니다. validation으로 checkpoint를 선택한 뒤 test를 한 번 평가하고, PyTorch와 ONNX 확률맵을 비교합니다.

## 산출물

best·resume checkpoint, 학습 지표, 실행 설정, 데이터 감사 보고서와 crack-probability ONNX를 생성합니다. 공식 FFT 경로의 ONNX 호환을 위해 동등한 실수 연산 변환을 사용합니다.

## 현재 상태와 한계

학습·평가·export 코드와 FP32 smoke 검증을 보존한 과거 BGCrack 계보입니다. 공식 FFT 경로는 FP16에서 불안정해 학습 기준을 FP32로 유지합니다. 현재 Jetson 크랙 런타임은 HrSegNet-B32이며, Steelcrack 성능을 실제 레일 일반화 성능으로 주장하지 않습니다.
