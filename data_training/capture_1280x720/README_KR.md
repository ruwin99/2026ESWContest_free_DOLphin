# 캡처 전용 1280×720 모델 준비 및 실행 방법

이 폴더는 `CAPTURE_MODELS_W1280_H720_TRAINING_GUIDE.md` 사양에 맞춘 캡처 전용 녹·크랙 모델 작업 공간이다. 기존 512×512 실시간 모델과 결과 파일은 변경하지 않는다.

## 현재 1차 기준

새로 촬영한 사진이나 새 라벨은 필요하지 않다. 기존 공개 데이터, 공식 split과 기존 체크포인트로 다음 작업을 수행한다.

- 캡처 녹: Virginia Tech CSSD DeepLabV3+ ResNet-101 OS8 체크포인트 재사용
- 캡처 크랙: Steelcrack으로 학습한 BGCrack V1 `best.pth` 재사용
- 녹 외부 입출력: `1×3×720×1280 → 1×4×720×1280`
- 크랙: 외부 720×1280 → 위·아래 8px씩만 패딩 → 내부 736×1280 → 외부 720×1280 crop
- BGCrack DCT: HFIE1 `184×320`, HFIE2 `92×160` 재생성
- 가장 깊은 MobileViT 특징 맵 `23×40`에만 아래쪽 특징 1행을 임시 패딩하고 블록 통과 직후 제거
- 기존 공개 512 이미지는 720×1280 canvas 중앙에 원본 크기 그대로 배치하고, 중앙 512×512 픽셀만 회귀 평가
- 512 이미지를 늘이거나 16:9로 변형하지 않음
- 실제 레일·카메라 환경 정확도는 `미검증`으로 표시

현재 단계에서는 `train-corrosion`이나 `train-crack`을 실행하지 않는다. 두 작업은 나중에 실제 시연에서 오검출·누락이 확인되어 직접 촬영 자료를 추가할 때만 사용하는 선택적 파인튜닝 작업이다.

## 1. 실제 사진 없이 baseline 전체 실행

녹 baseline:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\capture_1280x720\run_capture_models.ps1" -Task baseline-corrosion
```

크랙 baseline:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\capture_1280x720\run_capture_models.ps1" -Task baseline-crack
```

각 baseline 작업은 다음을 순서대로 실행한다.

1. RTX 5070 Ti FP32 고해상도 forward와 NaN/Inf 검사
2. 고정 shape Opset 18 ONNX 생성 및 ONNX Runtime 동등성 검사
3. 기존 공개 Validation 512 이미지를 중앙에 무변형 배치해 기존 512 출력과 캡처 shape 출력을 비교

이미 생성된 ONNX 파일:

```text
output/models/capture_1280x720/corrosion-capture-r101-os8-w1280-h720-fp32.onnx
output/models/capture_1280x720/crack-capture-bgcrack-ext-w1280-h720-int-w1280-h736-fp32.onnx
```

## 2. 공개 데이터 회귀 평가만 다시 실행

공개 Validation:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\capture_1280x720\run_capture_models.ps1" -Task regression-corrosion-val

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\capture_1280x720\run_capture_models.ps1" -Task regression-crack-val
```

설정과 출력 계약을 고정한 뒤 공개 Test는 각각 다음 작업으로 한 번만 실행한다.

```powershell
-Task regression-corrosion-test
-Task regression-crack-test
```

회귀 평가 정책:

```text
공개 원본: 512×512
외부 canvas: 1280×720
배치 위치: top=104, left=384
resize/stretch: 없음
평가 영역: 중앙 원본 512×512만 사용
padding 영역: 지표에서 제외
```

공개 512×512 회귀 시험의 바깥 canvas는 기본적으로 `reflect`로 채운다. 이것은 공개 정사각형 자료의 shape 회귀 시험만 위한 정책이며, 실제 720×1280 카메라 영상에 추가되는 외부 패딩이 아니다.

현재 Validation 회귀는 녹과 크랙 모두 가이드의 잠정 지표 저하 허용치를 통과하지 못했다. 따라서 ONNX 실행·형상 기준선은 준비됐지만 실제 카메라 정확도는 여전히 `미검증`이며, 공개 Test는 설정을 확정하기 전에는 실행하지 않는다.

결과는 `manifests/public_regression_<model>_<split>.json`에 저장된다. 이 결과는 shape 변경 전후의 공개 데이터 회귀 결과일 뿐 실제 카메라 환경 정확도가 아니다.

## 3. 환경과 개별 작업

환경 확인:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\capture_1280x720\run_capture_models.ps1" -Task verify
```

고해상도 forward만 확인:

```powershell
-Task smoke-corrosion
-Task smoke-crack
```

ONNX만 다시 생성:

```powershell
-Task export-corrosion-onnx
-Task export-crack-onnx
```

녹 ONNX 계약:

- 입력 `images`: OpenCV BGR FP32 `0..255`, `[1,3,720,1280]`
- 출력 `logits`: FP32 `[1,4,720,1280]`
- 클래스: `Good, Fair, Poor, Severe`

크랙 ONNX 계약:

- 입력 `images`: RGB FP32 `[-1,1]`, `[1,3,720,1280]`
- 출력 `crack_probability`: sigmoid 적용 완료, `[1,1,720,1280]`
- 이진 threshold: `0.5`

## 4. 향후 선택적 실촬영 파인튜닝

실제 시연에서 정확도가 부족할 때만 아래 폴더에 1280×720 이미지와 라벨을 추가한다.

```text
data_training/capture_1280x720/
  rust/{train,validation,test}/{images,masks}
  crack/{train,validation,test}/{images,masks,edges}
```

그때 데이터 감사:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\capture_1280x720\run_capture_models.ps1" -Task audit
```

선택적 파인튜닝 작업은 `finetune-corrosion`, `finetune-crack`이다. 데이터가 없는 현재 단계에서는 실행하지 않는다. 기존 `train-corrosion`, `train-crack`은 호환용 별칭으로 남아 있지만 동일하게 실촬영 데이터가 있을 때만 동작한다.

## 5. TensorRT

TensorRT plan은 대상 Jetson에서 직접 생성한다. 먼저 FP32 plan을 ONNX Runtime FP32 출력과 비교하고, 크랙 FP16 plan은 별도 파일로 만들어 NaN/Inf와 픽셀 판정 일치율을 확인한다. `--best`로 정밀도를 자동 하향한 결과는 검증 없이 사용하지 않는다.
