# 실시간 W1280 모델 ONNX 준비

`REALTIME_MODELS_W1280_H240_H128_TRAINING_GUIDE.md`의 1차 기준에 맞춰 기존 체크포인트를 새 고정 shape로 다시 export한다. 새 학습이나 새 실촬영 데이터는 필요하지 않다.

## 모델 계약

- 녹: `frame[480:720, :]`, 입력 `[1,3,240,1280]`, 출력 `logits [1,4,240,1280]`
- 크랙: `frame[592:720, :]`, 입력 `[1,3,128,1280]`, 출력 `crack_probability [1,1,128,1280]`
- 입력 resize, letterbox와 shape 보정 padding 없음
- 크랙 DCT: HFIE1 `32×320`, HFIE2 `16×160`
- 두 ONNX 모두 static FP32, Opset 18
- 크랙 출력은 sigmoid가 이미 적용된 확률이므로 런타임에서 sigmoid를 다시 적용하지 않음

## 두 파일 다시 생성

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo-root>\data_training\realtime_w1280\run_realtime_models.ps1" -Task export-all
```

개별 생성:

```powershell
-Task export-rust
-Task export-crack
```

생성 위치:

```text
output/models/realtime_w1280/realtime-rust-mnv2-os8-w1280-h240-fp32.onnx
output/models/realtime_w1280/realtime-crack-bgcrack-w1280-h128-fp32.onnx
```

TensorRT plan은 사용할 Jetson에서 직접 생성하고 FP32 ONNX Runtime 결과와 비교한다. 이 파일들은 기존 512 모델과 캡처용 720×1280 모델을 덮어쓰지 않는다. 실제 카메라 환경 정확도는 아직 미검증이다.
