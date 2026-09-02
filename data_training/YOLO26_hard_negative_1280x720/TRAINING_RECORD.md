# 최종 학습 기록

## 데이터 구성

| 구분 | 수량 | 용도 |
|---|---:|---|
| 기존 양성 train | 3,390 | 학습 |
| hard negative v2~v7 | 1,245 | 학습 |
| hard negative v8 그룹 1·2 | 110 | 학습 |
| 전체 train | 4,745 | 학습 |
| 기존 validation | 1,530 | 양성 성능 검증 |
| 기존 test | 829 | 기존 데이터 test |
| hard negative v8 그룹 3 | 55 | 개발 음성 평가 |
| 무버전 `for model test` | 66 | 봉인 최종 test 1회 |

통합 데이터 설정:

```text
<RAIL_ROBOT_ROOT>\data_training\yolo26_obstacle_hardneg_v1\workspace\datasets\waste_detect_hn_all1410_v2_1280_seed42\data.yaml
```

## 최종 학습 설정

```text
model       : baseline best.pt
task        : detect
class       : 0 obstacle
epochs      : 3
batch       : 8
workers     : 2
imgsz       : 1280
rect        : True
device      : CUDA device 0
optimizer   : AdamW
lr0         : 0.00002
seed        : 42
deterministic: True
AMP         : True
```

`train_cls_output_safe.py`가 아래 분류 출력 12개 텐서만 학습 가능하도록 제한했다.

```text
model.23.cv3.{0,1,2}.2.{weight,bias}
model.23.one2one_cv3.{0,1,2}.2.{weight,bias}
```

총 변경 가능 파라미터는 390개다. 최종 감사 결과 12개 대상 텐서는 모두 변경됐고 고정 텐서 변경은 0개였다.

## 최종 checkpoint

```text
<RAIL_ROBOT_ROOT>\outputs\training\yolo26_obstacle_hardneg_v1\yolo26n_hn_all1410_1280rect_clsout_lr2e5_seed42_b8_v2\weights\best.pt
```

- 선택 epoch: 3
- SHA256: `5f08e1fd963627aa81522d00b75b72b4c633016b162e7de529f9849786881d56`

Epoch 3 양성 validation 결과:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.89478 | 0.89613 | 0.93979 | 0.69246 |

## 봉인 최종 test 결과

설정: confidence `0.30`, IoU `0.70`, 카메라 1280×720, 66장.

| 모델 | FP 발생 프레임 | FP 박스 | 최대 confidence |
|---|---:|---:|---:|
| baseline | 7 | 9 | 0.556151 |
| 최종 candidate | 6 | 7 | 0.517260 |

이 결과를 사용한 추가 재학습이나 threshold 조정은 금지되어 있다.

## 최종 ONNX

```text
<RAIL_ROBOT_ROOT>\output\models\yolo26_obstacle_hardneg_v2\obstacle-yolo26n-hardneg-all1410-camera-w1280-h720-int-h736-fp32.onnx
```

- ONNX SHA256: `ed1dd0161033c225ca6c71240c21e3758d52fae0113d3df76a915f892e22da46`
- 입력: FP32 RGB `[1,3,736,1280]`
- 전처리: 1280×720 영상 위·아래 각각 8픽셀을 값 114로 패딩한 뒤 255로 나눔
- 출력: `[1,300,6]`
- Opset: 17

## 당시 핵심 환경

```text
Python       3.12.12
PyTorch      2.11.0+cu128
Ultralytics  8.4.122
NumPy        2.5.2
PyYAML       6.0.3
Pillow       12.3.0
ONNX         1.22.0
ONNX Runtime 1.29.0
```
