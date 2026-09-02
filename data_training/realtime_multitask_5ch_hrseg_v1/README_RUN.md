# 실시간 녹 + 크랙 5채널 재학습 준비 상태

이 폴더는 기존 `realtime_multitask_5ch` 모델과 완전히 분리된 신규 실험 영역입니다.
입력은 RGB FP32 `[1,3,240,1280]`, 출력은 raw logit FP32
`[1,5,240,1280]`이며 채널 순서는 `Good, Fair, Poor, Severe, Crack`입니다.

## 현재 완료된 것

- 기존 녹 ONNX teacher의 SHA와 입출력 계약 확인
- 기존 MobileNetV2-DeepLabV3+ OS8 녹 체크포인트 strict-load 경로 준비
- 공식 HrSegNet-B32 저장소와 CrackSeg9k 체크포인트 보존
- 공식 Paddle 체크포인트 180개 키/shape strict 검사
- HrSegNet-B32를 고정형 `[1,3,128,1280] -> [1,2,128,1280]` ONNX로 변환
- Paddle/ONNX 수치 일치 확인(최대 절대 오차 약 `2.18e-6`)
- manifest/teacher/SHA/group leakage/readiness 감사 도구
- teacher cache, 단계별 학습, validation 평가, FP32 ONNX export/verify 도구

공식 teacher 메타데이터는
`official_assets/hrsegnet-b32-crackseg9k-realtime-w1280-h128-fp32.metadata.json`에 있습니다.

## 현재 준비 상태

카메라 원본만 사용하는 최종 학습/평가 계약은 아직 차단 상태입니다. 다만 사용자가
제공한 공식 CrackSeg9k V4 압축 해제본 9,159장은 구조와 이미지/마스크 쌍 검사를
통과했으며, 공개 데이터 기반 `crack_bootstrap` 개발 학습은 실행할 수 있습니다.

1. `1280x720` 실제 카메라 원본에서 rows `112:240`에 맞춘 크랙 양성 GT가 없음
2. 물리 시편·촬영 세션·인코더 section 기준 group/provenance가 없음
3. 독립 validation과 외부 평가자가 보관할 locked test가 확정되지 않음
4. config의 `preregistered_acceptance`와 크랙 loss의 `null` 항목이 미확정
5. 따라서 `status.training_authorized`는 의도적으로 `false`

Steelcrack 512x512는 resize/pad 없이 이 조건을 만족할 수 없으므로 새 학습의
camera-native 크랙 양성 GT로 사용하지 않습니다. 기존 `for model test*` 사진은
정상 음성 데이터이며 크랙 양성을 대신할 수 없습니다.

## CrackSeg9k V4 공개 양성 데이터

현재 받은 공식 V4 압축 해제본은 9,159장의 400x400 이미지이며 Vol1과 Vol2가 모두
필요합니다.

- DOI: https://doi.org/10.7910/DVN/EGIEBY
- 공식 코드: https://github.com/Dhananjay42/crackseg9k
- `Final-Dataset-Vol1.zip` MD5: `0ee4b33617db30612184a2700116ba4d`
- `Final-Dataset-Vol2.zip` MD5: `d52bccf41c081d74fe50c9feec48ca39`

ZIP이 있다면 아래 폴더에 그대로 두어도 되고, 현재처럼 저장소 루트에
`Final-Dataset-Vol1`, `Final-Dataset-Vol2`를 압축 해제해 두어도 자동 탐색합니다.

```text
data_training/realtime_multitask_5ch_hrseg_v1/raw/crackseg9k_v4/downloads/
```

그 다음 체크섬 검사, 안전한 병합 압축 해제, `JPEGImages`/`SegmentationClass`/`ImageSets`
쌍 검사를 한 번에 실행합니다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ROOT\data_training\realtime_multitask_5ch_hrseg_v1\run_hrseg_v1.ps1" `
  -Task prepare-crackseg
```

결과는 `metrics/crackseg9k_v4_audit.json`에 저장됩니다. 이 데이터는 크랙 head의 공개
양성 supervision으로 사용할 수 있지만 400x400이므로 실제 1280x720 카메라 validation을
대체하지 않습니다. 압축 구조 감사 후 원본 비율을 유지하는 무패딩 1280x240 panel
생성 규칙을 고정하고 manifest에 `source=crackseg9k_v4`, `synthetic=true`로 기록합니다.

검사가 끝난 뒤 패딩/리사이즈 없이 네 개의 400x400 원본에서 320x240 crop을 뽑아
가로로 연결한 1280x240 학습 panel을 생성합니다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ROOT\data_training\realtime_multitask_5ch_hrseg_v1\run_hrseg_v1.ps1" `
  -Task build-crackseg-panels
```

생성된 bootstrap 전용 config는
`configs/mnv2_os8_crackseg9k_bootstrap_w1280_h240.yaml`입니다.

## CrackSeg9k bootstrap 실행

현재 데이터 준비와 teacher cache가 완료됐으므로 다음 명령부터 실행합니다.

```powershell
$ROOT = $env:RAIL_ROBOT_ROOT
$RUN = "$ROOT\data_training\realtime_multitask_5ch_hrseg_v1\run_hrseg_v1.ps1"
$CFG = "$ROOT\data_training\realtime_multitask_5ch_hrseg_v1\configs\mnv2_os8_crackseg9k_bootstrap_w1280_h240.yaml"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUN `
  -Task train -Config $CFG -Stage crack_bootstrap `
  -RunName crackseg9k-v4-bootstrap-seed17 `
  -Epochs 40 -Batch 4 -Accumulate 2 -Workers 4 -Seed 17
```

이 결과는 공개 도로/콘크리트 크랙 데이터로 만든 개발용 bootstrap입니다. 실제 시연
카메라에서 찍은 독립 사진 평가를 통과하기 전에는 최종 모델로 확정하거나 ONNX로
내보내지 않습니다.

## 현재 demo 검증 및 ONNX

epoch 40 후보는 공개 CrackSeg9k validation과 별도 시연 정상사진 검사를 통과했습니다.
고정 운용점은 crack 채널 raw logit `>= 1.0986123`(확률 `0.75`), 8방향 최소 연결성분
`1024px`입니다. 출력 ONNX는 다음과 같습니다.

```text
output/models/realtime_w1280/realtime-rust-crack-mnv2-os8-dualhead-w1280-h240-crackseg9k-demo-fp32.onnx
```

이 파일은 5채널 raw logit만 출력하므로 임계값과 연결성분 처리는 Jetson 실행 코드에서
적용합니다. 시연 정상 v7 221장 오탐은 0장이지만 카메라-native 크랙 양성 GT가 없으므로
실제 철도 환경 정확도를 의미하지는 않습니다.

## 지금 실행할 감사 명령

```powershell
$ROOT = $env:RAIL_ROBOT_ROOT
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ROOT\data_training\realtime_multitask_5ch_hrseg_v1\run_hrseg_v1.ps1" `
  -Task audit -AllowBlocked
```

결과는 `metrics/readiness_audit.json`에 저장됩니다. blocker가 남아 있으면 `train`과
`cache`는 자동으로 거부됩니다.

## 데이터가 준비된 뒤 순서

먼저 `manifests/train.csv`, `manifests/val.csv`를 채우고 config의 모든 `null`을
검토·확정한 뒤 `status.training_authorized: true`로 바꿉니다. split/group을
임의로 만들지 말고 실제 촬영 단위로 작성해야 합니다.

```powershell
$ROOT = $env:RAIL_ROBOT_ROOT
$RUN = "$ROOT\data_training\realtime_multitask_5ch_hrseg_v1\run_hrseg_v1.ps1"

# 1) 감사 통과
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUN -Task audit

# 2) 두 teacher logit을 FP32로 고정 cache
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUN -Task cache

# 3) Branch A: exact rust checkpoint에서 crack head bootstrap
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUN -Task train `
  -Stage crack_bootstrap -RunName b32-crack-seed17 -Epochs 40 -Batch 2 -Accumulate 4 -Workers 4 -Seed 17

# 4) 제한적 joint fine-tune (이전 best로 초기화; resume가 아님)
$INIT = "$ROOT\data_training\realtime_multitask_5ch_hrseg_v1\runs\b32-crack-seed17\weights\best.pt"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUN -Task train `
  -Stage joint -RunName b32-joint-seed17 -Initialize $INIT -Epochs 30 -Batch 2 -Accumulate 4 -Workers 4 -Seed 17
```

중단 후 같은 stage/run을 이어갈 때만 `-Resume ...\weights\last.pt`를 사용합니다.
config, train/val manifest, teacher cache SHA가 하나라도 달라지면 resume는 거부됩니다.

## ONNX

데이터와 무관한 TensorRT parser/build 확인용 구조 ONNX는 다음처럼 만듭니다.
이 파일은 정확도 평가나 제어에 사용하면 안 됩니다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RUN -Task structure-export
```

독립 locked test를 통과한 뒤에만 `status.locked_test_passed`와
`status.final_export_authorized`를 `true`로 승인하고, 그 최종 `best.pt`를
`-Task export`로 FP32 ONNX로 변환합니다. TensorRT FP16 engine은 최종 Jetson
자체에서 생성해야 합니다.

## 공식 출처

- [HrSegNet 공식 저장소](https://github.com/CHDyshli/HrSegNet4CrackSegmentation)
- [공식 HrSegNet-B32 체크포인트](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EVaZjUC9tVNMoMkbNOdmemEBh6xPEBUzo2-0ddjGl3bfRQ?e=MWs6Z9)
- [CrackSeg9k V4 데이터셋](https://doi.org/10.7910/DVN/EGIEBY)
- [CrackSeg9k 공식 저장소](https://github.com/Dhananjay42/crackseg9k)
