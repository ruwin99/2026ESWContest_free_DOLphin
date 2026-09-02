# 외부 데이터·모델·코드 출처 점검표

이 문서는 출처 기록용이며 법률 자문이 아니다. GitHub에 원본 데이터나 가중치를 재배포하기 전에는 각 공식 페이지의 현재 라이선스와 재배포 조건을 다시 확인한다.

| 자산 | 사용 목적 | 로컬 기록 | 공개본 처리 |
|---|---|---|---|
| Virginia Tech Corrosion Condition State Dataset 및 teacher | 녹 4등급 teacher·학생 학습 | `vt_kd/asset_manifest.json`, 데이터 CC0 기록 | 출처·DOI 표시. checkpoint 재배포 조건은 공식 페이지에서 재확인 |
| HrSegNet-B32 | 크랙 teacher·직접 추론 기준 | `realtime_multitask_5ch_hrseg_v1/README_RUN.md` | 공식 저장소·논문 인용. 소스·가중치 라이선스 확인 전 원본 checkpoint 미포함 |
| CrackSeg9k V4 | 크랙 bootstrap 데이터 | DOI와 공식 저장소 기록 | 데이터 라이선스·인용 문구 확인 전 원본 이미지 미포함 |
| Civil-dataset / BGCrack | 초기 크랙 모델 개발 | `steelcrack/asset_manifest.json`, Apache-2.0 기록 | NOTICE와 저작권 고지 유지, 수정 코드 구분 |
| Roboflow `-ohs3h/2-iemaw/1` | 이물질 양성 데이터 | workspace/project/version 기록 | 프로젝트 소유·다운로드·재배포 권한 확인. 원본 데이터는 기본 미포함 |
| Ultralytics YOLO | 이물질 학습·export 런타임 | 버전 `8.4.122` 기록 | 패키지/가중치는 미포함. 공식 공개판 AGPL-3.0 조건을 준수하거나 적용 가능한 Enterprise 라이선스를 별도로 확보해야 함 |
| PyTorch / torchvision / ONNX / ONNX Runtime | 학습·변환·검증 | requirements·environment 기록 | 각 패키지 LICENSE 파일 또는 공식 링크를 공개 README에 명시 |

## 발표 시 출처 표현

- “공개 데이터와 공식 pretrained checkpoint를 기반으로 프로젝트 환경에 맞게 미세조정·변환·검증하였다.”
- “HrSegNet-B32는 공식 모델을 사용했으며 팀이 수행한 부분은 입출력 계약 검증, ONNX 변환, 후처리 및 시스템 통합이다.”
- “이물질 YOLO는 프로젝트 데이터와 정상 배경 hard negative를 이용해 추가 학습하였다.”

## 공개 금지 기준

- 라이선스가 확인되지 않은 원본 데이터·가중치
- API 키가 포함된 다운로드 URL 또는 로그
- 참가자 개인 경로와 계정 정보
- 제3자가 소유한 전체 저장소를 출처 표시 없이 복사한 코드
