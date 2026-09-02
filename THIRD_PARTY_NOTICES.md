# Third-party notices

프로젝트가 참조하는 외부 모델·데이터·라이브러리는 각 저작권자와 라이선스의 적용을 받습니다. 이 저장소는 원본 데이터셋과 외부 checkpoint/가중치를 재배포하지 않습니다.

| 구성요소 | 출처 | 저장소 처리 |
| --- | --- | --- |
| Virginia Tech Corrosion Condition State Dataset / trained model | [dataset](https://data.lib.vt.edu/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663), [model](https://data.lib.vt.edu/articles/code/Trained_Model_for_the_Semantic_Segmentation_of_Corrosion_Condition_States/16628668) | 출처·SHA만 기록, 바이너리 제외 |
| HrSegNet | [CHDyshli/HrSegNet4CrackSegmentation](https://github.com/CHDyshli/HrSegNet4CrackSegmentation) | 공식 모델 기반 변환·검증 코드만 포함, checkpoint 제외 |
| CrackSeg9k | 각 학습 프로젝트 README의 DOI/공식 출처 | 여러 원 데이터가 결합된 자료이므로 이미지 제외 |
| Ultralytics | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 설치·호출용 프로젝트 코드만 포함하고 패키지 소스/가중치는 제외. 공식 공개판은 AGPL-3.0이며, 그 조건을 적용하지 않는 용도에는 별도 Enterprise 라이선스가 필요함 |
| PyTorch, torchvision, ONNX, ONNX Runtime | 각 패키지 공식 배포처 | 환경·호출 코드만 포함 |
| Firebase JavaScript SDK | [firebase/firebase-js-sdk](https://github.com/firebase/firebase-js-sdk), Apache-2.0 | npm 의존성과 연동 코드만 포함하고 자격증명·프로젝트 식별자는 제외 |
| STM32CubeF1 HAL / CMSIS | STMicroelectronics / Arm | 사용한 소스와 함께 `Drivers/**/LICENSE.txt` 보존 |
| 대시보드 데모 이미지 | `dashboard/public/demo/IMAGE_SOURCES.md` | 표시·저작자·라이선스 고지 보존 |

세부 학습 자산 표는 [`data_training/THIRD_PARTY_NOTICES.md`](data_training/THIRD_PARTY_NOTICES.md)를 참고하십시오. 팀 원저작 코드의 공개 범위는 루트 [`LICENSE`](LICENSE)에 적었으며, 이는 외부 구성요소의 라이선스를 변경하지 않습니다.
