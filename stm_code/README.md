# STM32 펌웨어

`code/encoder/encoder.ioc`는 STM32F103RB 기반 CubeMX 설정이고, `Core/`에 주행·encoder·UART·FRONT/SIDE cleaner/pump 제어가 있습니다. CubeIDE 빌드에 필요한 STM32F1 HAL과 CMSIS Core/Device 파일만 포함하고 Debug 산출물, EWARM 프로젝트와 사용하지 않는 CMSIS DSP/NN/RTOS는 제외했습니다.

## 빌드

1. STM32CubeIDE에서 `code/encoder/.project`를 import하거나 `encoder.ioc`를 엽니다.
2. STM32CubeF1 패키지와 toolchain이 준비됐는지 확인합니다.
3. 코드를 재생성할 경우 `USER CODE BEGIN/END` 구역이 보존됐는지 diff로 확인합니다.
4. Debug/Release를 clean build하고 NUCLEO-F103RB에 flash합니다.
5. 모터드라이버 전원을 끈 상태에서 ST-LINK VCP 115200 8-N-1 UART부터 시험합니다.

UART 메시지는 ASCII + CRLF입니다. 현재 protocol에는 version, sequence, CRC와 명령별 ACK가 없으므로 Jetson 코드와 이 펌웨어를 같은 제출 커밋 기준으로 사용해야 합니다.

## 클리너 PWM

TIM1은 `ARR=3599`이므로 3600 count가 한 주기입니다.

| compare | 실제 duty |
| ---: | ---: |
| 1200 | 33.3% |
| 2000 | 55.6% |
| 0 | OFF |

FRONT/SIDE 핀 매핑은 제출 소스 기준입니다. 실제 연결 전 방향 핀과 TIM1 채널을 멀티미터·오실로스코프로 대조하십시오. 미구현 안전 인터록은 [안전 문서](../docs/SAFETY_AND_LIMITATIONS.md)에 있습니다.
