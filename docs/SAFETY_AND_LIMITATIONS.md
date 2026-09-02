# 안전 및 시스템 한계

## 사용 범위

현재 소프트웨어는 전원 차단이 가능한 축소 레일 테스트베드의 연구·시연용입니다. 카메라 기반 녹·균열 결과는 표면 후보를 찾는 보조 정보이며 비파괴검사, 구조 안전진단 또는 잔존수명 평가를 대체하지 않습니다.

실제 크레인이나 가동 설비 시험은 시설관리자의 서면 승인, 운행 중지·잠금표찰, 위험성평가, 보호구, 감시자, 물리 비상정지와 안전거리 확보 없이 수행하면 안 됩니다.

## 현재 fail-safe와 미완료 항목

구현된 보호 동작:

- 모델·I/O·SHA 또는 첫 추론 검증 실패 시 UART 시작 전 중단
- 실시간 결과 준비 부족, 오류 또는 stale 판단 시 cleaner/pump OFF
- cleaner/pump 명령을 주기적으로 갱신하고 STM32에서 제한시간 초과 시 OFF
- 캡처 worker 종료 확인 전 상태 전환 제한

추가 검증·구현이 필요한 항목:

- 주행 모터 전용 heartbeat, 이동 timeout과 원격 비상 `STOP`
- UART protocol version, sequence, CRC와 명령별 ACK
- `CAPTURE_OK`의 Rail Section/encoder count 식별
- 독립 hardware watchdog(IWDG/WWDG) 시험
- 홈·끝단 limit, 물리 E-stop, 저전압·과전류 감시
- 보조 encoder(TIM4) 기반 상호검증, 단선과 바퀴 슬립 검출
- `REALTIME_READY` handshake 이후 주행 시작

## PWM 표기

STM32 TIM1은 ARR=3599이며 클리너 compare=1200/2000은 실제 duty 약 33.3%/55.6%입니다. Jetson/STM32 명령도 이 실제 비율을 사용합니다. 펌프 compare=3000은 약 83.3%입니다. 실물 구동계의 최종 전기·기계 출력은 오실로스코프와 회전수·유량으로 다시 확인해야 합니다.

## 비전 한계

- 조명, 반사, 레일 재질·도장색, 카메라 노출·초점, 렌즈 오염과 시야 밖 물체가 결과를 바꿀 수 있습니다.
- hard negative는 관측한 정상 환경의 오탐을 줄이지만 모든 현장 배경을 보장하지 않습니다.
- polygon/고정 ROI는 레일 위치가 고정된 시연에는 유효하지만 현장에서는 rail localization과 ROI 이탈 감지가 별도로 필요합니다.
- crack ratio는 해당 ROI에서 후처리 후 양성 mask가 차지하는 픽셀 비율이지 균열의 물리 폭·깊이 또는 위험도 자체가 아닙니다.
- 화면 FPS와 모델 단독 지연은 다릅니다. 제어 지연은 camera read, scheduler, inference, history window와 UART 갱신까지 포함해 측정해야 합니다.
