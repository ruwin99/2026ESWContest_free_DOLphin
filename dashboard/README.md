# 검사 결과 대시보드

## 목적

Jetson이 생성한 레일 표면 검사 결과를 날짜, 크레인과 Rail Section별로 보여주는 조회 화면입니다. 녹·크랙 후보를 빠르게 확인하고 청소 전후 결과를 비교하는 용도이며, 모터나 워터펌프 제어에는 연결되지 않습니다.

## 보여주는 정보

- SIDE/TOP 카메라별 녹 점유율과 크랙 후보
- 레일 구간별 검출 위치와 마스크 미리보기
- 초기 검사와 재검사의 절대 감소량·상대 개선율
- 미완료·분석 오류를 `감지 없음`과 구분한 상태
- 브라우저에 저장되는 로컬 일정 플래너

실제 검사 자료가 없으면 출처가 표시된 데모 자료를 사용합니다. 데모 사진은 모델 출력이 아니며 출처는 [`public/demo/IMAGE_SOURCES.md`](public/demo/IMAGE_SOURCES.md)에 있습니다.

## 데이터 흐름

```text
Jetson 검사
  └─ 실행 정보 JSON + 녹·크랙 마스크 PNG
       ├─ 로컬 구성: 동기화된 inspection-export.json과 미리보기 표시
       └─ Firebase 구성(선택): Firestore 기록 + Storage 미리보기 표시
```

데이터 계약은 `runs`, `model_provenance`, `captures`, `analyses`, `artifacts`, `run_summaries` 배열로 구성되며 비율은 0~1 실수로 저장합니다. 누락된 과거 `camera_role`은 `side`로 처리합니다. 목표가 4였던 기존 기록도 그대로 표시하고, 현재 기본 목표 10과 임의로 합치지 않습니다.

## 로컬·Firebase 구성

로컬 구성은 Jetson에서 복사한 JSON과 PNG를 읽으며 Firebase 연결 없이 독립적으로 동작합니다.

Firebase 구성은 선택 경로입니다.

- Authentication: Google 로그인과 사용자 식별
- Firestore: `inspection_exports/{run_id}`에 실행별 검사 기록 저장
- Storage: `dashboard/media/**`에 녹·크랙 미리보기 저장
- 관리자 화면: 승인된 계정의 검사 기록 폴더 업로드와 서버 기록 조회

Firebase 설정이 없거나 로그인하지 않은 경우에는 서버 검사 기록을 요청하지 않고 데모 화면을 표시합니다. Firebase Hosting·Authentication·Firestore·Storage를 실제 프로젝트에 배포한 통합시험은 아직 완료하지 않았습니다.

## 보안과 현재 범위

- 관리자 권한은 Firestore의 `admin_users/{Google Auth UID}` 등록 여부로 확인합니다.
- 관리자 UID 문서는 클라이언트에서 생성하거나 수정할 수 없습니다.
- Firestore와 Storage 보안 규칙이 실제 데이터 접근을 통제합니다.
- 원본 JPEG는 웹 공개 폴더나 Firebase 공개 자산으로 복사하지 않습니다.
- 검사 사진에 사람·차량번호·설비 라벨이 포함될 수 있으므로 실제 배포 전 접근제어, 보존기간과 삭제 정책 검증이 필요합니다.
- `db/schema.ts`와 `drizzle/*.sql`의 D1 스키마는 향후 연동용이며 현재 D1/R2 연결은 비활성 상태입니다.
- 이 화면은 검사 결과 조회·업로드용이며 로봇의 모터·워터펌프 제어와 안전 판단을 수행하지 않습니다.
