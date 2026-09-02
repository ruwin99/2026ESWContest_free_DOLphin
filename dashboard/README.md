# 검사 결과 대시보드

Jetson이 저장한 녹·크랙 검사 JSON/PNG를 날짜, 크레인과 Rail Section별로 조회하는 읽기 전용 화면입니다. 모터나 펌프 제어에는 사용하지 않습니다.

## 주요 기능

- 초기 검사와 재검사 결과 비교
- SIDE/TOP 카메라 결과 구분
- 녹 점유율, 크랙 후보와 마스크 표시
- 청소 전후 절대 감소량과 상대 개선율 표시
- 미완료·오류 기록을 `감지 없음`이 아니라 별도 상태로 표시
- 로컬 일정 플래너 제공

실제 검사 자료가 없으면 출처가 표시된 데모 자료를 사용합니다. 데모 사진은 모델 출력이 아니며 출처는 [`public/demo/IMAGE_SOURCES.md`](public/demo/IMAGE_SOURCES.md)에 있습니다.

## 실행

저장소의 `dashboard` 폴더에서 실행합니다.

```powershell
npm ci
npm run data:sync
npm run dev
```

기본 주소는 `http://localhost:3000`입니다. Codex 번들 Node를 사용하는 PC에서는 아래 명령도 사용할 수 있습니다.

```powershell
.\dashboard.cmd sync
.\dashboard.cmd start
```

다른 위치의 검사 결과를 가져올 때:

```powershell
.\dashboard.cmd sync --source "D:\copied_outputs\dashboard"
```

## 데이터 흐름

```text
code/outputs/dashboard/
├─ runs/<report_key>.json
└─ media/<run_id>/*_rust.png, *_crack.png
               │
               ▼ npm run data:sync
dashboard/public/data/inspection-export.json + assets/
```

원본 JPEG는 웹 공개 폴더로 복사하지 않습니다. 데이터 계약은 `runs`, `model_provenance`, `captures`, `analyses`, `artifacts`, `run_summaries` 배열로 구성되며 비율은 0~1 실수로 저장합니다. 누락된 과거 `camera_role`은 호환을 위해 `side`로 처리합니다.

## 검사

```powershell
npm run typecheck
npm test
npm run build
```

`db/schema.ts`와 `drizzle/*.sql`에는 향후 D1 연결을 위한 스키마가 있습니다. 현재 D1/R2 binding은 비활성 상태이며 일정은 브라우저 로컬 저장소에만 보관됩니다.

검사 사진에는 사람·차량번호·설비 라벨이 포함될 수 있습니다. 공개 배포 전 접근제어, 보존기간과 삭제 정책을 정하고 자격증명을 소스나 브라우저에 넣지 마십시오.
