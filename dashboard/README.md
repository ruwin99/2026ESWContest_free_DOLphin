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

원본 JPEG는 웹 공개 폴더로 복사하지 않습니다. 데이터 계약은 `runs`, `model_provenance`, `captures`, `analyses`, `artifacts`, `run_summaries` 배열로 구성되며 비율은 0~1 실수로 저장합니다. 누락된 과거 `camera_role`은 호환을 위해 `side`로 처리합니다. 목표가 4였던 기존 기록도 그대로 표시하고, 현재 기본 목표 10과 임의로 합치지 않습니다.

## Firebase 배포(선택)

Firebase 전용 화면은 Google 로그인 후 Firestore의 검사 JSON과 Storage의 마스크 이미지를 읽고, 승인된 관리자만 기존 검사 폴더를 업로드합니다. 기본 `npm run dev`는 Firebase를 불러오지 않으므로 기존 로컬 JSON 방식과 독립적으로 동작합니다.

```powershell
Copy-Item .env.example .env.local
npm install --global firebase-tools
firebase login
npm run firebase:build
firebase use --add
firebase deploy --only hosting,firestore:rules,storage
```

`.env.local`에는 Firebase Console의 Web App 설정을 넣고 저장소에는 커밋하지 않습니다. Google 인증 제공자를 활성화한 뒤 Firestore에 `admin_users/{Google Auth UID}` 문서를 Firebase Console 또는 신뢰된 Admin SDK로 등록해야 합니다. 실제 UID는 코드에 넣지 않으며 이 문서는 클라이언트에서 생성·수정할 수 없습니다. Storage 규칙이 이 Firestore 문서를 조회하므로 첫 배포에서는 Firebase 안내에 따라 두 서비스 간 권한 연결을 활성화해야 할 수 있습니다.

`VITE_FIREBASE_*` Web App 설정은 빌드된 브라우저 코드에서 확인될 수 있으므로 권한 판단에 사용하지 않습니다. 실제 접근은 보안 규칙이 통제하며 서비스 계정 JSON과 Admin SDK 개인 키는 브라우저나 저장소에 넣지 않습니다.

Cloud Storage를 사용하기 전에 [현재 요금제 요구사항](https://firebase.google.com/docs/storage/web/start)을 확인하고 Blaze 요금제 전환이 필요하면 예산 알림을 먼저 설정합니다. 로컬 Google 로그인 시험에서는 Firebase Authentication의 Authorized domains에 `localhost`가 없으면 직접 추가합니다.

- Firestore: `inspection_exports/{run_id}`에 실행별 데이터 계약 저장
- Storage: `dashboard/media/**`에 녹·크랙 미리보기 저장
- 미설정 또는 비로그인 상태: 기존 로컬 JSON/데모 화면 사용

관리자 화면의 `검사 데이터 업로드`에서는 `npm run data:sync`가 만든 `public/data` 폴더를 선택합니다. 이 폴더에 `inspection-export.json`과 해당 녹·크랙 미리보기 파일이 함께 있어야 합니다.

## 검사

```powershell
npm run typecheck
npm test
npm run build
```

`db/schema.ts`와 `drizzle/*.sql`에는 향후 D1 연결을 위한 스키마가 있습니다. 현재 D1/R2 binding은 비활성 상태이며 일정은 브라우저 로컬 저장소에만 보관됩니다. Firebase는 검사 결과 조회·업로드용이며 모터 제어에는 사용하지 않습니다.

검사 사진에는 사람·차량번호·설비 라벨이 포함될 수 있습니다. 공개 배포 전 접근제어, 보존기간과 삭제 정책을 정하고 자격증명을 소스나 브라우저에 넣지 마십시오.
