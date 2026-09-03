import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("Firebase 제출 파일과 환경변수 기반 설정이 유지된다", async () => {
  const [clientConfig, viteConfig, envExample, firebaseConfig] = await Promise.all([
    read("firebase-client/firebase.ts"),
    read("firebase.vite.config.ts"),
    read(".env.example"),
    read("firebase.json"),
  ]);

  for (const name of [
    "VITE_FIREBASE_API_KEY",
    "VITE_FIREBASE_AUTH_DOMAIN",
    "VITE_FIREBASE_PROJECT_ID",
    "VITE_FIREBASE_STORAGE_BUCKET",
    "VITE_FIREBASE_MESSAGING_SENDER_ID",
    "VITE_FIREBASE_APP_ID",
  ]) {
    assert.match(clientConfig, new RegExp(`import\\.meta\\.env\\.${name}`));
    assert.match(envExample, new RegExp(`^${name}=`, "m"));
  }

  assert.match(viteConfig, /envDir:\s*fileURLToPath/);
  assert.doesNotMatch(clientConfig, /AIza[0-9A-Za-z_-]{20,}/);
  assert.equal(JSON.parse(firebaseConfig).firestore.rules, "firestore.rules");
});

test("Firebase 관리자와 Jetson 업로더 권한이 분리된다", async () => {
  const [firestoreRules, storageRules, authControl] = await Promise.all([
    read("firestore.rules"),
    read("storage.rules"),
    read("firebase-client/AdminAuthControl.tsx"),
  ]);

  assert.match(firestoreRules, /admin_users\/\$\(request\.auth\.uid\)/);
  assert.match(firestoreRules, /match \/admin_users\/\{userId\}[\s\S]*allow write: if false;/);
  assert.match(firestoreRules, /device_uploaders\/\$\(request\.auth\.uid\)/);
  assert.match(firestoreRules, /match \/inspection_exports\/\{runId\}/);
  assert.match(firestoreRules, /resource\.data\.uploader_uid == request\.auth\.uid/);
  assert.match(storageRules, /firestore\.exists\(/);
  assert.match(storageRules, /device_uploaders\/\$\(request\.auth\.uid\)/);
  assert.match(storageRules, /match \/dashboard\/media\/\{runId\}\/\{fileName\}/);
  assert.match(storageRules, /match \/captures\/\{allPaths=\*\*\}/);
  assert.match(storageRules, /match \/dashboard\/manifests\/\{fileName\}/);
  assert.match(storageRules, /request\.resource\.metadata\.uploaderUid == request\.auth\.uid/);
  assert.match(storageRules, /resource == null/);
  assert.match(storageRules, /resource\.metadata\.uploaderUid == request\.auth\.uid/);
  assert.match(authControl, /doc\(db, "admin_users", nextUser\.uid\)/);
  assert.doesNotMatch(`${firestoreRules}\n${storageRules}`, /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/);
});

test("Firebase 미설정 빌드는 로컬 fetch 대신 데모 화면으로 고정된다", async () => {
  const dashboard = await read("firebase-client/FirebaseDashboard.tsx");
  assert.match(dashboard, /if \(!firebaseConfigured\)/);
  assert.match(dashboard, /remoteState="demo"/);
  assert.match(dashboard, /Firebase 환경변수가 설정되지 않아 예시 데이터를 표시합니다/);
});

test("Firebase 비로그인·권한 확인 전 화면도 로컬 JSON을 폴링하지 않는다", async () => {
  const dashboard = await read("firebase-client/FirebaseDashboard.tsx");
  assert.match(dashboard, /remoteDocument=\{isAdmin \? serverDocument : null\}/);
  assert.match(dashboard, /remoteState=\{isAdmin \? serverState : "demo"\}/);
  assert.match(dashboard, /관리자 로그인 후 Firebase 검사 기록을 불러옵니다/);
});
