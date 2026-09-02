import { useCallback, useEffect, useRef, useState } from "react";
import type { InspectionExport } from "../app/data-contract";
import { InspectionDashboard } from "../app/InspectionDashboard";
import { AdminAuthControl } from "./AdminAuthControl";
import { DataImportControl } from "./DataImportControl";
import { loadServerInspectionData } from "./firebase-data";
import { firebaseConfigured } from "./firebase";

function ConfiguredFirebaseDashboard() {
  const [isAdmin, setIsAdmin] = useState(false);
  const [serverDocument, setServerDocument] = useState<InspectionExport | null>(null);
  const [serverState, setServerState] = useState<"loading" | "live" | "demo" | "error">("loading");
  const [serverMessage, setServerMessage] = useState<string | null>(null);
  const objectUrlsRef = useRef<string[]>([]);

  const replaceObjectUrls = (nextUrls: string[]) => {
    objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrlsRef.current = nextUrls;
  };

  const refreshServerData = useCallback(async () => {
    setServerState("loading");
    setServerMessage("Firestore에서 점검 기록을 불러오는 중입니다.");
    try {
      const result = await loadServerInspectionData();
      replaceObjectUrls(result.objectUrls);
      setServerDocument(result.document);
      setServerState(result.document ? "live" : "demo");
      setServerMessage(result.document ? null : "서버에 저장된 점검 기록이 없어 예시 데이터를 표시합니다.");
    } catch (error: unknown) {
      replaceObjectUrls([]);
      setServerDocument(null);
      setServerState("error");
      setServerMessage(`서버 데이터를 읽지 못했습니다: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, []);

  useEffect(() => {
    if (isAdmin) void refreshServerData();
    else {
      replaceObjectUrls([]);
      setServerDocument(null);
    }
  }, [isAdmin, refreshServerData]);

  useEffect(() => () => replaceObjectUrls([]), []);

  return (
    <InspectionDashboard
      remoteDocument={isAdmin ? serverDocument : null}
      remoteState={isAdmin ? serverState : "demo"}
      remoteMessage={isAdmin ? serverMessage : "관리자 로그인 후 Firebase 검사 기록을 불러옵니다."}
      adminAuthControl={(
        <AdminAuthControl onAdminChange={setIsAdmin}>
          {isAdmin ? <DataImportControl onImported={refreshServerData} /> : null}
        </AdminAuthControl>
      )}
    />
  );
}

export function FirebaseDashboard() {
  if (!firebaseConfigured) {
    return (
      <InspectionDashboard
        remoteState="demo"
        remoteMessage="Firebase 환경변수가 설정되지 않아 예시 데이터를 표시합니다."
      />
    );
  }
  return <ConfiguredFirebaseDashboard />;
}
