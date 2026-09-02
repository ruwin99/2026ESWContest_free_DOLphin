import {
  collection,
  doc,
  getDocs,
  orderBy,
  query,
  serverTimestamp,
  writeBatch,
} from "firebase/firestore";
import { getBlob, ref, uploadBytes } from "firebase/storage";
import type { InspectionExport } from "../app/data-contract";
import { getFirebaseServices } from "./firebase";

type StoredInspectionExport = {
  payload?: InspectionExport;
};

const isInspectionExport = (value: unknown): value is InspectionExport => {
  if (value == null || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return record.schema_version === 1
    && ["runs", "model_provenance", "captures", "analyses", "artifacts", "run_summaries"]
      .every((key) => Array.isArray(record[key]));
};

const safeStoragePath = (path: string) => (
  path.startsWith("dashboard/media/") && !path.includes("..") && !path.startsWith("/")
);

const fileNameFromPath = (path: string) => path.split("/").at(-1) ?? "";

const exportForRun = (document: InspectionExport, runId: string): InspectionExport => {
  const captures = document.captures.filter((capture) => capture.run_id === runId);
  const captureIds = new Set(captures.map((capture) => capture.capture_id));
  return {
    schema_version: document.schema_version,
    exported_at_utc: document.exported_at_utc,
    runs: document.runs.filter((run) => run.run_id === runId),
    model_provenance: document.model_provenance.filter((item) => item.run_id === runId),
    captures,
    analyses: document.analyses.filter((analysis) => captureIds.has(analysis.capture_id)),
    artifacts: document.artifacts
      .filter((artifact) => captureIds.has(artifact.capture_id))
      .map((artifact) => ({ ...artifact, public_url: null })),
    run_summaries: document.run_summaries.filter((summary) => summary.run_id === runId),
  };
};

export async function importInspectionFiles(
  files: File[],
  onProgress: (message: string) => void,
) {
  const { db, storage } = getFirebaseServices();
  const jsonFile = files.find((file) => file.name === "inspection-export.json");
  if (!jsonFile) throw new Error("inspection-export.json 파일을 찾지 못했습니다.");

  const parsed: unknown = JSON.parse(await jsonFile.text());
  if (!isInspectionExport(parsed)) throw new Error("지원하지 않는 검사 데이터 형식입니다.");

  const filesByName = new Map(files.map((file) => [file.name, file]));
  const previewArtifacts = parsed.artifacts.filter((artifact) => (
    (artifact.artifact_type === "rust_preview" || artifact.artifact_type === "crack_preview")
      && artifact.public_url
      && safeStoragePath(artifact.object_key)
  ));

  const missingFiles = previewArtifacts.filter(
    (artifact) => !filesByName.has(fileNameFromPath(artifact.object_key)),
  );
  if (missingFiles.length) {
    throw new Error(`마스크 이미지 ${missingFiles.length}개를 찾지 못했습니다.`);
  }

  for (const [index, artifact] of previewArtifacts.entries()) {
    const file = filesByName.get(fileNameFromPath(artifact.object_key));
    if (!file) continue;
    onProgress(`마스크 이미지 업로드 ${index + 1}/${previewArtifacts.length}`);
    await uploadBytes(ref(storage, artifact.object_key), file, {
      contentType: artifact.media_type || file.type,
      customMetadata: {
        captureId: artifact.capture_id,
        artifactType: artifact.artifact_type,
        source: "dashboard-import",
      },
    });
  }

  onProgress("검사 기록을 Firestore 문서로 변환 중");
  const batch = writeBatch(db);
  parsed.runs.forEach((run) => {
    batch.set(doc(db, "inspection_exports", run.run_id), {
      payload: exportForRun(parsed, run.run_id),
      source: "dashboard-import",
      imported_at: serverTimestamp(),
    });
  });
  await batch.commit();

  return {
    runCount: parsed.runs.length,
    previewCount: previewArtifacts.length,
  };
}

export async function loadServerInspectionData() {
  const { db, storage } = getFirebaseServices();
  const exportsQuery = query(collection(db, "inspection_exports"), orderBy("imported_at", "desc"));
  const snapshot = await getDocs(exportsQuery);
  const exports = snapshot.docs
    .map((item) => (item.data() as StoredInspectionExport).payload)
    .filter((item): item is InspectionExport => isInspectionExport(item));

  if (!exports.length) return { document: null, objectUrls: [] as string[] };

  const combined: InspectionExport = {
    schema_version: 1,
    exported_at_utc: exports[0].exported_at_utc,
    runs: exports.flatMap((item) => item.runs),
    model_provenance: exports.flatMap((item) => item.model_provenance),
    captures: exports.flatMap((item) => item.captures),
    analyses: exports.flatMap((item) => item.analyses),
    artifacts: exports.flatMap((item) => item.artifacts),
    run_summaries: exports.flatMap((item) => item.run_summaries),
  };

  const objectUrls: string[] = [];
  const artifacts = await Promise.all(combined.artifacts.map(async (artifact) => {
    if (artifact.artifact_type !== "rust_preview" && artifact.artifact_type !== "crack_preview") {
      return artifact;
    }
    if (!safeStoragePath(artifact.object_key)) return artifact;
    try {
      const blob = await getBlob(ref(storage, artifact.object_key));
      const objectUrl = URL.createObjectURL(blob);
      objectUrls.push(objectUrl);
      return { ...artifact, public_url: objectUrl };
    } catch {
      return artifact;
    }
  }));

  return { document: { ...combined, artifacts }, objectUrls };
}
