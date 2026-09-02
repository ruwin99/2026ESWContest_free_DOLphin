"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  buildRunViews,
  type AnalysisIssueView,
  type CameraRole,
  type DefectType,
  type DetectionView,
  type InspectionExport,
} from "./data-contract";
import { demoExport } from "./demo-export";
import { PortPlanner, type InspectionDayState } from "./PortPlanner";

const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
const EMPTY_EXPORT: InspectionExport = {
  schema_version: 1,
  exported_at_utc: "",
  runs: [],
  model_provenance: [],
  captures: [],
  analyses: [],
  artifacts: [],
  run_summaries: [],
};

const phaseLabel = (phase: DetectionView["phase"]) => (
  phase === "initial" ? "청소 전" : phase === "rescan" ? "청소 후" : "수동"
);
const cameraRoleLabel = (role: CameraRole) => (role === "top" ? "윗면" : "옆면");
const DATA_REFRESH_INTERVAL_MS = 5_000;

type DashboardView = "overview" | "inspections" | "planner";

const VIEW_META: Record<DashboardView, { label: string; eyebrow: string; icon: string; description: string }> = {
  overview: {
    label: "종합 현황",
    eyebrow: "OPERATIONS OVERVIEW",
    icon: "◫",
    description: "오늘의 레일 점검 상태와 청소 전후 변화를 빠르게 확인합니다.",
  },
  inspections: {
    label: "검출 기록",
    eyebrow: "SURFACE INSPECTION",
    icon: "◎",
    description: "레일 구간별 녹·크랙 위치와 마스크 사진을 분리해 확인합니다.",
  },
  planner: {
    label: "항만 플래너",
    eyebrow: "PORT SCHEDULE",
    icon: "▦",
    description: "항만 작업, 점검, 유지보수 일정을 달력에서 계획합니다.",
  },
};

const isInspectionExport = (value: unknown): value is InspectionExport => {
  if (value == null || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return record.schema_version === 1
    && ["runs", "model_provenance", "captures", "analyses", "artifacts", "run_summaries"]
      .every((key) => Array.isArray(record[key]));
};

function MaskPreview({ detection }: { detection: DetectionView }) {
  const isDemoPhoto = detection.maskUrl?.startsWith("/demo/") ?? false;
  return (
    <div className={`mask-preview ${isDemoPhoto ? "demo-photo" : ""}`}>
      {detection.maskUrl ? (
        // Object-storage URLs are already-sized inspection artifacts and must not be transformed.
        // eslint-disable-next-line @next/next/no-img-element
        <img src={detection.maskUrl} alt={`${detection.type === "rust" ? "녹" : "크랙"} ${isDemoPhoto ? "데모 사진" : "마스크 미리보기"}`} />
      ) : (
        <div className="mask-missing">마스크 파일 경로<br />{detection.maskObjectKey ?? "없음"}</div>
      )}
    </div>
  );
}

function AnalysisIssue({ issue }: { issue: AnalysisIssueView }) {
  return (
    <article className="analysis-issue">
      <div><strong>{cameraRoleLabel(issue.cameraRole)} · {issue.zone == null ? "수동 캡처" : `${issue.zone}번 레일 구간`}</strong><span>{issue.capturedAt}</span></div>
      <p>{issue.message}</p>
    </article>
  );
}

function RailSegmentMap({ detections, railSectionTarget }: { detections: DetectionView[]; railSectionTarget: number }) {
  const segments = Array.from({ length: railSectionTarget }, (_, index) => index + 1).map((segment) => ({
    segment,
    rust: detections.filter((item) => item.zone === segment && item.type === "rust").length,
    crack: detections.filter((item) => item.zone === segment && item.type === "crack").length,
  }));

  return (
    <section className="rail-segment-map" aria-label="전체 레일 구간별 검출 위치">
      <div className="rail-map-heading">
        <div><span className="tiny-label">RAIL POSITION MAP</span><strong>전체 레일 구간 위치</strong></div>
        <div className="rail-map-legend"><span><i className="rust-point" />녹</span><span><i className="crack-point" />크랙</span></div>
      </div>
      <div
        className="rail-track"
        role="list"
        style={{
          gridTemplateColumns: `repeat(${railSectionTarget}, minmax(90px, 1fr))`,
          minWidth: `${railSectionTarget * 90}px`,
        }}
      >
        {segments.map(({ segment, rust, crack }) => (
          <div className={`rail-segment ${rust || crack ? "detected" : ""}`} role="listitem" key={segment} aria-label={`${segment}번 레일 구간, 녹 ${rust}건, 크랙 ${crack}건`}>
            <div className="rail-marker-row" aria-hidden="true">
              {rust ? <span className="rail-marker rust-point">{rust}</span> : null}
              {crack ? <span className="rail-marker crack-point">{crack}</span> : null}
            </div>
            <strong>{segment}번 레일 구간</strong>
            <small>{rust || crack ? [rust ? `녹 ${rust}` : "", crack ? `크랙 ${crack}` : ""].filter(Boolean).join(" · ") : "검출 없음"}</small>
          </div>
        ))}
      </div>
      <p className="rail-map-note">초기 검사와 재검사에서 감지된 위치를 {railSectionTarget}개 Rail Section 기준으로 합산해 표시합니다.</p>
    </section>
  );
}

export function InspectionDashboard({
  initialDocument = EMPTY_EXPORT,
  adminAuthControl,
  remoteDocument,
  remoteState,
  remoteMessage,
}: {
  initialDocument?: InspectionExport;
  adminAuthControl?: ReactNode;
  remoteDocument?: InspectionExport | null;
  remoteState?: "loading" | "live" | "demo" | "error";
  remoteMessage?: string | null;
}) {
  const [document, setDocument] = useState(initialDocument);
  const [dataState, setDataState] = useState<"loading" | "live" | "demo" | "empty" | "error">("loading");
  const [dataMessage, setDataMessage] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<DashboardView>("overview");
  const [selectedCraneId, setSelectedCraneId] = useState("");
  const runs = useMemo(() => buildRunViews(document), [document]);
  const craneOptions = useMemo(
    () => [...new Map(runs.map((run) => [run.craneId, { id: run.craneId, label: run.craneLabel }])).values()],
    [runs],
  );
  const availableDates = useMemo(
    () => [...new Set(runs.map((run) => run.date).filter(Boolean))].sort().reverse(),
    [runs],
  );
  const [selectedDate, setSelectedDate] = useState(availableDates[0] ?? "");
  const [selectedRunId, setSelectedRunId] = useState(runs[0]?.id ?? "");

  useEffect(() => {
    if (remoteDocument !== undefined || remoteState !== undefined) {
      setDocument(remoteDocument ?? demoExport);
      setDataState(remoteState ?? (remoteDocument ? "live" : "demo"));
      setDataMessage(remoteMessage ?? null);
      return;
    }

    const controller = new AbortController();
    const loadInspectionData = () => fetch("/data/inspection-export.json", { cache: "no-store", signal: controller.signal })
      .then(async (response) => {
        if (response.status === 404) {
          setDocument(demoExport);
          setDataState("demo");
          setDataMessage("실제 캡처 JSON이 아직 동기화되지 않아 예시 데이터를 표시합니다.");
          return null;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload: unknown = await response.json();
        if (!isInspectionExport(payload)) throw new Error("지원하지 않는 데이터 계약입니다.");
        if (!payload.runs.length) {
          setDocument(demoExport);
          setDataState("demo");
          setDataMessage("동기화된 실제 점검 기록이 없어 웹 공개 이미지를 사용한 데모 데이터를 표시합니다.");
          return payload;
        }
        setDocument(payload);
        setDataState("live");
        setDataMessage(null);
        return payload;
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setDataState("error");
        setDataMessage(`실제 캡처 데이터를 읽지 못했습니다: ${error instanceof Error ? error.message : String(error)}`);
      });
    void loadInspectionData();
    const refreshTimer = window.setInterval(loadInspectionData, DATA_REFRESH_INTERVAL_MS);
    return () => {
      window.clearInterval(refreshTimer);
      controller.abort();
    };
  }, [remoteDocument, remoteMessage, remoteState]);

  const effectiveCraneId = selectedCraneId || craneOptions[0]?.id || "UNASSIGNED";
  const effectiveDate = selectedDate || availableDates[0] || "";

  const runsForDate = runs.filter((run) => run.date === effectiveDate && run.craneId === effectiveCraneId);
  const selectedRun = runsForDate.find((run) => run.id === selectedRunId) ?? runsForDate[0];
  const changeDate = (date: string) => {
    setSelectedDate(date);
    const firstRun = runs.find((run) => run.date === date && run.craneId === effectiveCraneId);
    if (firstRun) setSelectedRunId(firstRun.id);
  };
  const changeCrane = (craneId: string) => {
    setSelectedCraneId(craneId);
    const firstRun = runs.find((run) => run.craneId === craneId);
    if (firstRun) {
      setSelectedDate(firstRun.date);
      setSelectedRunId(firstRun.id);
    }
  };

  const inspectionDayStates = useMemo(() => {
    const byDate = new Map<string, typeof runs>();
    runs.forEach((run) => byDate.set(run.date, [...(byDate.get(run.date) ?? []), run]));
    return Object.fromEntries([...byDate.entries()].flatMap(([date, dateRuns]) => {
      const hasProblem = dateRuns.some((run) => {
        const resultDetections = run.summaryComplete
          ? run.detections.filter((item) => item.phase === "rescan")
          : run.detections;
        const resultIssues = run.summaryComplete
          ? run.issues.filter((item) => item.phase === "rescan")
          : run.issues;
        return run.status === "failed" || resultIssues.length > 0 || resultDetections.length > 0;
      });
      if (hasProblem) return [[date, "problem" as InspectionDayState]];
      const isClean = dateRuns.some((run) => run.status === "complete" && run.summaryComplete);
      return isClean ? [[date, "clean" as InspectionDayState]] : [];
    }));
  }, [runs]);

  const stateLabel = dataState === "live" ? (remoteDocument !== undefined ? "서버 점검 기록" : "로컬 캡처 기록")
    : dataState === "loading" ? "데이터 확인 중"
      : dataState === "empty" ? "기록 없음"
        : dataState === "error" ? "데이터 오류" : "예시 데이터";

  const activeMeta = VIEW_META[activeView];
  const detectionCount = selectedRun?.detections.length ?? 0;

  return (
    <div className="dashboard-frame">
      <aside className="app-sidebar" aria-label="대시보드 기능 메뉴">
        <div className="sidebar-brand">
          <div className="brand-mark">
            {/* User-provided project identity asset. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/dolphin-mark.png" alt="프로젝트 돌고래 아이콘" />
          </div>
          <div><span>B0441</span><strong>RAIL CARE</strong></div>
        </div>

        <nav className="sidebar-nav">
          {(Object.entries(VIEW_META) as [DashboardView, (typeof VIEW_META)[DashboardView]][]).map(([view, meta]) => (
            <button
              type="button"
              className={activeView === view ? "active" : ""}
              aria-current={activeView === view ? "page" : undefined}
              onClick={() => setActiveView(view)}
              key={view}
            >
              <span className="nav-icon" aria-hidden="true">{meta.icon}</span>
              <span className="nav-copy"><strong>{meta.label}</strong><small>{meta.eyebrow}</small></span>
              {view === "inspections" && detectionCount ? <span className="nav-count">{detectionCount}</span> : null}
            </button>
          ))}
        </nav>

        <div className="sidebar-status">
          <span className={`sidebar-pulse ${dataState}`} />
          <div><small>DATA STATUS</small><strong>{stateLabel}</strong></div>
        </div>
        {adminAuthControl}
        <div className="sidebar-date"><small>SELECTED DATE</small><strong>{effectiveDate ? effectiveDate.replaceAll("-", ".") : "—"}</strong></div>
      </aside>

      <main className="dashboard-shell">
        <header className="topbar">
          <div className="page-heading">
            <p className="eyebrow">{activeMeta.eyebrow}</p>
            <h1>{activeMeta.label}</h1>
            <span>{activeMeta.description}</span>
          </div>
          <div className={`system-state ${dataState}`}><span /> {stateLabel}</div>
        </header>

        {dataMessage ? <div className={`data-notice ${dataState}`}>{dataMessage}</div> : null}

        <div className="view-stage" key={activeView}>
          {activeView === "overview" ? (
            <>
              <section className="hero-grid">
                <div className="hero-copy">
                  <p className="hero-year">2026 SMART PORT OPERATIONS</p>
                  <div className="wave-rule" aria-hidden="true" />
                  <h2>항만 레일을<br /><em>더 스마트하게 관리합니다.</em></h2>
                  <p>필요한 기능만 골라 확인하는 운영 화면입니다. 점검 기록과 일정은 왼쪽 메뉴에서 각각 열 수 있습니다.</p>
                  <div className="hero-actions">
                    <button type="button" onClick={() => setActiveView("inspections")}>검출 기록 열기 <span>→</span></button>
                    <button type="button" className="secondary" onClick={() => setActiveView("planner")}>플래너 열기</button>
                  </div>
                </div>

                <aside className="date-panel" aria-label="현재 점검 요약">
                  <span className="poster-badge">SMART RAIL CARE</span>
                  <div className="date-panel-heading">
                    <div><span className="tiny-label">TODAY&apos;S STATUS</span><strong>선택한 날짜의<br />레일 점검 현황</strong></div>
                    <span className="calendar-glyph" aria-hidden="true">◎</span>
                  </div>
                  <div className="overview-live-stat">
                    <span><i className="rust-point" />녹 검출 <strong>{selectedRun?.detections.filter((item) => item.type === "rust").length ?? 0}</strong></span>
                    <span><i className="crack-point" />크랙 검출 <strong>{selectedRun?.detections.filter((item) => item.type === "crack").length ?? 0}</strong></span>
                  </div>
                  <span className="planner-intro-date">선택일 · {effectiveDate ? effectiveDate.replaceAll("-", ".") : "날짜를 선택하세요"}</span>
                </aside>
              </section>

              {selectedRun ? (
                <>
                  <section className="run-strip">
                    <div><span className="tiny-label">선택한 점검</span><strong>{selectedRun.id}</strong></div>
                    <div className="run-meta"><span>{selectedRun.startedAt} 시작 · {selectedRun.timezone}</span><span className={`status ${selectedRun.status}`}>{selectedRun.status === "complete" ? "임무 완료" : selectedRun.status === "failed" ? "임무 실패" : "진행 중"}</span></div>
                  </section>
                  <section className="metric-grid" aria-label="청소 전후 오염률 요약">
                    <article className="metric-card before"><span>청소 전 오염률</span><strong>{selectedRun.summaryComplete && selectedRun.beforeRatio != null ? percent(selectedRun.beforeRatio) : "집계 중"}</strong><small>초기 Rail Section {selectedRun.initialRailSectionCount}/{selectedRun.railSectionTarget} · 픽셀 가중</small></article>
                    <article className="metric-card after"><span>청소 후 오염률</span><strong>{selectedRun.summaryComplete && selectedRun.afterRatio != null ? percent(selectedRun.afterRatio) : "집계 중"}</strong><small>재검사 Rail Section {selectedRun.rescanRailSectionCount}/{selectedRun.railSectionTarget} · 픽셀 가중</small></article>
                    <article className="metric-card reduction"><span>감소된 정도</span><strong>{selectedRun.reduction == null ? "—" : `${(selectedRun.reduction * 100).toFixed(2)}%p`}</strong><small>{selectedRun.improvement == null ? `${selectedRun.railSectionTarget}+${selectedRun.railSectionTarget} Rail Section 완료 후 계산` : `상대 개선율 ${percent(selectedRun.improvement)}`}</small></article>
                  </section>
                </>
              ) : <section className="no-record">선택한 날짜에는 점검 기록이 없습니다.</section>}
            </>
          ) : null}

          {activeView === "inspections" ? (
            <>
              {dataState === "demo" ? (
                <div className="demo-credit">
                  <strong>데모 이미지 출처</strong>
                  <span>녹: <a href="https://www.piqsels.com/en/public-domain-photo-olsgg" target="_blank" rel="noreferrer">Piqsels · Public Domain</a></span>
                  <span>크랙: <a href="https://www.flickr.com/photos/shaireproductions/3632302149" target="_blank" rel="noreferrer">Cracked Wall by Sherrie Thai</a> · <a href="https://creativecommons.org/licenses/by/2.0/" target="_blank" rel="noreferrer">CC BY 2.0</a></span>
                  <small>모델 추론 결과가 아닌 화면 시험용 예시입니다.</small>
                </div>
              ) : null}
              <section className="crane-selector" aria-label="검출 기록 크레인 선택">
                <div>
                  <span className="tiny-label">CRANE FLEET</span>
                  <strong>검사할 크레인을 선택하세요</strong>
                  <small>등록된 크레인 {craneOptions.length}대 · 선택한 장비의 기록만 표시</small>
                </div>
                <label>
                  <span>크레인</span>
                  <select aria-label="크레인 선택" value={effectiveCraneId} onChange={(event) => changeCrane(event.target.value)}>
                    {craneOptions.map((crane) => <option value={crane.id} key={crane.id}>{crane.label} · {crane.id}</option>)}
                  </select>
                </label>
              </section>
              {selectedRun ? (
                <>
                  <section className="run-strip">
                    <div>
                      <span className="tiny-label">선택한 점검</span>
                      {runsForDate.length > 1 ? (
                        <select aria-label="점검 실행 선택" value={selectedRun.id} onChange={(event) => setSelectedRunId(event.target.value)}>
                          {runsForDate.map((run) => <option value={run.id} key={run.id}>{run.id}</option>)}
                        </select>
                      ) : <strong>{selectedRun.id}</strong>}
                    </div>
                    <div className="run-meta"><span>{selectedRun.startedAt} 시작 · {selectedRun.timezone}</span><span className={`status ${selectedRun.status}`}>{selectedRun.status === "complete" ? "임무 완료" : selectedRun.status === "failed" ? "임무 실패" : "진행 중"}</span></div>
                  </section>
                  {selectedRun.failureReason ? <div className="run-failure"><strong>실패 사유</strong><span>{selectedRun.failureReason}</span></div> : null}
                  <section className="detections-section">
                    <div className="section-heading"><div><p className="section-kicker">DETECTED AREAS</p><h3>검출 구역</h3></div><p>레일 구간 번호는 단계별 촬영 순서를 뜻합니다.</p></div>
                    <RailSegmentMap detections={selectedRun.detections} railSectionTarget={selectedRun.railSectionTarget} />
                    <div className="detection-columns">
                      {(["rust", "crack"] as DefectType[]).map((type) => {
                        const items = selectedRun.detections.filter((detection) => detection.type === type);
                        const issues = selectedRun.issues.filter((issue) => issue.type === type);
                        return (
                          <section className={`detection-column ${type}`} key={type}>
                            <div className="column-heading"><div><span className="defect-dot" /><strong>{type === "rust" ? "녹 감지" : "크랙 감지"}</strong></div><span>{items.length}개 검출 · {issues.length}개 확인 필요</span></div>
                            <div className="detection-list">
                              {items.map((item) => (
                                <article className="detection-card" key={item.id}>
                                  <MaskPreview detection={item} />
                                  <div className="detection-copy"><div><span className="phase-chip">{phaseLabel(item.phase)} · {cameraRoleLabel(item.cameraRole)}</span><span className="time">{item.capturedAt}</span></div><div className="zone-row"><strong>{item.zone == null ? "수동 캡처" : `${item.zone}번 레일 구간`}</strong><span>{percent(item.ratio)}</span></div></div>
                                </article>
                              ))}
                              {issues.map((issue) => <AnalysisIssue issue={issue} key={issue.id} />)}
                              {!items.length && !issues.length ? <div className="empty-card">분석이 정상 완료됐으며 감지된 {type === "rust" ? "녹" : "크랙"}이 없습니다.</div> : null}
                            </div>
                          </section>
                        );
                      })}
                    </div>
                  </section>
                </>
              ) : <section className="no-record">선택한 날짜에는 점검 기록이 없습니다.</section>}
            </>
          ) : null}

          {activeView === "planner" ? <PortPlanner inspectionDayStates={inspectionDayStates} selectedDate={effectiveDate} onSelectDate={changeDate} /> : null}
        </div>

        <footer><span>SMART MARITIME LOGISTICS × ICT · 모델 예측 기반 표면 점검 기록</span><span>크랙 표시는 균열 후보이며 구조 강도·잔존수명 판정이 아닙니다.</span></footer>
      </main>
    </div>
  );
}
