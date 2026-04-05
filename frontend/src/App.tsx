import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { api } from "./api";
import { DashboardLayout } from "./DashboardLayout";
import type { JsonRecord, OverviewResponse } from "./types";

const tabs = [
  "Master Control",
  "Data Pipeline",
  "Factor Lab",
  "Training Studio",
  "Testing Console",
  "Risk Lab",
  "Portfolio/Backtest",
  "Execution Simulator",
  "Model Registry",
  "Monitoring"
];

const PANEL_COLLAPSE_STORAGE_KEY = "quant-platform-panel-collapse:workspace";

function loadCollapsedPanelMap(storageKey: string): Record<string, boolean> {
  if (typeof window === "undefined") {
    return {};
  }

  try {
    const stored = window.localStorage.getItem(storageKey);
    if (!stored) {
      return {};
    }

    const parsed = JSON.parse(stored);
    return parsed && typeof parsed === "object"
      ? Object.entries(parsed as Record<string, unknown>).reduce<Record<string, boolean>>((nextPanels, [panelId, value]) => {
          if (Boolean(value)) {
            nextPanels[panelId] = true;
          }
          return nextPanels;
        }, {})
      : {};
  } catch {
    return {};
  }
}

function MetricCard({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value ?? "n/a"}</strong>
    </div>
  );
}

function StatusPill({ value }: { value: string }) {
  const tone = ["running", "completed", "promoted"].includes(value)
    ? "good"
    : ["paused", "queued", "draft"].includes(value)
      ? "muted"
      : ["failed", "rejected", "stopped"].includes(value)
        ? "bad"
        : "info";
  return <span className={`status-pill tone-${tone}`}>{value}</span>;
}

function RequirementPill({ value }: { value: string }) {
  const tone = value === "required" ? "bad" : value === "conditional" ? "info" : "muted";
  return <span className={`status-pill tone-${tone}`}>{value}</span>;
}

function AssessmentPill({ value }: { value: string | null | undefined }) {
  const normalized = String(value ?? "unknown").toLowerCase();
  const tone = ["high", "healthy"].includes(normalized)
    ? "good"
    : ["medium", "warning"].includes(normalized)
      ? "info"
      : ["low", "critical"].includes(normalized)
        ? "bad"
        : "muted";
  return <span className={`status-pill tone-${tone}`}>{normalized}</span>;
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <p>{body}</p>
    </div>
  );
}

function normalizeTagList(values: unknown[]): string[] {
  const tags: string[] = [];
  const seen = new Set<string>();

  for (const value of values) {
    const normalized = String(value ?? "").trim().replace(/\s+/g, " ");
    if (!normalized) {
      continue;
    }

    const key = normalized.toLowerCase();
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    tags.push(normalized);
  }

  return tags;
}

function splitTagInput(value: string): string[] {
  return normalizeTagList(value.split(/[,\n]/));
}

function summarizeTags(tags: string[], maxVisible = 3): string {
  if (!tags.length) {
    return "";
  }

  if (tags.length <= maxVisible) {
    return tags.join(", ");
  }

  return `${tags.slice(0, maxVisible).join(", ")} +${tags.length - maxVisible}`;
}

function TagList({ tags }: { tags: string[] }) {
  if (!tags.length) {
    return <span className="tag-empty">untagged</span>;
  }

  return (
    <div className="tag-list">
      {tags.map((tag) => (
        <span key={tag} className="tag-pill">
          {tag}
        </span>
      ))}
    </div>
  );
}

interface DenseColumn<T> {
  key: string;
  label: string;
  className?: string;
  pin?: "left";
  stickyOffset?: number;
  sortable?: boolean;
  sortValue?: (row: T) => string | number | null | undefined;
  filterValue?: (row: T) => string;
  render: (row: T) => ReactNode;
}

function DenseTable<T>({
  columns,
  rows,
  rowKey,
  emptyTitle,
  emptyBody,
  onRowClick,
  selectedRowId,
  filterPlaceholder,
  defaultSort
}: {
  columns: DenseColumn<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  emptyTitle: string;
  emptyBody: string;
  onRowClick?: (row: T) => void;
  selectedRowId?: string | null;
  filterPlaceholder?: string;
  defaultSort?: {
    key: string;
    direction: "asc" | "desc";
  };
}) {
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<string | null>(defaultSort?.key ?? null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">(defaultSort?.direction ?? "asc");

  const normalizedQuery = query.trim().toLowerCase();

  const preparedRows = useMemo(() => {
    const stringify = (value: unknown): string => {
      if (value === null || value === undefined) {
        return "";
      }
      if (typeof value === "number" || typeof value === "boolean") {
        return String(value);
      }
      if (typeof value === "object") {
        return JSON.stringify(value);
      }
      return String(value);
    };

    const readCellValue = (row: T, column: DenseColumn<T>): string => {
      if (column.filterValue) {
        return column.filterValue(row);
      }
      if (column.sortValue) {
        return stringify(column.sortValue(row));
      }
      const record = row as Record<string, unknown>;
      if (column.key in record) {
        return stringify(record[column.key]);
      }
      return stringify(row);
    };

    const filtered = rows.filter((row) => {
      if (!normalizedQuery) {
        return true;
      }

      return columns.some((column) => readCellValue(row, column).toLowerCase().includes(normalizedQuery));
    });

    if (!sortKey) {
      return filtered;
    }

    const activeColumn = columns.find((column) => column.key === sortKey);
    if (!activeColumn) {
      return filtered;
    }

    const sorted = [...filtered].sort((leftRow, rightRow) => {
      const leftRaw = activeColumn.sortValue ? activeColumn.sortValue(leftRow) : readCellValue(leftRow, activeColumn);
      const rightRaw = activeColumn.sortValue ? activeColumn.sortValue(rightRow) : readCellValue(rightRow, activeColumn);

      if (leftRaw === null || leftRaw === undefined) {
        return 1;
      }
      if (rightRaw === null || rightRaw === undefined) {
        return -1;
      }

      if (typeof leftRaw === "number" && typeof rightRaw === "number") {
        return sortDirection === "asc" ? leftRaw - rightRaw : rightRaw - leftRaw;
      }

      const leftText = stringify(leftRaw).toLowerCase();
      const rightText = stringify(rightRaw).toLowerCase();
      const order = leftText.localeCompare(rightText, undefined, { numeric: true });
      return sortDirection === "asc" ? order : -order;
    });

    return sorted;
  }, [columns, normalizedQuery, rows, sortDirection, sortKey]);

  function toggleSort(column: DenseColumn<T>) {
    if (!column.sortable) {
      return;
    }

    if (sortKey === column.key) {
      setSortDirection((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }

    setSortKey(column.key);
    setSortDirection("asc");
  }

  if (!preparedRows.length && !rows.length) {
    return <EmptyState title={emptyTitle} body={emptyBody} />;
  }

  return (
    <div className="dense-table-frame">
      {filterPlaceholder ? (
        <div className="dense-table-toolbar">
          <span>{preparedRows.length} rows</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={filterPlaceholder}
          />
        </div>
      ) : null}
      {!preparedRows.length ? <EmptyState title="No matching rows" body="Adjust the table filter to bring rows back into view." /> : null}
      <div className="dense-table-shell">
      <table className="dense-table">
        <thead>
          <tr>
            {columns.map((column) => {
              const isActiveSort = sortKey === column.key;
              const stickyStyle = column.pin === "left" ? { left: `${column.stickyOffset ?? 0}px` } : undefined;
              const className = [column.className, column.pin === "left" ? "pinned-left" : ""].filter(Boolean).join(" ");

              return (
                <th key={column.key} className={className} style={stickyStyle}>
                  {column.sortable ? (
                    <button type="button" className={`table-sort-button ${isActiveSort ? "is-active" : ""}`} onClick={() => toggleSort(column)}>
                      <span>{column.label}</span>
                      <span className="sort-indicator">{isActiveSort ? (sortDirection === "asc" ? "ASC" : "DESC") : "SORT"}</span>
                    </button>
                  ) : (
                    column.label
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {preparedRows.map((row) => {
            const key = rowKey(row);
            const selectable = Boolean(onRowClick);
            const selected = selectedRowId === key;

            return (
              <tr
                key={key}
                className={`${selectable ? "is-clickable" : ""} ${selected ? "is-selected" : ""}`.trim()}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
              >
                {columns.map((column) => {
                  const stickyStyle = column.pin === "left" ? { left: `${column.stickyOffset ?? 0}px` } : undefined;
                  const className = [column.className, column.pin === "left" ? "pinned-left" : ""].filter(Boolean).join(" ");

                  return (
                    <td key={column.key} className={className} style={stickyStyle}>
                      {column.render(row)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
    </div>
  );
}

function pretty(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (value === null || value === undefined) {
    return "n/a";
  }
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function formatPercent(value: unknown): string {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) {
    return "n/a";
  }
  const rounded = Math.abs(numeric - Math.round(numeric)) < 0.005 ? String(Math.round(numeric)) : numeric.toFixed(2);
  return `${rounded}%`;
}

function shortId(value: unknown): string {
  if (typeof value !== "string" || value.length <= 18) {
    return pretty(value);
  }
  return `${value.slice(0, 8)}...${value.slice(-6)}`;
}

function formatTimestamp(value: unknown): string {
  if (typeof value !== "string") {
    return "n/a";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(parsed);
}

function compactText(value: unknown, maxLength = 120): string {
  const text = typeof value === "string" ? value : pretty(value);
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 3)}...`;
}

function parseNumber(value: string, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatDatasetOptionLabel(dataset: JsonRecord): string {
  const tags = normalizeTagList(Array.isArray(dataset.tags) ? dataset.tags : []);
  return tags.length
    ? `${dataset.name} [${summarizeTags(tags)}] (${shortId(dataset.id)})`
    : `${dataset.name} (${shortId(dataset.id)})`;
}

export default function App() {
  const [activeTab, setActiveTab] = useState(tabs[0]);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [catalog, setCatalog] = useState<JsonRecord | null>(null);
  const [runtimeCapabilities, setRuntimeCapabilities] = useState<JsonRecord | null>(null);
  const [datasets, setDatasets] = useState<JsonRecord[]>([]);
  const [features, setFeatures] = useState<JsonRecord[]>([]);
  const [models, setModels] = useState<JsonRecord[]>([]);
  const [trainingRuns, setTrainingRuns] = useState<JsonRecord[]>([]);
  const [testingRuns, setTestingRuns] = useState<JsonRecord[]>([]);
  const [monitoring, setMonitoring] = useState<JsonRecord | null>(null);
  const [selectedRun, setSelectedRun] = useState<{ kind: "training" | "testing"; id: string } | null>(null);
  const [runEvents, setRunEvents] = useState<JsonRecord[]>([]);
  const [runMetrics, setRunMetrics] = useState<JsonRecord[]>([]);
  const [runTraces, setRunTraces] = useState<JsonRecord[]>([]);
  const [runArtifacts, setRunArtifacts] = useState<JsonRecord[]>([]);
  const [overrideValue, setOverrideValue] = useState("0.02");
  const [importPath, setImportPath] = useState("");
  const [importName, setImportName] = useState("findf PIT Import");
  const [importTagText, setImportTagText] = useState("");
  const [selectedImportTags, setSelectedImportTags] = useState<string[]>([]);
  const [savedDatasetTags, setSavedDatasetTags] = useState<JsonRecord[]>([]);
  const [newSavedTagName, setNewSavedTagName] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [featureDatasetId, setFeatureDatasetId] = useState("");
  const [selectedLayerId, setSelectedLayerId] = useState("");
  const [trainingForm, setTrainingForm] = useState({
    name: "Master Control Training",
    dataset_version_id: "",
    feature_set_version_id: "",
    model_kind: "lightgbm",
    epochs: "8",
    learning_rate: "0.02",
    hidden_dim: "64",
    checkpoint_frequency: "1",
    horizon_days: "5",
    compute_target: "auto",
    precision_mode: "auto",
    batch_size: "128",
    sequence_length: "24",
    gradient_clip_norm: "1.0"
  });
  const [testingForm, setTestingForm] = useState({
    name: "Out-of-Sample Backtest",
    model_version_id: "",
    feature_set_version_id: "",
    execution_mode: "paper",
    rebalance_decile: "0.10",
    stress_iterations: "300"
  });
  const [runtimeSelfCheckForm, setRuntimeSelfCheckForm] = useState({
    compute_target: "auto",
    precision_mode: "auto",
    batch_size: "32",
    sequence_length: "20",
    gradient_clip_norm: "1.0",
    model_kind: "pytorch_mlp",
    input_dim: "8"
  });
  const [runtimeSelfCheckResult, setRuntimeSelfCheckResult] = useState<JsonRecord | null>(null);
  const [runtimeSelfCheckRunning, setRuntimeSelfCheckRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [collapsedPanels, setCollapsedPanels] = useState<Record<string, boolean>>(() => loadCollapsedPanelMap(PANEL_COLLAPSE_STORAGE_KEY));

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(PANEL_COLLAPSE_STORAGE_KEY, JSON.stringify(collapsedPanels));
  }, [collapsedPanels]);

  function isPanelCollapsed(panelId: string): boolean {
    return Boolean(collapsedPanels[panelId]);
  }

  function togglePanelCollapse(panelId: string) {
    setCollapsedPanels((currentPanels) => ({
      ...currentPanels,
      [panelId]: !currentPanels[panelId]
    }));
  }

  function renderPanelCollapseButton(panelId: string, label: string) {
    const collapsed = isPanelCollapsed(panelId);

    return (
      <button
        type="button"
        className={`panel-collapse-button ${collapsed ? "is-collapsed" : ""}`}
        onClick={() => togglePanelCollapse(panelId)}
        aria-expanded={!collapsed}
        aria-label={`${collapsed ? "Expand" : "Collapse"} ${label}`}
        title={`${collapsed ? "Expand" : "Collapse"} ${label}`}
      >
        <span className="panel-collapse-glyph" aria-hidden="true">
          {collapsed ? "+" : "-"}
        </span>
        {collapsed ? "Expand" : "Collapse"}
      </button>
    );
  }

  function renderCollapsedPanelNote(message: string) {
    return <div className="panel-collapsed-note">{message}</div>;
  }

  async function refresh() {
    try {
      const [overviewData, catalogData, runtimeData, datasetData, featureData, modelData, trainingData, testingData, monitoringData, savedTagData] =
        await Promise.all([
          api.get<OverviewResponse>("/api/overview"),
          api.get<JsonRecord>("/api/catalog"),
          api.get<JsonRecord>("/api/runtime-capabilities"),
          api.get<JsonRecord[]>("/api/datasets"),
          api.get<JsonRecord[]>("/api/features"),
          api.get<JsonRecord[]>("/api/model-versions"),
          api.get<JsonRecord[]>("/api/training-runs"),
          api.get<JsonRecord[]>("/api/testing-runs"),
          api.get<JsonRecord>("/api/monitoring"),
          api.get<JsonRecord[]>("/api/dataset-tags")
        ]);
      setOverview(overviewData);
      setCatalog(catalogData);
      setRuntimeCapabilities(runtimeData);
      setDatasets(datasetData);
      setFeatures(featureData);
      setModels(modelData);
      setTrainingRuns(trainingData);
      setTestingRuns(testingData);
      setMonitoring(monitoringData);
      setSavedDatasetTags(savedTagData);
      setError(null);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Failed to load API data.");
    }
  }

  async function refreshRunDetail(run: { kind: "training" | "testing"; id: string }) {
    const [events, metrics, traces, artifacts] = await Promise.all([
      api.get<JsonRecord[]>(`/api/runs/${run.kind}/${run.id}/events`),
      api.get<JsonRecord[]>(`/api/runs/${run.kind}/${run.id}/metrics`),
      api.get<JsonRecord[]>(`/api/runs/${run.kind}/${run.id}/traces`),
      api.get<JsonRecord[]>(`/api/runs/${run.kind}/${run.id}/artifacts`)
    ]);
    setRunEvents(events);
    setRunMetrics(metrics);
    setRunTraces(traces);
    setRunArtifacts(artifacts);
  }

  useEffect(() => {
    refresh();
    const interval = window.setInterval(refresh, 5000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!selectedRun) return;
    refreshRunDetail(selectedRun).catch((detailError) => setError(detailError instanceof Error ? detailError.message : "Failed to load run detail."));
    const detailInterval = window.setInterval(() => {
      refreshRunDetail(selectedRun).catch((detailError) => setError(detailError instanceof Error ? detailError.message : "Failed to load run detail."));
      refresh().catch((refreshError) => setError(refreshError instanceof Error ? refreshError.message : "Failed to refresh run state."));
    }, 3000);
    const source = new EventSource(`${api.baseUrl}/api/stream/${selectedRun.kind}/${selectedRun.id}`);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as JsonRecord;
      setRunEvents((current) => [...current.filter((item) => item.id !== event.id), event].sort((a, b) => a.id - b.id));
    };
    source.onerror = () => source.close();
    return () => {
      window.clearInterval(detailInterval);
      source.close();
    };
  }, [selectedRun]);

  const modelSpecs = useMemo(() => ((catalog?.model_specs ?? []) as JsonRecord[]), [catalog]);
  const researchLayers = useMemo(() => ((catalog?.research_layers ?? []) as JsonRecord[]), [catalog]);
  const importTags = useMemo(() => normalizeTagList([...selectedImportTags, ...splitTagInput(importTagText)]), [importTagText, selectedImportTags]);

  useEffect(() => {
    setSelectedDatasetId((current) => (current && datasets.some((dataset) => dataset.id === current) ? current : String(datasets[0]?.id ?? "")));
  }, [datasets]);

  useEffect(() => {
    setFeatureDatasetId((current) => (current && datasets.some((dataset) => dataset.id === current) ? current : String(datasets[0]?.id ?? "")));
  }, [datasets]);

  useEffect(() => {
    const availableTags = new Set(savedDatasetTags.map((tag) => String(tag.name)));
    setSelectedImportTags((current) => current.filter((tag) => availableTags.has(tag)));
  }, [savedDatasetTags]);

  useEffect(() => {
    setSelectedLayerId((current) => (current && researchLayers.some((layer) => layer.id === current) ? current : String(researchLayers[0]?.id ?? "")));
  }, [researchLayers]);

  useEffect(() => {
    setTrainingForm((current) => ({
      ...current,
      dataset_version_id:
        current.dataset_version_id && datasets.some((dataset) => dataset.id === current.dataset_version_id)
          ? current.dataset_version_id
          : "",
      feature_set_version_id:
        current.feature_set_version_id && features.some((feature) => feature.id === current.feature_set_version_id)
          ? current.feature_set_version_id
          : "",
      model_kind:
        current.model_kind && modelSpecs.some((spec) => spec.kind === current.model_kind)
          ? current.model_kind
          : String(modelSpecs[0]?.kind ?? current.model_kind)
    }));
  }, [datasets, features, modelSpecs]);

  useEffect(() => {
    setTestingForm((current) => ({
      ...current,
      model_version_id:
        current.model_version_id && models.some((model) => model.id === current.model_version_id)
          ? current.model_version_id
          : String(models[0]?.id ?? ""),
      feature_set_version_id:
        current.feature_set_version_id && features.some((feature) => feature.id === current.feature_set_version_id)
          ? current.feature_set_version_id
          : ""
    }));
  }, [features, models]);

  useEffect(() => {
    setRuntimeSelfCheckForm((current) => ({
      ...current,
      compute_target:
        current.compute_target && Array.isArray(runtimeCapabilities?.supported_compute_targets) && (runtimeCapabilities.supported_compute_targets as string[]).includes(current.compute_target)
          ? current.compute_target
          : String(runtimeCapabilities?.recommended_compute_target ?? "auto"),
      precision_mode:
        current.precision_mode && Array.isArray(runtimeCapabilities?.supported_precision_modes) && (runtimeCapabilities.supported_precision_modes as string[]).includes(current.precision_mode)
          ? current.precision_mode
          : "auto"
    }));
  }, [runtimeCapabilities]);

  const selectedRunData = useMemo(() => {
    if (!selectedRun) return null;
    return selectedRun.kind === "training"
      ? trainingRuns.find((run) => run.id === selectedRun.id) ?? null
      : testingRuns.find((run) => run.id === selectedRun.id) ?? null;
  }, [selectedRun, testingRuns, trainingRuns]);

  const selectedResearchLayer = useMemo(
    () => researchLayers.find((layer) => String(layer.id) === selectedLayerId) ?? null,
    [researchLayers, selectedLayerId]
  );

  async function createDataset() {
    await api.post("/api/datasets", {});
    refresh();
  }

  function toggleImportTag(tagName: string) {
    setSelectedImportTags((current) =>
      current.includes(tagName) ? current.filter((tag) => tag !== tagName) : [...current, tagName]
    );
  }

  async function importDataset() {
    if (!importPath.trim()) return;
    await api.post("/api/datasets/import-parquet", {
      path: importPath.trim(),
      name: importName.trim() || undefined,
      tags: importTags
    });
    setImportPath("");
    setImportTagText("");
    setSelectedImportTags([]);
    refresh();
  }

  async function createSavedTag() {
    if (!newSavedTagName.trim()) return;
    const savedTag = await api.post<JsonRecord>("/api/dataset-tags", {
      name: newSavedTagName.trim()
    });
    const savedTagName = String(savedTag.name);
    setNewSavedTagName("");
    setSelectedImportTags((current) => (current.includes(savedTagName) ? current : [...current, savedTagName]));
    refresh();
  }

  async function deleteSavedTag(tagId: string, tagName: string) {
    await api.delete(`/api/dataset-tags/${tagId}`);
    setSelectedImportTags((current) => current.filter((tag) => tag !== tagName));
    refresh();
  }

  async function createFeatureSet() {
    if (!featureDatasetId) return;
    await api.post("/api/features", {
      dataset_version_id: featureDatasetId
    });
    refresh();
  }

  async function startTraining() {
    const response = await api.post<JsonRecord>("/api/training-runs", {
      name: trainingForm.name.trim() || "Master Control Training",
      model_kind: trainingForm.model_kind,
      epochs: parseNumber(trainingForm.epochs, 8),
      learning_rate: parseNumber(trainingForm.learning_rate, 0.02),
      hidden_dim: parseNumber(trainingForm.hidden_dim, 64),
      checkpoint_frequency: parseNumber(trainingForm.checkpoint_frequency, 1),
      horizon_days: parseNumber(trainingForm.horizon_days, 5),
      runtime_settings: {
        compute_target: trainingForm.compute_target,
        precision_mode: trainingForm.precision_mode,
        batch_size: parseNumber(trainingForm.batch_size, 128),
        sequence_length: parseNumber(trainingForm.sequence_length, 24),
        gradient_clip_norm: parseNumber(trainingForm.gradient_clip_norm, 1.0)
      },
      ...(trainingForm.dataset_version_id ? { dataset_version_id: trainingForm.dataset_version_id } : {}),
      ...(trainingForm.feature_set_version_id ? { feature_set_version_id: trainingForm.feature_set_version_id } : {})
    });
    setSelectedRun({ kind: "training", id: response.id });
    await refresh();
    await refreshRunDetail({ kind: "training", id: response.id });
  }

  async function startTesting() {
    if (!testingForm.model_version_id) return;
    const response = await api.post<JsonRecord>("/api/testing-runs", {
      name: testingForm.name.trim() || "Out-of-Sample Backtest",
      model_version_id: testingForm.model_version_id,
      execution_mode: testingForm.execution_mode,
      rebalance_decile: parseNumber(testingForm.rebalance_decile, 0.1),
      stress_iterations: parseNumber(testingForm.stress_iterations, 300),
      ...(testingForm.feature_set_version_id ? { feature_set_version_id: testingForm.feature_set_version_id } : {})
    });
    setSelectedRun({ kind: "testing", id: response.id });
    await refresh();
    await refreshRunDetail({ kind: "testing", id: response.id });
  }

  async function controlRun(action: "pause" | "resume" | "stop") {
    if (!selectedRun) return;
    await api.post(`/api/${selectedRun.kind}-runs/${selectedRun.id}/${action}`);
    refresh();
    refreshRunDetail(selectedRun);
  }

  async function applyOverride() {
    if (!selectedRun || selectedRun.kind !== "training") return;
    await api.post(`/api/training-runs/${selectedRun.id}/overrides`, {
      overrides: { learning_rate: Number(overrideValue) || 0.01 }
    });
    refreshRunDetail(selectedRun);
  }

  async function updateModelStatus(modelId: string, action: "promote" | "reject") {
    await api.post(`/api/model-versions/${modelId}/${action}`);
    refresh();
  }

  async function updateLayerControls(layerId: string, updates: JsonRecord) {
    await api.post(`/api/research-layers/${layerId}/controls`, updates);
    await refresh();
  }

  async function setPreferredLayerModel(layerId: string, modelKind: string) {
    await updateLayerControls(layerId, { preferred_model_kind: modelKind });
  }

  async function toggleLayerCandidate(layerId: string, modelKind: string) {
    const layer = researchLayers.find((item) => String(item.id) === layerId);
    if (!layer) return;
    const current = Array.isArray(layer.control_state?.candidate_model_kinds)
      ? layer.control_state.candidate_model_kinds.map((value: unknown) => String(value))
      : [];
    const next = current.includes(modelKind)
      ? current.filter((value: string) => value !== modelKind)
      : [...current, modelKind];
    await updateLayerControls(layerId, { candidate_model_kinds: next });
  }

  async function toggleLayerProcessStep(layerId: string, stepId: string) {
    const layer = researchLayers.find((item) => String(item.id) === layerId);
    if (!layer) return;
    const current = typeof layer.control_state?.process_step_state === "object" && layer.control_state?.process_step_state
      ? { ...layer.control_state.process_step_state }
      : {};
    current[stepId] = !current[stepId];
    await updateLayerControls(layerId, { process_step_state: current });
  }

  async function updateLayerRuntimeSetting(layerId: string, key: string, value: string | number) {
    const layer = researchLayers.find((item) => String(item.id) === layerId);
    if (!layer) return;
    const current = typeof layer.control_state?.runtime_settings === "object" && layer.control_state?.runtime_settings
      ? { ...layer.control_state.runtime_settings }
      : {};
    current[key] = value;
    await updateLayerControls(layerId, { runtime_settings: current });
  }

  async function runRuntimeSelfCheck() {
    setRuntimeSelfCheckRunning(true);
    try {
      const payload = await api.post<JsonRecord>("/api/runtime-self-check", {
        compute_target: runtimeSelfCheckForm.compute_target,
        precision_mode: runtimeSelfCheckForm.precision_mode,
        batch_size: parseNumber(runtimeSelfCheckForm.batch_size, 32),
        sequence_length: parseNumber(runtimeSelfCheckForm.sequence_length, 20),
        gradient_clip_norm: parseNumber(runtimeSelfCheckForm.gradient_clip_norm, 1.0),
        model_kind: runtimeSelfCheckForm.model_kind,
        input_dim: parseNumber(runtimeSelfCheckForm.input_dim, 8)
      });
      setRuntimeSelfCheckResult(payload);
    } finally {
      setRuntimeSelfCheckRunning(false);
    }
  }

  const latestDataset = datasets[0] ?? null;
  const latestDatasetAssessment = latestDataset && typeof latestDataset.summary?.assessment === "object"
    ? (latestDataset.summary.assessment as JsonRecord)
    : null;
  const selectedDataset = useMemo(
    () => datasets.find((dataset) => String(dataset.id) === selectedDatasetId) ?? latestDataset,
    [datasets, latestDataset, selectedDatasetId]
  );
  const selectedDatasetAssessment = selectedDataset && typeof selectedDataset.summary?.assessment === "object"
    ? (selectedDataset.summary.assessment as JsonRecord)
    : null;
  const selectedDatasetIssues = useMemo(
    () => (Array.isArray(selectedDatasetAssessment?.issues) ? (selectedDatasetAssessment.issues as JsonRecord[]) : []),
    [selectedDatasetAssessment]
  );
  const selectedDatasetGapSamples = useMemo(
    () => (Array.isArray(selectedDatasetAssessment?.gaps?.gap_samples) ? (selectedDatasetAssessment.gaps.gap_samples as JsonRecord[]) : []),
    [selectedDatasetAssessment]
  );
  const selectedDatasetColumnCoverage = useMemo(() => {
    if (!selectedDatasetAssessment || typeof selectedDatasetAssessment.column_completeness !== "object" || !selectedDatasetAssessment.column_completeness) {
      return [] as Array<{ name: string; detail: JsonRecord }>;
    }

    return Object.entries(selectedDatasetAssessment.column_completeness as Record<string, JsonRecord>)
      .map(([name, detail]) => ({ name, detail }))
      .sort((left, right) => {
        const leftPct = Number(left.detail?.non_null_pct ?? 0);
        const rightPct = Number(right.detail?.non_null_pct ?? 0);
        if (leftPct !== rightPct) {
          return leftPct - rightPct;
        }
        return left.name.localeCompare(right.name);
      });
  }, [selectedDatasetAssessment]);
  const latestFeatureSet = features[0] ?? null;
  const latestModel = models[0] ?? null;
  const activeTrainingCount = trainingRuns.filter((run) => ["queued", "running", "paused"].includes(String(run.state))).length;
  const activeTestingCount = testingRuns.filter((run) => ["queued", "running", "paused"].includes(String(run.state))).length;
  const promotedModelCount = models.filter((model) => model.status === "promoted").length;
  const selectedProgressRaw = runEvents.length ? runEvents[runEvents.length - 1]?.progress_pct : null;
  const selectedProgress = typeof selectedProgressRaw === "number" ? Math.max(0, Math.min(100, selectedProgressRaw)) : null;
  const recentEvents = [...runEvents].slice(-16).reverse();
  const recentTraces = [...runTraces].slice(-6).reverse();
  const primaryMetrics = [...runMetrics].slice(0, 6);

  const datasetColumns: DenseColumn<JsonRecord>[] = [
    {
      key: "name",
      label: "Name",
      className: "cell-wide",
      pin: "left",
      sortable: true,
      sortValue: (dataset) => String(dataset.name ?? ""),
      filterValue: (dataset) => `${dataset.name ?? ""} ${dataset.id ?? ""} ${normalizeTagList(Array.isArray(dataset.tags) ? dataset.tags : []).join(" ")}`,
      render: (dataset) => <strong>{dataset.name}</strong>
    },
    {
      key: "tags",
      label: "Tags",
      className: "cell-wide",
      sortable: true,
      sortValue: (dataset) => normalizeTagList(Array.isArray(dataset.tags) ? dataset.tags : []).join(", "),
      filterValue: (dataset) => normalizeTagList(Array.isArray(dataset.tags) ? dataset.tags : []).join(" "),
      render: (dataset) => <TagList tags={normalizeTagList(Array.isArray(dataset.tags) ? dataset.tags : [])} />
    },
    { key: "id", label: "ID", className: "mono-cell", sortable: true, sortValue: (dataset) => String(dataset.id ?? ""), render: (dataset) => shortId(dataset.id) },
    {
      key: "data_level",
      label: "Data Level",
      sortable: true,
      sortValue: (dataset) => Number(dataset.summary?.assessment?.score_pct ?? -1),
      render: (dataset) => <AssessmentPill value={String(dataset.summary?.assessment?.data_level ?? dataset.summary?.assessment?.status ?? "unknown")} />
    },
    { key: "rows", label: "Rows", className: "mono-cell", sortable: true, sortValue: (dataset) => Number(dataset.summary?.rows ?? 0), render: (dataset) => pretty(dataset.summary?.rows) },
    {
      key: "completeness",
      label: "Complete",
      className: "mono-cell",
      sortable: true,
      sortValue: (dataset) => Number(dataset.summary?.assessment?.completeness_pct ?? 0),
      render: (dataset) => formatPercent(dataset.summary?.assessment?.completeness_pct)
    },
    {
      key: "gap_sessions",
      label: "Gap Sess",
      className: "mono-cell",
      sortable: true,
      sortValue: (dataset) => Number(dataset.summary?.assessment?.gaps?.missing_sessions ?? 0),
      render: (dataset) => pretty(dataset.summary?.assessment?.gaps?.missing_sessions ?? 0)
    },
    { key: "source", label: "Source", className: "mono-cell", sortable: true, sortValue: (dataset) => String(dataset.source_id ?? ""), render: (dataset) => pretty(dataset.source_id) },
    {
      key: "path",
      label: "Path",
      className: "mono-cell cell-wide",
      sortable: true,
      sortValue: (dataset) => String(dataset.summary?.artifacts?.source_path ?? dataset.summary?.artifacts?.raw_path ?? ""),
      filterValue: (dataset) => String(dataset.summary?.artifacts?.source_path ?? dataset.summary?.artifacts?.raw_path ?? ""),
      render: (dataset) => dataset.summary?.artifacts?.source_path ?? dataset.summary?.artifacts?.raw_path ?? "generated locally"
    },
    {
      key: "news_rows",
      label: "News Events",
      className: "mono-cell",
      sortable: true,
      sortValue: (dataset) => Number(dataset.summary?.news_events?.rows ?? 0),
      render: (dataset) => pretty(dataset.summary?.news_events?.rows ?? 0)
    }
  ];

  const featureColumns: DenseColumn<JsonRecord>[] = [
    { key: "name", label: "Name", className: "cell-wide", pin: "left", sortable: true, sortValue: (feature) => String(feature.name ?? ""), filterValue: (feature) => `${feature.name ?? ""} ${feature.id ?? ""}`, render: (feature) => <strong>{feature.name}</strong> },
    { key: "id", label: "ID", className: "mono-cell", sortable: true, sortValue: (feature) => String(feature.id ?? ""), render: (feature) => shortId(feature.id) },
    { key: "dataset", label: "Dataset", className: "mono-cell", sortable: true, sortValue: (feature) => String(feature.dataset_version_id ?? ""), render: (feature) => shortId(feature.dataset_version_id) },
    { key: "rows", label: "Rows", className: "mono-cell", sortable: true, sortValue: (feature) => Number(feature.summary?.rows ?? 0), render: (feature) => pretty(feature.summary?.rows) },
    {
      key: "ticker_text_coverage",
      label: "Ticker Text Cov",
      className: "mono-cell",
      sortable: true,
      sortValue: (feature) => Number(feature.summary?.text_embedding_summary?.ticker_text_coverage ?? 0),
      render: (feature) => pretty(feature.summary?.text_embedding_summary?.ticker_text_coverage ?? 0)
    },
    {
      key: "macro_text_coverage",
      label: "Macro Text Cov",
      className: "mono-cell",
      sortable: true,
      sortValue: (feature) => Number(feature.summary?.text_embedding_summary?.macro_text_coverage ?? 0),
      render: (feature) => pretty(feature.summary?.text_embedding_summary?.macro_text_coverage ?? 0)
    }
  ];

  const factorColumns: DenseColumn<JsonRecord>[] = [
    { key: "category", label: "Category", className: "mono-cell", sortable: true, sortValue: (factor) => String(factor.category ?? ""), render: (factor) => pretty(factor.category) },
    { key: "name", label: "Name", className: "cell-wide", pin: "left", sortable: true, sortValue: (factor) => String(factor.name ?? ""), filterValue: (factor) => `${factor.name ?? ""} ${factor.formula ?? ""}`, render: (factor) => <strong>{factor.name}</strong> },
    { key: "formula", label: "Formula", className: "mono-cell cell-wide", sortable: true, sortValue: (factor) => String(factor.formula ?? ""), render: (factor) => compactText(factor.formula, 140) },
    { key: "config", label: "Config", className: "mono-cell cell-wide", sortable: true, sortValue: (factor) => JSON.stringify(factor.config ?? {}), render: (factor) => compactText(factor.config, 140) }
  ];

  const modelSpecColumns: DenseColumn<JsonRecord>[] = [
    { key: "name", label: "Name", className: "cell-wide", pin: "left", sortable: true, sortValue: (spec) => String(spec.name ?? ""), filterValue: (spec) => `${spec.name ?? ""} ${spec.kind ?? ""}`, render: (spec) => <strong>{spec.name}</strong> },
    { key: "kind", label: "Kind", className: "mono-cell", sortable: true, sortValue: (spec) => String(spec.kind ?? ""), render: (spec) => pretty(spec.kind) },
    { key: "description", label: "Description", className: "cell-wide", sortable: true, sortValue: (spec) => String(spec.description ?? ""), render: (spec) => compactText(spec.description, 140) }
  ];

  const trainingRunColumns: DenseColumn<JsonRecord>[] = [
    { key: "id", label: "Run", className: "mono-cell", pin: "left", sortable: true, sortValue: (run) => String(run.id ?? ""), filterValue: (run) => `${run.id ?? ""} ${run.current_stage ?? ""} ${run.state ?? ""}`, render: (run) => shortId(run.id) },
    { key: "model_kind", label: "Model", className: "mono-cell", sortable: true, sortValue: (run) => String(run.config?.model_kind ?? ""), render: (run) => pretty(run.config?.model_kind) },
    { key: "stage", label: "Stage", className: "cell-wide", sortable: true, sortValue: (run) => String(run.current_stage ?? ""), render: (run) => pretty(run.current_stage) },
    { key: "state", label: "State", sortable: true, sortValue: (run) => String(run.state ?? ""), render: (run) => <StatusPill value={String(run.state)} /> },
    { key: "updated", label: "Updated", className: "mono-cell", sortable: true, sortValue: (run) => String(run.updated_at ?? ""), render: (run) => formatTimestamp(run.updated_at) }
  ];

  const modelColumns: DenseColumn<JsonRecord>[] = [
    { key: "name", label: "Name", className: "cell-wide", pin: "left", sortable: true, sortValue: (model) => String(model.name ?? ""), filterValue: (model) => `${model.name ?? ""} ${model.status ?? ""}`, render: (model) => <strong>{model.name}</strong> },
    { key: "status", label: "Status", sortable: true, sortValue: (model) => String(model.status ?? ""), render: (model) => <StatusPill value={String(model.status)} /> },
    { key: "sharpe", label: "Sharpe", className: "mono-cell", sortable: true, sortValue: (model) => Number(model.metrics?.sharpe ?? Number.NEGATIVE_INFINITY), render: (model) => pretty(model.metrics?.sharpe) },
    { key: "rank_ic", label: "Rank IC", className: "mono-cell", sortable: true, sortValue: (model) => Number(model.metrics?.rank_ic ?? Number.NEGATIVE_INFINITY), render: (model) => pretty(model.metrics?.rank_ic) }
  ];

  const testingRunColumns: DenseColumn<JsonRecord>[] = [
    { key: "id", label: "Run", className: "mono-cell", pin: "left", sortable: true, sortValue: (run) => String(run.id ?? ""), filterValue: (run) => `${run.id ?? ""} ${run.current_stage ?? ""} ${run.state ?? ""}`, render: (run) => shortId(run.id) },
    { key: "model_version", label: "Model", className: "mono-cell", sortable: true, sortValue: (run) => String(run.model_version_id ?? ""), render: (run) => shortId(run.model_version_id) },
    { key: "stage", label: "Stage", className: "cell-wide", sortable: true, sortValue: (run) => String(run.current_stage ?? ""), render: (run) => pretty(run.current_stage) },
    { key: "state", label: "State", sortable: true, sortValue: (run) => String(run.state ?? ""), render: (run) => <StatusPill value={String(run.state)} /> },
    { key: "updated", label: "Updated", className: "mono-cell", sortable: true, sortValue: (run) => String(run.updated_at ?? ""), render: (run) => formatTimestamp(run.updated_at) }
  ];

  const metricColumns: DenseColumn<JsonRecord>[] = [
    { key: "name", label: "Metric", className: "cell-wide", sortable: true, sortValue: (metric) => String(metric.name ?? ""), render: (metric) => <strong>{metric.name}</strong> },
    { key: "group", label: "Group", className: "mono-cell", sortable: true, sortValue: (metric) => String(metric.group_name ?? ""), render: (metric) => pretty(metric.group_name) },
    { key: "value", label: "Value", className: "mono-cell", sortable: true, sortValue: (metric) => typeof metric.value === "number" ? metric.value : String(metric.value ?? ""), render: (metric) => pretty(metric.value) }
  ];

  const modelRegistryColumns: DenseColumn<JsonRecord>[] = [
    { key: "name", label: "Name", className: "cell-wide", pin: "left", sortable: true, sortValue: (model) => String(model.name ?? ""), filterValue: (model) => `${model.name ?? ""} ${model.id ?? ""} ${model.status ?? ""}`, render: (model) => <strong>{model.name}</strong> },
    { key: "id", label: "ID", className: "mono-cell", sortable: true, sortValue: (model) => String(model.id ?? ""), render: (model) => shortId(model.id) },
    { key: "status", label: "Status", sortable: true, sortValue: (model) => String(model.status ?? ""), render: (model) => <StatusPill value={String(model.status)} /> },
    { key: "sharpe", label: "Sharpe", className: "mono-cell", sortable: true, sortValue: (model) => Number(model.metrics?.sharpe ?? Number.NEGATIVE_INFINITY), render: (model) => pretty(model.metrics?.sharpe) },
    { key: "rank_ic", label: "Rank IC", className: "mono-cell", sortable: true, sortValue: (model) => Number(model.metrics?.rank_ic ?? Number.NEGATIVE_INFINITY), render: (model) => pretty(model.metrics?.rank_ic) },
    {
      key: "actions",
      label: "Actions",
      className: "cell-actions",
      render: (model) => (
        <div className="inline-controls compact-controls">
          <button onClick={(event) => {
            event.stopPropagation();
            updateModelStatus(model.id, "promote");
          }}>Promote</button>
          <button className="ghost" onClick={(event) => {
            event.stopPropagation();
            updateModelStatus(model.id, "reject");
          }}>
            Reject
          </button>
        </div>
      )
    }
  ];

  const eventColumns: DenseColumn<JsonRecord>[] = [
    { key: "time", label: "Time", className: "mono-cell", sortable: true, sortValue: (event) => String(event.timestamp ?? ""), render: (event) => formatTimestamp(event.timestamp) },
    { key: "phase", label: "Phase / Stage", className: "mono-cell cell-wide", sortable: true, sortValue: (event) => `${event.phase ?? ""} ${event.stage ?? ""}`, render: (event) => `${pretty(event.phase)} / ${pretty(event.stage)}` },
    { key: "message", label: "Message", className: "cell-wide", sortable: true, sortValue: (event) => String(event.message ?? ""), render: (event) => compactText(event.message, 180) },
    { key: "progress", label: "Prog", className: "mono-cell", sortable: true, sortValue: (event) => Number(event.progress_pct ?? Number.NEGATIVE_INFINITY), render: (event) => `${pretty(event.progress_pct)}%` }
  ];

  const traceColumns: DenseColumn<JsonRecord>[] = [
    { key: "formula", label: "Formula", className: "mono-cell", sortable: true, sortValue: (trace) => String(trace.formula_id ?? ""), render: (trace) => compactText(trace.formula_id, 30) },
    { key: "label", label: "Label", className: "cell-wide", sortable: true, sortValue: (trace) => String(trace.label ?? ""), render: (trace) => compactText(trace.label, 120) },
    { key: "output", label: "Output", className: "mono-cell cell-wide", sortable: true, sortValue: (trace) => JSON.stringify(trace.output ?? {}), render: (trace) => compactText(trace.output, 180) }
  ];

  const artifactColumns: DenseColumn<JsonRecord>[] = [
    { key: "type", label: "Type", className: "mono-cell", sortable: true, sortValue: (artifact) => String(artifact.artifact_type ?? ""), render: (artifact) => pretty(artifact.artifact_type) },
    { key: "path", label: "Path", className: "mono-cell cell-wide", sortable: true, sortValue: (artifact) => String(artifact.path ?? ""), render: (artifact) => pretty(artifact.path) }
  ];

  const researchLayerColumns: DenseColumn<JsonRecord>[] = [
    { key: "name", label: "Layer", className: "cell-wide", pin: "left", sortable: true, sortValue: (layer) => String(layer.name ?? ""), filterValue: (layer) => `${layer.name ?? ""} ${layer.id ?? ""} ${layer.stage ?? ""}`, render: (layer) => <strong>{layer.name}</strong> },
    { key: "stage", label: "Stage", className: "mono-cell", sortable: true, sortValue: (layer) => String(layer.stage ?? ""), render: (layer) => pretty(layer.stage) },
    { key: "status", label: "Status", sortable: true, sortValue: (layer) => String(layer.status ?? ""), render: (layer) => <StatusPill value={String(layer.status)} /> },
    { key: "selected_model", label: "Selected Model", className: "mono-cell", sortable: true, sortValue: (layer) => String(layer.control_state?.preferred_model_kind ?? ""), render: (layer) => pretty(layer.control_state?.preferred_model_kind ?? "n/a") },
    { key: "compute_target", label: "Compute", className: "mono-cell", sortable: true, sortValue: (layer) => String(layer.control_state?.runtime_settings?.compute_target ?? ""), render: (layer) => pretty(layer.control_state?.runtime_settings?.compute_target ?? "auto") },
    { key: "latest_status", label: "Latest Run", className: "mono-cell", sortable: true, sortValue: (layer) => String(layer.latest_observability?.latest_status ?? ""), render: (layer) => pretty(layer.latest_observability?.latest_status) }
  ];

  function renderMainPanel() {
    switch (activeTab) {
      case "Master Control": {
        const layer = selectedResearchLayer;
        const layerCandidates = Array.isArray(layer?.model_catalog?.candidates) ? (layer.model_catalog.candidates as JsonRecord[]) : [];
        const layerSteps = Array.isArray(layer?.process_steps) ? (layer.process_steps as JsonRecord[]) : [];
        const runtimeCatalog = (layer?.runtime_catalog ?? {}) as JsonRecord;
        const layerRuntimeSettings = (layer?.control_state?.runtime_settings ?? {}) as JsonRecord;
        const layerInputs = Array.isArray(layer?.data_contract?.input_columns)
          ? layer.data_contract.input_columns
          : Array.isArray(layer?.data_contract?.required_columns)
            ? layer.data_contract.required_columns
            : [];
        const layerOutputs = Array.isArray(layer?.data_contract?.prediction_columns)
          ? layer.data_contract.prediction_columns
          : Array.isArray(layer?.data_contract?.output_columns)
            ? layer.data_contract.output_columns
            : Array.isArray(layer?.data_contract?.target_output_columns)
              ? layer.data_contract.target_output_columns
              : [];

        return (
          <DashboardLayout
            layoutKey="master-control"
            defaultRows={[
              ["research-layers", "selected-layer"],
              ["candidate-models", "runtime-control"],
              ["process-controls"]
            ]}
            panels={[
              { id: "research-layers", label: "Research Layers" },
              { id: "selected-layer", label: "Layer Detail" },
              { id: "candidate-models", label: "Candidate Models" },
              { id: "runtime-control", label: "Device And Efficiency" },
              { id: "process-controls", label: "Preprocessing And Formulas" }
            ]}
          >
            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Layer Governance</span>
                  <h2>Research Layers</h2>
                </div>
              </div>
              <p className="panel-copy">
                Pick a layer to inspect its input contract, model candidates, comparison policy, required preprocessing,
                and optional steps you can enable or disable from the control center.
              </p>
              <DenseTable
                columns={researchLayerColumns}
                rows={researchLayers}
                rowKey={(layerRow) => String(layerRow.id)}
                emptyTitle="No research layers"
                emptyBody="Seeded research layers will appear here once the control plane initializes."
                onRowClick={(layerRow) => setSelectedLayerId(String(layerRow.id))}
                selectedRowId={selectedLayerId}
                filterPlaceholder="Filter layers by name, id, or stage"
                defaultSort={{ key: "stage", direction: "asc" }}
              />
            </section>

            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Selected Layer</span>
                  <h2>{layer?.name ?? "Layer Detail"}</h2>
                </div>
                {layer?.status ? <StatusPill value={String(layer.status)} /> : null}
              </div>
              {!layer ? (
                <EmptyState title="No layer selected" body="Choose a research layer from the table to inspect its contracts and controls." />
              ) : (
                <>
                  <p className="panel-copy">{layer.description}</p>
                  <div className="metric-grid compact">
                    <MetricCard label="Selected Model" value={layer.control_state?.preferred_model_kind ?? "n/a"} />
                    <MetricCard label="Comparison Metric" value={layer.control_state?.selection_metric ?? "n/a"} />
                    <MetricCard label="Enabled Steps" value={layerSteps.filter((step) => step.enabled).length} />
                    <MetricCard label="Latest Status" value={layer.latest_observability?.latest_status ?? "n/a"} />
                    <MetricCard label="Compute Target" value={layer.control_state?.runtime_settings?.compute_target ?? "n/a"} />
                    <MetricCard label="Precision" value={layer.control_state?.runtime_settings?.precision_mode ?? "n/a"} />
                  </div>
                  <div className="contract-grid">
                    <article className="contract-card">
                      <small>Input Kind</small>
                      <strong>{pretty(layer.data_contract?.input_kind ?? "n/a")}</strong>
                      <p>{compactText(layerInputs.join(", ") || "No explicit input column contract published.", 180)}</p>
                    </article>
                    <article className="contract-card">
                      <small>Output Kind</small>
                      <strong>{pretty(layer.data_contract?.output_kind ?? layer.data_contract?.prediction_columns?.[0] ?? "n/a")}</strong>
                      <p>{compactText(layerOutputs.join(", ") || "No explicit output column contract published.", 180)}</p>
                    </article>
                    <article className="contract-card">
                      <small>Latest Metrics</small>
                      <strong>{Object.keys(layer.latest_observability?.latest_metrics ?? {}).length}</strong>
                      <p>{compactText(layer.latest_observability?.latest_metrics, 180)}</p>
                    </article>
                    <article className="contract-card">
                      <small>Runtime</small>
                      <strong>{pretty(layer.control_state?.runtime_settings?.compute_target ?? "auto")}</strong>
                      <p>{compactText(layer.latest_comparison?.runtime_summary ?? layer.control_state?.runtime_settings ?? {}, 180)}</p>
                    </article>
                  </div>
                  {layer.latest_comparison ? (
                    <div className="comparison-block">
                      <div className="comparison-head">
                        <span>Latest Comparison</span>
                        <strong>{pretty(layer.latest_comparison?.selected_model_kind)}</strong>
                      </div>
                      <pre className="json-block compact-json">{JSON.stringify(layer.latest_comparison, null, 2)}</pre>
                    </div>
                  ) : null}
                </>
              )}
            </section>

            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Model Selection</span>
                  <h2>Candidate Models</h2>
                </div>
              </div>
              {!layer ? (
                <EmptyState title="No layer selected" body="Select a layer to manage its candidate models and preferred model family." />
              ) : !layerCandidates.length ? (
                <EmptyState title="No candidate models" body="This layer is algorithmic or procedural rather than trainable." />
              ) : (
                <div className="control-list">
                  {layerCandidates.map((candidate) => (
                    <article key={String(candidate.kind)} className="control-card">
                      <div className="control-card-head">
                        <div>
                          <strong>{candidate.label}</strong>
                          <span>{candidate.kind}</span>
                        </div>
                        {candidate.recommended ? <StatusPill value="recommended" /> : null}
                      </div>
                      <p>{candidate.rationale}</p>
                      <div className="pill-row">
                        <StatusPill value={String(candidate.implementation_mode)} />
                        <StatusPill value={String(candidate.input_fit)} />
                        <StatusPill value={`devices:${(candidate.acceleration_modes ?? []).join("/") || "cpu"}`} />
                        {candidate.selected ? <StatusPill value="selected" /> : null}
                      </div>
                      <div className="inline-controls wrap">
                        <button
                          onClick={() => setPreferredLayerModel(String(layer.id), String(candidate.kind))}
                          disabled={candidate.selected}
                        >
                          {candidate.selected ? "Preferred" : "Use As Preferred"}
                        </button>
                        <button
                          className={candidate.enabled_for_comparison ? "ghost is-active-toggle" : "ghost"}
                          onClick={() => toggleLayerCandidate(String(layer.id), String(candidate.kind))}
                        >
                          {candidate.enabled_for_comparison ? "Included In Compare" : "Add To Compare"}
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </section>

            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Runtime Control</span>
                  <h2>Device And Efficiency</h2>
                </div>
              </div>
              {!layer ? (
                <EmptyState title="No layer selected" body="Select a trainable layer to choose device, precision, batch size, and sequence length." />
              ) : !Object.keys(runtimeCatalog).length ? (
                <EmptyState title="No runtime controls" body="This layer is not trainable, so there is no device or runtime configuration to manage." />
              ) : (
                <>
                  <p className="panel-copy">
                    Pick the preferred compute target for this layer and tune the runtime budget. Torch-backed models can
                    use GPU acceleration when the local runtime exposes CUDA/ROCm or DirectML.
                  </p>
                  <div className="metric-grid compact">
                    <MetricCard label="Recommended Target" value={runtimeCapabilities?.recommended_compute_target ?? "cpu"} />
                    <MetricCard label="Detected Devices" value={Array.isArray(runtimeCapabilities?.devices) ? runtimeCapabilities?.devices.length : 0} />
                    <MetricCard label="Batch Size" value={layerRuntimeSettings.batch_size ?? "n/a"} />
                    <MetricCard label="Seq Length" value={runtimeCatalog.supports_sequence_length ? (layerRuntimeSettings.sequence_length ?? "n/a") : "n/a"} />
                  </div>
                  <div className="control-form two-column">
                    <label>
                      <span>Compute Target</span>
                      <select
                        value={String(layerRuntimeSettings.compute_target ?? "auto")}
                        onChange={(event) => updateLayerRuntimeSetting(String(layer.id), "compute_target", event.target.value)}
                      >
                        {Array.isArray(runtimeCatalog.supported_compute_targets)
                          ? runtimeCatalog.supported_compute_targets.map((target) => (
                              <option key={String(target)} value={String(target)}>
                                {String(target)}
                              </option>
                            ))
                          : null}
                      </select>
                    </label>
                    <label>
                      <span>Precision</span>
                      <select
                        value={String(layerRuntimeSettings.precision_mode ?? "auto")}
                        onChange={(event) => updateLayerRuntimeSetting(String(layer.id), "precision_mode", event.target.value)}
                      >
                        {Array.isArray(runtimeCatalog.supported_precision_modes)
                          ? runtimeCatalog.supported_precision_modes.map((mode) => (
                              <option key={String(mode)} value={String(mode)}>
                                {String(mode)}
                              </option>
                            ))
                          : null}
                      </select>
                    </label>
                    <label>
                      <span>Batch Size</span>
                      <input
                        value={String(layerRuntimeSettings.batch_size ?? "")}
                        onChange={(event) => updateLayerRuntimeSetting(String(layer.id), "batch_size", parseNumber(event.target.value, 128))}
                      />
                    </label>
                    <label>
                      <span>Grad Clip</span>
                      <input
                        value={String(layerRuntimeSettings.gradient_clip_norm ?? "")}
                        onChange={(event) => updateLayerRuntimeSetting(String(layer.id), "gradient_clip_norm", parseNumber(event.target.value, 1.0))}
                      />
                    </label>
                    {runtimeCatalog.supports_sequence_length ? (
                      <label>
                        <span>Sequence Length</span>
                        <input
                          value={String(layerRuntimeSettings.sequence_length ?? "")}
                          onChange={(event) => updateLayerRuntimeSetting(String(layer.id), "sequence_length", parseNumber(event.target.value, 24))}
                        />
                      </label>
                    ) : null}
                  </div>
                  {Array.isArray(runtimeCapabilities?.devices) ? (
                    <div className="control-list">
                      {(runtimeCapabilities.devices as JsonRecord[]).map((device, index) => (
                        <article key={`${device.kind}-${index}`} className="control-card">
                          <div className="control-card-head">
                            <div>
                              <strong>{device.label ?? device.kind}</strong>
                              <span>{device.kind}</span>
                            </div>
                            <StatusPill value={String(device.backend ?? device.provider ?? "runtime")} />
                          </div>
                          <p>{compactText(device, 180)}</p>
                        </article>
                      ))}
                    </div>
                  ) : null}
                </>
              )}
            </section>

            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Process Controls</span>
                  <h2>Preprocessing And Formulas</h2>
                </div>
              </div>
              {!layer ? (
                <EmptyState title="No layer selected" body="Select a layer to inspect formulas, algorithms, and optional process steps." />
              ) : (
                <div className="control-list">
                  {layerSteps.map((step) => (
                    <article key={String(step.id)} className="control-card">
                      <div className="control-card-head">
                        <div>
                          <strong>{step.name}</strong>
                          <span>{step.id}</span>
                        </div>
                        <RequirementPill value={String(step.requirement_level)} />
                      </div>
                      <p>{step.algorithm}</p>
                      <div className="formula-block">
                        <small>Formula</small>
                        <code>{step.formula}</code>
                      </div>
                      <div className="contract-grid compact-contract-grid">
                        <article className="contract-card compact-contract">
                          <small>Inputs</small>
                          <p>{compactText(pretty(step.inputs), 160)}</p>
                        </article>
                        <article className="contract-card compact-contract">
                          <small>Outputs</small>
                          <p>{compactText(pretty(step.outputs), 160)}</p>
                        </article>
                      </div>
                      <p className="control-note">{step.validity_reason}</p>
                      <div className="inline-controls wrap">
                        <StatusPill value={step.enabled ? "enabled" : "disabled"} />
                        <button
                          className="ghost"
                          disabled={!step.can_disable}
                          onClick={() => toggleLayerProcessStep(String(layer.id), String(step.id))}
                        >
                          {step.can_disable ? (step.enabled ? "Disable Step" : "Enable Step") : "Locked"}
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </section>
          </DashboardLayout>
        );
      }
        case "Data Pipeline":
          return (
            <DashboardLayout
              layoutKey="data-pipeline"
              defaultRows={[
                ["datasets", "dataset-inspection"],
                ["feature-store", "manage-tags"]
              ]}
              panels={[
                { id: "datasets", label: "Datasets" },
                { id: "dataset-inspection", label: "Dataset Inspection" },
                { id: "feature-store", label: "Feature Store" },
                { id: "manage-tags", label: "Manage Tags" }
              ]}
            >
            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Foundation</span>
                  <h2>Datasets</h2>
                </div>
                <button onClick={createDataset}>Build Synthetic PIT</button>
              </div>
              <p className="panel-copy">
                Register immutable point-in-time snapshots here, then hand them off to the feature store without mixing
                imported production data with local experiments.
              </p>
              <div className="import-box">
                <strong>Import Real Parquet</strong>
                <p>Point this at a local `.parquet` file or a directory of parquet parts emitted by `findf`.</p>
                <div className="inline-controls wrap">
                  <input
                    value={importName}
                    onChange={(event) => setImportName(event.target.value)}
                    placeholder="Dataset name"
                  />
                  <input
                    className="wide-input"
                    value={importPath}
                    onChange={(event) => setImportPath(event.target.value)}
                    placeholder="C:\\path\\to\\findf\\pit_snapshot.parquet"
                  />
                  <button onClick={importDataset} disabled={!importPath.trim()}>
                    Import Parquet
                  </button>
                </div>
                <div className="control-form">
                  <label>
                    <span>Tags Or Labels</span>
                    <input
                      value={importTagText}
                      onChange={(event) => setImportTagText(event.target.value)}
                      placeholder="production, usa, vendor-a, april refresh"
                    />
                  </label>
                </div>
                <div className="saved-tag-block">
                  <span>Saved Tags</span>
                  {savedDatasetTags.length ? (
                    <div className="tag-choice-list">
                      {savedDatasetTags.map((tag) => {
                        const tagName = String(tag.name);
                        const selected = selectedImportTags.includes(tagName);
                        return (
                          <button
                            key={String(tag.id)}
                            type="button"
                            className={`tag-choice ${selected ? "is-selected" : ""}`}
                            onClick={() => toggleImportTag(tagName)}
                          >
                            {tagName}
                          </button>
                        );
                      })}
                    </div>
                  ) : (
                    <small>No saved tags yet. Add a few in Manage Tags to make imports faster.</small>
                  )}
                </div>
                <div className="selected-tag-preview">
                  <span>Import Preview</span>
                  <TagList tags={importTags} />
                </div>
                <small>Imported datasets also get automatic tags from their ticker set, sector mix, and column coverage.</small>
                <small>
                  Required columns: entity_id, ticker, sector, effective_at, known_at, ingested_at, source_version,
                  open/high/low/close, volume, ev_ebitda, roic, momentum_20d, momentum_60d, sentiment_1d,
                  sentiment_5d, macro_surprise, earnings_signal.
                </small>
                <small>
                  Optional sidecar: `news_events.parquet` with raw headlines/bodies and PIT timestamps for ticker news
                  plus macro/economic news embedding features.
                </small>
              </div>
                <DenseTable
                  columns={datasetColumns}
                  rows={datasets}
                  rowKey={(dataset) => String(dataset.id)}
                  onRowClick={(dataset) => setSelectedDatasetId(String(dataset.id))}
                  selectedRowId={selectedDataset ? String(selectedDataset.id) : null}
                  emptyTitle="No datasets yet"
                  emptyBody="Build a synthetic PIT snapshot or import a `findf` parquet export to start the pipeline."
                  filterPlaceholder="Filter datasets by name, id, path, or tag"
                  defaultSort={{ key: "rows", direction: "desc" }}
                />
              </section>
              <section className="panel">
                <div className="panel-header">
                  <div>
                    <span className="panel-kicker">Inspection Layer</span>
                    <h2>Dataset Assessment</h2>
                  </div>
                  {selectedDatasetAssessment ? <AssessmentPill value={String(selectedDatasetAssessment.data_level ?? selectedDatasetAssessment.status)} /> : null}
                </div>
                <p className="panel-copy">
                  Examine completeness, continuity, and structural PIT quality before this snapshot moves into feature generation or training.
                </p>
                {selectedDataset ? (
                  <div className="assessment-shell">
                    <div className="assessment-header">
                      <div className="assessment-heading">
                        <strong>{selectedDataset.name}</strong>
                        <span>{shortId(selectedDataset.id)}</span>
                      </div>
                      {selectedDatasetAssessment ? <StatusPill value={String(selectedDatasetAssessment.status ?? "unknown")} /> : null}
                    </div>
                    <div className="assessment-source mono-cell">
                      {selectedDataset.summary?.artifacts?.source_path ?? selectedDataset.summary?.artifacts?.raw_path ?? "generated locally"}
                    </div>
                    <div className="metric-grid compact">
                      <MetricCard label="Score" value={formatPercent(selectedDatasetAssessment?.score_pct)} />
                      <MetricCard label="Completeness" value={formatPercent(selectedDatasetAssessment?.completeness_pct)} />
                      <MetricCard label="Continuity" value={formatPercent(selectedDatasetAssessment?.continuity_pct)} />
                      <MetricCard label="Quality" value={formatPercent(selectedDatasetAssessment?.quality_pct)} />
                      <MetricCard label="Missing Sessions" value={pretty(selectedDatasetAssessment?.gaps?.missing_sessions ?? 0)} />
                      <MetricCard label="Duplicate Keys" value={pretty(selectedDatasetAssessment?.gaps?.duplicate_key_rows ?? 0)} />
                      <MetricCard label="PIT Violations" value={pretty(selectedDatasetAssessment?.quality_checks?.timestamp_order_violations ?? 0)} />
                      <MetricCard label="OHLC Issues" value={pretty(selectedDatasetAssessment?.quality_checks?.ohlc_violations ?? 0)} />
                    </div>
                    <div className="assessment-grid">
                      <div className="assessment-section">
                        <span className="panel-kicker">Findings</span>
                        {selectedDatasetIssues.length ? (
                          <div className="assessment-issue-list">
                            {selectedDatasetIssues.map((issue, index) => (
                              <article key={`${issue.title ?? "issue"}-${index}`} className={`assessment-issue severity-${String(issue.severity ?? "warning")}`}>
                                <div className="assessment-issue-header">
                                  <strong>{String(issue.title ?? "Assessment finding")}</strong>
                                  <AssessmentPill value={String(issue.severity ?? "warning")} />
                                </div>
                                <p>{String(issue.detail ?? "")}</p>
                                <small>{String(issue.recommendation ?? "")}</small>
                              </article>
                            ))}
                          </div>
                        ) : (
                          <div className="inline-note">
                            No structural gaps or PIT integrity issues were detected in the selected snapshot.
                          </div>
                        )}
                      </div>
                      <div className="assessment-section">
                        <span className="panel-kicker">Column Coverage</span>
                        <div className="assessment-column-list">
                          {selectedDatasetColumnCoverage.map(({ name, detail }) => (
                            <div key={name} className="assessment-column-row">
                              <div>
                                <strong>{name}</strong>
                                <span>{detail?.required ? "required" : "optional"}</span>
                              </div>
                              <div className="mono-cell">
                                <strong>{formatPercent(detail?.non_null_pct)}</strong>
                                <span>{pretty(detail?.missing_values ?? 0)} missing</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                    {selectedDatasetGapSamples.length ? (
                      <div className="assessment-section">
                        <span className="panel-kicker">Gap Samples</span>
                        <div className="assessment-gap-list">
                          {selectedDatasetGapSamples.map((sample, index) => (
                            <div key={`${sample.instrument ?? "gap"}-${index}`} className="assessment-gap-row">
                              <div>
                                <strong>{String(sample.instrument ?? "Unknown instrument")}</strong>
                                <span>{pretty(sample.missing_sessions ?? 0)} missing sessions</span>
                              </div>
                              <div className="mono-cell">{Array.isArray(sample.sample_dates) ? sample.sample_dates.join(", ") : "n/a"}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <EmptyState
                    title="No dataset selected"
                    body="Import or build a dataset first, then click it to inspect completeness, gap coverage, and quality checks."
                  />
                )}
              </section>
              <section className="panel">
                <div className="panel-header">
                  <div>
                    <span className="panel-kicker">Phase 0-2</span>
                  <h2>Feature Store</h2>
                </div>
                <button onClick={createFeatureSet} disabled={!datasets.length}>
                  Materialize Features
                </button>
              </div>
              <p className="panel-copy">
                Winsorization, cross-sectional normalization, sector neutralization, outlier flags, and final composite
                signals are versioned here with audit traces.
              </p>
              <div className="control-form">
                <label>
                  <span>Dataset Source</span>
                  <select value={featureDatasetId} onChange={(event) => setFeatureDatasetId(event.target.value)}>
                    {datasets.map((dataset) => (
                      <option key={String(dataset.id)} value={String(dataset.id)}>
                        {formatDatasetOptionLabel(dataset)}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <DenseTable
                columns={featureColumns}
                rows={features}
                rowKey={(feature) => String(feature.id)}
                emptyTitle="Feature store is idle"
                emptyBody="Materialize a feature set from the latest PIT dataset to unlock training and testing."
                defaultSort={{ key: "rows", direction: "desc" }}
              />
            </section>
            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Dataset Taxonomy</span>
                  <h2>Manage Tags</h2>
                </div>
              </div>
              <p className="panel-copy">
                Save reusable tags for data vendor, region, freshness, or experiment cohorts, then click them during
                import to label snapshots consistently.
              </p>
              <div className="inline-controls wrap">
                <input
                  className="wide-input"
                  value={newSavedTagName}
                  onChange={(event) => setNewSavedTagName(event.target.value)}
                  placeholder="New saved tag"
                />
                <button onClick={createSavedTag} disabled={!newSavedTagName.trim()}>
                  Save Tag
                </button>
              </div>
              <div className="saved-tag-library">
                {savedDatasetTags.length ? (
                  savedDatasetTags.map((tag) => (
                    <div key={String(tag.id)} className="saved-tag-row">
                      <TagList tags={[String(tag.name)]} />
                      <button className="ghost" onClick={() => deleteSavedTag(String(tag.id), String(tag.name))}>
                        Remove
                      </button>
                    </div>
                  ))
                ) : (
                  <EmptyState
                    title="No saved tags"
                    body="Create a few common labels here so imports can be tagged with a single click."
                  />
                )}
              </div>
            </section>
          </DashboardLayout>
        );
      case "Factor Lab":
        return (
          <section className="panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Research Registry</span>
                <h2>Factor Registry</h2>
              </div>
            </div>
            <p className="panel-copy">
              Inspect the current factor catalog, formulas, and config payloads that feed feature materialization and
              model experiments.
            </p>
            <DenseTable
              columns={factorColumns}
              rows={(catalog?.factors ?? []) as JsonRecord[]}
              rowKey={(factor) => String(factor.id)}
              emptyTitle="No factors loaded"
              emptyBody="Seeded and imported factor definitions will appear here once the catalog is available."
              defaultSort={{ key: "category", direction: "asc" }}
            />
          </section>
        );
      case "Training Studio":
        return (
          <DashboardLayout
            layoutKey="training-studio"
            defaultRows={[
              ["launch-training", "model-catalog"],
              ["training-runs"]
            ]}
            panels={[
              { id: "launch-training", label: "Launch Training" },
              { id: "model-catalog", label: "Model Catalog" },
              { id: "training-runs", label: "Training Runs" }
            ]}
          >
            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Phase 3</span>
                  <h2>Launch Training</h2>
                </div>
                <button onClick={startTraining}>Start Training</button>
              </div>
              <p className="panel-copy">
                Queue training jobs with explicit data, features, model kind, and hyperparameters so the control panel
                becomes the source of truth for run instantiation.
              </p>
              {runtimeCapabilities ? (
                <div className="inline-note">
                  Runtime detects {Array.isArray(runtimeCapabilities.devices) ? runtimeCapabilities.devices.length : 0} device paths.
                  Recommended target: {String(runtimeCapabilities.recommended_compute_target ?? "cpu")}.
                </div>
              ) : null}
              {trainingForm.model_kind === "layered_decision" ? (
                <div className="inline-note">
                  Layered training will use the Master Control selections for each research layer, including model
                  comparisons, preprocessing steps, and per-layer device/runtime preferences.
                </div>
              ) : null}
              <div className="control-form two-column">
                <label>
                  <span>Run Name</span>
                  <input value={trainingForm.name} onChange={(event) => setTrainingForm((current) => ({ ...current, name: event.target.value }))} />
                </label>
                <label>
                  <span>Model Kind</span>
                  <select value={trainingForm.model_kind} onChange={(event) => setTrainingForm((current) => ({ ...current, model_kind: event.target.value }))}>
                    {modelSpecs.map((spec) => (
                      <option key={String(spec.id)} value={String(spec.kind)}>
                        {spec.name} ({spec.kind})
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Dataset</span>
                  <select value={trainingForm.dataset_version_id} onChange={(event) => setTrainingForm((current) => ({ ...current, dataset_version_id: event.target.value }))}>
                    <option value="">Auto-build synthetic dataset</option>
                    {datasets.map((dataset) => (
                      <option key={String(dataset.id)} value={String(dataset.id)}>
                        {formatDatasetOptionLabel(dataset)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Feature Set</span>
                  <select value={trainingForm.feature_set_version_id} onChange={(event) => setTrainingForm((current) => ({ ...current, feature_set_version_id: event.target.value }))}>
                    <option value="">Auto-materialize from dataset</option>
                    {features.map((feature) => (
                      <option key={String(feature.id)} value={String(feature.id)}>
                        {feature.name} ({shortId(feature.id)})
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Epochs</span>
                  <input value={trainingForm.epochs} onChange={(event) => setTrainingForm((current) => ({ ...current, epochs: event.target.value }))} />
                </label>
                <label>
                  <span>Learning Rate</span>
                  <input value={trainingForm.learning_rate} onChange={(event) => setTrainingForm((current) => ({ ...current, learning_rate: event.target.value }))} />
                </label>
                <label>
                  <span>Hidden Dim</span>
                  <input value={trainingForm.hidden_dim} onChange={(event) => setTrainingForm((current) => ({ ...current, hidden_dim: event.target.value }))} />
                </label>
                <label>
                  <span>Checkpoint Freq</span>
                  <input value={trainingForm.checkpoint_frequency} onChange={(event) => setTrainingForm((current) => ({ ...current, checkpoint_frequency: event.target.value }))} />
                </label>
                <label>
                  <span>Horizon Days</span>
                  <input value={trainingForm.horizon_days} onChange={(event) => setTrainingForm((current) => ({ ...current, horizon_days: event.target.value }))} />
                </label>
                <label>
                  <span>Compute Target</span>
                  <select value={trainingForm.compute_target} onChange={(event) => setTrainingForm((current) => ({ ...current, compute_target: event.target.value }))}>
                    {Array.isArray(runtimeCapabilities?.supported_compute_targets)
                      ? (runtimeCapabilities.supported_compute_targets as string[]).map((target) => (
                          <option key={target} value={target}>
                            {target}
                          </option>
                        ))
                      : <option value="auto">auto</option>}
                  </select>
                </label>
                <label>
                  <span>Precision</span>
                  <select value={trainingForm.precision_mode} onChange={(event) => setTrainingForm((current) => ({ ...current, precision_mode: event.target.value }))}>
                    {Array.isArray(runtimeCapabilities?.supported_precision_modes)
                      ? (runtimeCapabilities.supported_precision_modes as string[]).map((mode) => (
                          <option key={mode} value={mode}>
                            {mode}
                          </option>
                        ))
                      : <option value="auto">auto</option>}
                  </select>
                </label>
                <label>
                  <span>Batch Size</span>
                  <input value={trainingForm.batch_size} onChange={(event) => setTrainingForm((current) => ({ ...current, batch_size: event.target.value }))} />
                </label>
                <label>
                  <span>Sequence Length</span>
                  <input value={trainingForm.sequence_length} onChange={(event) => setTrainingForm((current) => ({ ...current, sequence_length: event.target.value }))} />
                </label>
                <label>
                  <span>Grad Clip</span>
                  <input value={trainingForm.gradient_clip_norm} onChange={(event) => setTrainingForm((current) => ({ ...current, gradient_clip_norm: event.target.value }))} />
                </label>
              </div>
              <div className="metric-grid compact">
                <MetricCard label="Selected Dataset" value={trainingForm.dataset_version_id ? shortId(trainingForm.dataset_version_id) : "auto"} />
                <MetricCard label="Selected Features" value={trainingForm.feature_set_version_id ? shortId(trainingForm.feature_set_version_id) : "auto"} />
                <MetricCard label="Epoch Budget" value={trainingForm.epochs} />
                <MetricCard label="Learning Rate" value={trainingForm.learning_rate} />
                <MetricCard label="Compute Target" value={trainingForm.compute_target} />
                <MetricCard label="Batch Size" value={trainingForm.batch_size} />
              </div>
            </section>
            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Available Specs</span>
                  <h2>Model Catalog</h2>
                </div>
              </div>
              <DenseTable
                columns={modelSpecColumns}
                rows={modelSpecs}
                rowKey={(spec) => String(spec.id)}
                emptyTitle="No model specs"
                emptyBody="Seeded model specifications will show up here for training."
                defaultSort={{ key: "name", direction: "asc" }}
              />
            </section>
            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Live Queue</span>
                  <h2>Training Runs</h2>
                </div>
              </div>
              <DenseTable
                columns={trainingRunColumns}
                rows={trainingRuns}
                rowKey={(run) => String(run.id)}
                emptyTitle="No training runs"
                emptyBody="Start a model training run to stream epochs, checkpoints, metrics, and artifacts here."
                onRowClick={(run) => setSelectedRun({ kind: "training", id: String(run.id) })}
                selectedRowId={selectedRun?.kind === "training" ? selectedRun.id : null}
                filterPlaceholder="Filter runs by id, stage, or state"
                defaultSort={{ key: "updated", direction: "desc" }}
              />
            </section>
          </DashboardLayout>
        );
      case "Testing Console":
        return (
          <DashboardLayout
            layoutKey="testing-console"
            defaultRows={[
              ["launch-testing", "frozen-models"],
              ["testing-runs"]
            ]}
            panels={[
              { id: "launch-testing", label: "Launch Testing" },
              { id: "frozen-models", label: "Frozen Models" },
              { id: "testing-runs", label: "Testing Runs" }
            ]}
          >
            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Phase 4-7</span>
                  <h2>Launch Testing</h2>
                </div>
                <button onClick={startTesting} disabled={!models.length}>
                  Start Testing
                </button>
              </div>
              <p className="panel-copy">
                Instantiate testing runs with an explicit frozen model, feature source, stress budget, and execution
                profile so monitoring is reproducible from the UI.
              </p>
              <div className="control-form two-column">
                <label>
                  <span>Run Name</span>
                  <input value={testingForm.name} onChange={(event) => setTestingForm((current) => ({ ...current, name: event.target.value }))} />
                </label>
                <label>
                  <span>Model Version</span>
                  <select value={testingForm.model_version_id} onChange={(event) => setTestingForm((current) => ({ ...current, model_version_id: event.target.value }))}>
                    {models.map((model) => (
                      <option key={String(model.id)} value={String(model.id)}>
                        {model.name} ({shortId(model.id)})
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Feature Set</span>
                  <select value={testingForm.feature_set_version_id} onChange={(event) => setTestingForm((current) => ({ ...current, feature_set_version_id: event.target.value }))}>
                    <option value="">Use model-linked feature set</option>
                    {features.map((feature) => (
                      <option key={String(feature.id)} value={String(feature.id)}>
                        {feature.name} ({shortId(feature.id)})
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Execution Mode</span>
                  <select value={testingForm.execution_mode} onChange={(event) => setTestingForm((current) => ({ ...current, execution_mode: event.target.value }))}>
                    <option value="paper">paper</option>
                  </select>
                </label>
                <label>
                  <span>Rebalance Decile</span>
                  <input value={testingForm.rebalance_decile} onChange={(event) => setTestingForm((current) => ({ ...current, rebalance_decile: event.target.value }))} />
                </label>
                <label>
                  <span>Stress Iterations</span>
                  <input value={testingForm.stress_iterations} onChange={(event) => setTestingForm((current) => ({ ...current, stress_iterations: event.target.value }))} />
                </label>
              </div>
              <div className="metric-grid compact">
                <MetricCard label="Selected Model" value={testingForm.model_version_id ? shortId(testingForm.model_version_id) : "n/a"} />
                <MetricCard label="Feature Source" value={testingForm.feature_set_version_id ? shortId(testingForm.feature_set_version_id) : "model-linked"} />
                <MetricCard label="Stress Runs" value={testingForm.stress_iterations} />
                <MetricCard label="Rebalance" value={testingForm.rebalance_decile} />
              </div>
            </section>
            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Frozen Inventory</span>
                  <h2>Frozen Models</h2>
                </div>
              </div>
              <DenseTable
                columns={modelColumns}
                rows={models}
                rowKey={(model) => String(model.id)}
                emptyTitle="No frozen models"
                emptyBody="Finish a training run to register a model version before testing."
                filterPlaceholder="Filter models by name or status"
                defaultSort={{ key: "sharpe", direction: "desc" }}
              />
            </section>
            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Out Of Sample</span>
                  <h2>Testing Runs</h2>
                </div>
              </div>
              <DenseTable
                columns={testingRunColumns}
                rows={testingRuns}
                rowKey={(run) => String(run.id)}
                emptyTitle="No testing runs"
                emptyBody="Select a registered model and launch a testing session to inspect results here."
                onRowClick={(run) => setSelectedRun({ kind: "testing", id: String(run.id) })}
                selectedRowId={selectedRun?.kind === "testing" ? selectedRun.id : null}
                filterPlaceholder="Filter runs by id, stage, or state"
                defaultSort={{ key: "updated", direction: "desc" }}
              />
            </section>
          </DashboardLayout>
        );
      case "Risk Lab":
        return (
          <section className="panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Stress Surface</span>
                <h2>Risk Metrics</h2>
              </div>
              <div className="panel-header-actions">{renderPanelCollapseButton("main-risk-metrics", "Risk Metrics")}</div>
            </div>
            {isPanelCollapsed("main-risk-metrics")
              ? renderCollapsedPanelNote("Risk metrics are tucked away for now. Expand this section to inspect the latest surface.")
              : (
                  <DenseTable
                    columns={metricColumns}
                    rows={runMetrics.filter((metric) => metric.group_name === "risk")}
                    rowKey={(metric) => String(metric.id)}
                    emptyTitle="No risk metrics"
                    emptyBody="Run a testing session to populate risk and stress metrics."
                    defaultSort={{ key: "value", direction: "desc" }}
                  />
                )}
          </section>
        );
      case "Portfolio/Backtest":
        return (
          <section className="panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Allocator</span>
                <h2>Portfolio Metrics</h2>
              </div>
              <div className="panel-header-actions">{renderPanelCollapseButton("main-portfolio-metrics", "Portfolio Metrics")}</div>
            </div>
            {isPanelCollapsed("main-portfolio-metrics")
              ? renderCollapsedPanelNote("Portfolio metrics are hidden until you need them. Expand to review the current backtest readout.")
              : (
                  <DenseTable
                    columns={metricColumns}
                    rows={runMetrics.filter((metric) => metric.group_name === "testing")}
                    rowKey={(metric) => String(metric.id)}
                    emptyTitle="No portfolio metrics"
                    emptyBody="Backtest metrics will appear here after a testing session runs."
                    defaultSort={{ key: "value", direction: "desc" }}
                  />
                )}
          </section>
        );
      case "Execution Simulator":
        return (
          <section className="panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Paper Execution</span>
                <h2>Execution Metrics</h2>
              </div>
              <div className="panel-header-actions">{renderPanelCollapseButton("main-execution-metrics", "Execution Metrics")}</div>
            </div>
            {isPanelCollapsed("main-execution-metrics")
              ? renderCollapsedPanelNote("Execution metrics are collapsed. Expand to inspect the latest paper-execution output.")
              : (
                  <DenseTable
                    columns={metricColumns}
                    rows={runMetrics.filter((metric) => metric.group_name === "execution")}
                    rowKey={(metric) => String(metric.id)}
                    emptyTitle="No execution metrics"
                    emptyBody="Paper execution metrics will populate here after testing."
                    defaultSort={{ key: "value", direction: "desc" }}
                  />
                )}
          </section>
        );
      case "Model Registry":
        return (
          <section className="panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Governance</span>
                <h2>Model Registry</h2>
              </div>
              <div className="panel-header-actions">{renderPanelCollapseButton("main-model-registry", "Model Registry")}</div>
            </div>
            {isPanelCollapsed("main-model-registry")
              ? renderCollapsedPanelNote("The model registry is collapsed. Expand it to review promotion and rejection decisions.")
              : (
                  <>
                    <p className="panel-copy">
                      Promote or reject trained models after reviewing checkpoints, out-of-sample metrics, and factual
                      calculation traces.
                    </p>
                    <DenseTable
                      columns={modelRegistryColumns}
                      rows={models}
                      rowKey={(model) => String(model.id)}
                      emptyTitle="Registry is empty"
                      emptyBody="Training a model creates a version here, where it can be promoted or rejected."
                      filterPlaceholder="Filter registry by name, id, or status"
                      defaultSort={{ key: "sharpe", direction: "desc" }}
                    />
                  </>
                )}
          </section>
        );
      case "Monitoring":
      default:
        return (
          <DashboardLayout
            layoutKey="monitoring"
            defaultRows={[["monitoring", "runtime-self-check"]]}
            panels={[
              { id: "monitoring", label: "Monitoring" },
              { id: "runtime-self-check", label: "Runtime Self-Check" }
            ]}
          >
            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Phase 8</span>
                  <h2>Monitoring</h2>
                </div>
              </div>
              <p className="panel-copy">
                Review control-plane health, recent performance, retrain triggers, and the last monitoring snapshot in one
                place.
              </p>
              <div className="metric-grid">
                <MetricCard label="Run Success" value={pretty(monitoring?.run_success_rate)} />
                <MetricCard label="Retrain Triggers" value={pretty(monitoring?.retrain_trigger_count)} />
                <MetricCard label="Latest Sharpe" value={pretty(monitoring?.latest_metrics?.sharpe)} />
                <MetricCard label="Latest Rank IC" value={pretty(monitoring?.latest_metrics?.rank_ic)} />
              </div>
              <pre className="json-block">{JSON.stringify(monitoring, null, 2)}</pre>
            </section>
            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-kicker">Hardware Probe</span>
                  <h2>Runtime Self-Check</h2>
                </div>
                <button onClick={runRuntimeSelfCheck} disabled={runtimeSelfCheckRunning}>
                  {runtimeSelfCheckRunning ? "Running..." : "Run Self-Check"}
                </button>
              </div>
              <p className="panel-copy">
                Validate tensor allocation, forward pass, backward pass, and one optimizer step on the selected runtime
                path before launching larger training jobs.
              </p>
              <div className="control-form two-column">
                <label>
                  <span>Compute Target</span>
                  <select
                    value={runtimeSelfCheckForm.compute_target}
                    onChange={(event) => setRuntimeSelfCheckForm((current) => ({ ...current, compute_target: event.target.value }))}
                  >
                    {Array.isArray(runtimeCapabilities?.supported_compute_targets)
                      ? (runtimeCapabilities.supported_compute_targets as string[]).map((target) => (
                          <option key={target} value={target}>
                            {target}
                          </option>
                        ))
                      : <option value="auto">auto</option>}
                  </select>
                </label>
                <label>
                  <span>Precision</span>
                  <select
                    value={runtimeSelfCheckForm.precision_mode}
                    onChange={(event) => setRuntimeSelfCheckForm((current) => ({ ...current, precision_mode: event.target.value }))}
                  >
                    {Array.isArray(runtimeCapabilities?.supported_precision_modes)
                      ? (runtimeCapabilities.supported_precision_modes as string[]).map((mode) => (
                          <option key={mode} value={mode}>
                            {mode}
                          </option>
                        ))
                      : <option value="auto">auto</option>}
                  </select>
                </label>
                <label>
                  <span>Model Kind</span>
                  <select
                    value={runtimeSelfCheckForm.model_kind}
                    onChange={(event) => setRuntimeSelfCheckForm((current) => ({ ...current, model_kind: event.target.value }))}
                  >
                    <option value="pytorch_mlp">pytorch_mlp</option>
                    <option value="gru">gru</option>
                    <option value="temporal_cnn">temporal_cnn</option>
                  </select>
                </label>
                <label>
                  <span>Input Dim</span>
                  <input value={runtimeSelfCheckForm.input_dim} onChange={(event) => setRuntimeSelfCheckForm((current) => ({ ...current, input_dim: event.target.value }))} />
                </label>
                <label>
                  <span>Batch Size</span>
                  <input value={runtimeSelfCheckForm.batch_size} onChange={(event) => setRuntimeSelfCheckForm((current) => ({ ...current, batch_size: event.target.value }))} />
                </label>
                <label>
                  <span>Sequence Length</span>
                  <input value={runtimeSelfCheckForm.sequence_length} onChange={(event) => setRuntimeSelfCheckForm((current) => ({ ...current, sequence_length: event.target.value }))} />
                </label>
                <label>
                  <span>Grad Clip</span>
                  <input value={runtimeSelfCheckForm.gradient_clip_norm} onChange={(event) => setRuntimeSelfCheckForm((current) => ({ ...current, gradient_clip_norm: event.target.value }))} />
                </label>
              </div>
              <div className="metric-grid compact">
                <MetricCard label="Recommended" value={runtimeCapabilities?.recommended_compute_target ?? "cpu"} />
                <MetricCard label="Result" value={runtimeSelfCheckResult ? (runtimeSelfCheckResult.success ? "pass" : "fail") : "not run"} />
                <MetricCard label="Target Matched" value={runtimeSelfCheckResult ? (runtimeSelfCheckResult.requested_target_satisfied ? "yes" : "no") : "n/a"} />
                <MetricCard label="Resolved Target" value={runtimeSelfCheckResult?.resolved_runtime?.resolved_compute_target ?? "n/a"} />
                <MetricCard label="Elapsed ms" value={pretty(runtimeSelfCheckResult?.metrics?.elapsed_ms)} />
              </div>
              {Array.isArray(runtimeCapabilities?.notes) && runtimeCapabilities?.notes?.length ? (
                <pre className="json-block compact-json">{JSON.stringify(runtimeCapabilities.notes, null, 2)}</pre>
              ) : null}
              {runtimeSelfCheckResult ? (
                <>
                  <div className="contract-grid">
                    <article className="contract-card">
                      <small>Checks</small>
                      <p>{compactText(runtimeSelfCheckResult.checks, 180)}</p>
                    </article>
                    <article className="contract-card">
                      <small>Resolved Runtime</small>
                      <p>{compactText(runtimeSelfCheckResult.resolved_runtime, 180)}</p>
                    </article>
                    <article className="contract-card">
                      <small>Errors</small>
                      <p>{compactText(runtimeSelfCheckResult.errors ?? [], 180)}</p>
                    </article>
                  </div>
                  <pre className="json-block">{JSON.stringify(runtimeSelfCheckResult, null, 2)}</pre>
                </>
              ) : null}
            </section>
          </DashboardLayout>
        );
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Local Quant Factory</p>
          <h1>Master Control Panel</h1>
          <p>Training and testing stay isolated while metrics, traces, and run state stream live.</p>
          <div className="hero-tags">
            <StatusPill value="training isolated" />
            <StatusPill value="testing frozen" />
            <StatusPill value="live traces" />
          </div>
        </div>
        <div className="hero-stack">
          <article className="hero-block">
            <small>Selected run</small>
            <strong>{selectedRun ? shortId(selectedRun.id) : "Standby"}</strong>
            <p>
              {selectedRunData
                ? `${selectedRun?.kind ?? "run"} flow - ${selectedRunData.current_stage ?? "awaiting stage"}`
                : "Choose a run to inspect metrics, logs, formulas, artifacts, and execution traces."}
            </p>
          </article>
          <div className="hero-metrics">
            <MetricCard label="Datasets" value={overview?.counts?.dataset_versions} />
            <MetricCard label="Feature Sets" value={overview?.counts?.feature_set_versions} />
            <MetricCard label="Models" value={overview?.counts?.model_versions} />
            <MetricCard label="Latest Slippage" value={monitoring?.latest_metrics?.avg_slippage_bps} />
          </div>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

        <section className="command-strip">
          <article className="command-card">
            <small>Dataset track</small>
            <strong>{latestDataset?.name ?? "No PIT snapshot"}</strong>
            <span>
              {latestDataset
                ? `${pretty(latestDataset.summary?.rows)} rows loaded / ${String(latestDatasetAssessment?.data_level ?? "n/a")} data level`
                : "Import or build a dataset"}
            </span>
          </article>
        <article className="command-card">
          <small>Feature state</small>
          <strong>{latestFeatureSet?.name ?? "Feature store idle"}</strong>
          <span>{latestFeatureSet ? `${pretty(latestFeatureSet.summary?.rows)} rows ready` : "Materialize features next"}</span>
        </article>
        <article className="command-card">
          <small>Run queues</small>
          <strong>{activeTrainingCount} training / {activeTestingCount} testing</strong>
          <span>Training and testing stay separate end to end</span>
        </article>
        <article className="command-card">
          <small>Registry</small>
          <strong>{promotedModelCount} promoted</strong>
          <span>{latestModel ? `${latestModel.name} is the latest version` : "No registered models yet"}</span>
        </article>
      </section>

      <nav className="tab-strip">
        {tabs.map((tab) => (
          <button key={tab} className={tab === activeTab ? "active" : ""} onClick={() => setActiveTab(tab)}>
            {tab}
          </button>
        ))}
      </nav>

      <main className="workspace">
        <section className="workspace-main">{renderMainPanel()}</section>
        <aside className="workspace-side">
          <section className="panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Operations</span>
                <h2>Run Control</h2>
              </div>
              <div className="panel-header-actions">
                {selectedRunData?.state ? <StatusPill value={String(selectedRunData.state)} /> : null}
                {renderPanelCollapseButton("side-run-control", "Run Control")}
              </div>
            </div>
            {isPanelCollapsed("side-run-control")
              ? renderCollapsedPanelNote("Run control is collapsed. Expand it to manage pause, resume, stop, and override actions.")
              : (
                  <>
                    <div className="run-summary">
                      <span>{selectedRun?.kind ?? "No run selected"}</span>
                      <strong>{selectedRun ? shortId(selectedRun.id) : "Select a run"}</strong>
                      <span>{selectedRunData?.current_stage ?? "n/a"}</span>
                      <span>{selectedRunData?.updated_at ? formatTimestamp(selectedRunData.updated_at) : "Waiting for updates"}</span>
                    </div>
                    <div className="progress-block">
                      <div className="progress-meta">
                        <span>Progress</span>
                        <strong>{selectedProgress !== null ? `${selectedProgress.toFixed(0)}%` : "n/a"}</strong>
                      </div>
                      <div className="progress-track">
                        <div className="progress-bar" style={{ width: `${selectedProgress ?? 0}%` }} />
                      </div>
                    </div>
                    <div className="inline-controls">
                      <button onClick={() => controlRun("pause")} disabled={!selectedRun}>
                        Pause
                      </button>
                      <button onClick={() => controlRun("resume")} disabled={!selectedRun}>
                        Resume
                      </button>
                      <button className="ghost" onClick={() => controlRun("stop")} disabled={!selectedRun}>
                        Stop
                      </button>
                    </div>
                    <div className="inline-controls">
                      <input value={overrideValue} onChange={(event) => setOverrideValue(event.target.value)} />
                      <button onClick={applyOverride} disabled={selectedRun?.kind !== "training"}>
                        Queue LR Override
                      </button>
                    </div>
                  </>
                )}
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Signal Readout</span>
                <h2>Selected Metrics</h2>
              </div>
              <div className="panel-header-actions">{renderPanelCollapseButton("side-selected-metrics", "Selected Metrics")}</div>
            </div>
            {isPanelCollapsed("side-selected-metrics")
              ? renderCollapsedPanelNote("Selected metrics are hidden. Expand the section to compare the current signal readout.")
              : (
                  <DenseTable
                    columns={metricColumns}
                    rows={primaryMetrics}
                    rowKey={(metric) => String(metric.id)}
                    emptyTitle="No metrics loaded"
                    emptyBody="Select a run to load training, risk, testing, and execution metrics into the sidebar."
                    defaultSort={{ key: "value", direction: "desc" }}
                  />
                )}
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Timeline</span>
                <h2>Live Events</h2>
              </div>
              <div className="panel-header-actions">{renderPanelCollapseButton("side-live-events", "Live Events")}</div>
            </div>
            {isPanelCollapsed("side-live-events")
              ? renderCollapsedPanelNote("Live events are collapsed. Expand to watch the stream as runs report new activity.")
              : (
                  <DenseTable
                    columns={eventColumns}
                    rows={recentEvents}
                    rowKey={(event) => String(event.id)}
                    emptyTitle="No event stream yet"
                    emptyBody="Run events will appear here as soon as a training or testing flow starts emitting progress."
                    defaultSort={{ key: "time", direction: "desc" }}
                  />
                )}
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Audit Trail</span>
                <h2>Calculations</h2>
              </div>
              <div className="panel-header-actions">{renderPanelCollapseButton("side-calculations", "Calculations")}</div>
            </div>
            {isPanelCollapsed("side-calculations")
              ? renderCollapsedPanelNote("Calculation traces are tucked away. Expand this panel to audit the latest formulas and outputs.")
              : (
                  <DenseTable
                    columns={traceColumns}
                    rows={recentTraces}
                    rowKey={(trace) => String(trace.id)}
                    emptyTitle="No calculations loaded"
                    emptyBody="Calculation traces will show the latest formulas and outputs for the selected run."
                    defaultSort={{ key: "label", direction: "asc" }}
                  />
                )}
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Outputs</span>
                <h2>Artifacts</h2>
              </div>
              <div className="panel-header-actions">{renderPanelCollapseButton("side-artifacts", "Artifacts")}</div>
            </div>
            {isPanelCollapsed("side-artifacts")
              ? renderCollapsedPanelNote("Artifacts are collapsed. Expand this section to review checkpoints, reports, and exports.")
              : (
                  <DenseTable
                    columns={artifactColumns}
                    rows={runArtifacts}
                    rowKey={(artifact) => String(artifact.id)}
                    emptyTitle="No artifacts yet"
                    emptyBody="Checkpoints, reports, and exported summaries for the selected run will land here."
                    defaultSort={{ key: "type", direction: "asc" }}
                  />
                )}
          </section>
        </aside>
      </main>
    </div>
  );
}
