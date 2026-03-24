import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type { JsonRecord, OverviewResponse } from "./types";

const tabs = [
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

function MetricCard({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value ?? "n/a"}</strong>
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

export default function App() {
  const [activeTab, setActiveTab] = useState(tabs[0]);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [catalog, setCatalog] = useState<JsonRecord | null>(null);
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
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      const [overviewData, catalogData, datasetData, featureData, modelData, trainingData, testingData, monitoringData] =
        await Promise.all([
          api.get<OverviewResponse>("/api/overview"),
          api.get<JsonRecord>("/api/catalog"),
          api.get<JsonRecord[]>("/api/datasets"),
          api.get<JsonRecord[]>("/api/features"),
          api.get<JsonRecord[]>("/api/model-versions"),
          api.get<JsonRecord[]>("/api/training-runs"),
          api.get<JsonRecord[]>("/api/testing-runs"),
          api.get<JsonRecord>("/api/monitoring")
        ]);
      setOverview(overviewData);
      setCatalog(catalogData);
      setDatasets(datasetData);
      setFeatures(featureData);
      setModels(modelData);
      setTrainingRuns(trainingData);
      setTestingRuns(testingData);
      setMonitoring(monitoringData);
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
    const source = new EventSource(`${api.baseUrl}/api/stream/${selectedRun.kind}/${selectedRun.id}`);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as JsonRecord;
      setRunEvents((current) => [...current.filter((item) => item.id !== event.id), event].sort((a, b) => a.id - b.id));
    };
    source.onerror = () => source.close();
    return () => source.close();
  }, [selectedRun]);

  const selectedRunData = useMemo(() => {
    if (!selectedRun) return null;
    return selectedRun.kind === "training"
      ? trainingRuns.find((run) => run.id === selectedRun.id) ?? null
      : testingRuns.find((run) => run.id === selectedRun.id) ?? null;
  }, [selectedRun, testingRuns, trainingRuns]);

  async function createDataset() {
    await api.post("/api/datasets", {});
    refresh();
  }

  async function createFeatureSet() {
    if (!datasets[0]) return;
    await api.post("/api/features", {
      dataset_version_id: datasets[0].id
    });
    refresh();
  }

  async function startTraining() {
    const response = await api.post<JsonRecord>("/api/training-runs", {
      dataset_version_id: datasets[0]?.id,
      feature_set_version_id: features[0]?.id,
      model_kind: "pytorch_mlp",
      name: "Master Control Training",
      epochs: 6,
      learning_rate: Number(overrideValue) || 0.02
    });
    setSelectedRun({ kind: "training", id: response.id });
    refresh();
  }

  async function startTesting() {
    if (!models[0]) return;
    const response = await api.post<JsonRecord>("/api/testing-runs", {
      model_version_id: models[0].id,
      feature_set_version_id: features[0]?.id
    });
    setSelectedRun({ kind: "testing", id: response.id });
    refresh();
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

  const latestTestingMetrics = testingRuns[0] ? runMetrics.filter((metric) => metric.run_id === testingRuns[0].id) : runMetrics;

  function renderMainPanel() {
    switch (activeTab) {
      case "Data Pipeline":
        return (
          <div className="panel-grid">
            <section className="panel">
              <div className="panel-header">
                <h2>Datasets</h2>
                <button onClick={createDataset}>Build PIT Dataset</button>
              </div>
              {datasets.map((dataset) => (
                <article key={dataset.id} className="row-card">
                  <strong>{dataset.name}</strong>
                  <span>{dataset.id}</span>
                  <span>{pretty(dataset.summary?.rows)} rows</span>
                </article>
              ))}
            </section>
            <section className="panel">
              <div className="panel-header">
                <h2>Feature Store</h2>
                <button onClick={createFeatureSet} disabled={!datasets.length}>
                  Materialize Features
                </button>
              </div>
              {features.map((feature) => (
                <article key={feature.id} className="row-card">
                  <strong>{feature.name}</strong>
                  <span>{feature.id}</span>
                  <span>{pretty(feature.summary?.rows)} usable rows</span>
                </article>
              ))}
            </section>
          </div>
        );
      case "Factor Lab":
        return (
          <section className="panel">
            <div className="panel-header">
              <h2>Factor Registry</h2>
            </div>
            <div className="factor-grid">
              {(catalog?.factors ?? []).map((factor: JsonRecord) => (
                <article key={factor.id} className="factor-card">
                  <small>{factor.category}</small>
                  <h3>{factor.name}</h3>
                  <code>{factor.formula}</code>
                  <pre>{JSON.stringify(factor.config, null, 2)}</pre>
                </article>
              ))}
            </div>
          </section>
        );
      case "Training Studio":
        return (
          <div className="panel-grid">
            <section className="panel">
              <div className="panel-header">
                <h2>Model Catalog</h2>
                <div className="inline-controls">
                  <input value={overrideValue} onChange={(event) => setOverrideValue(event.target.value)} />
                  <button onClick={startTraining}>Start Training</button>
                </div>
              </div>
              {(catalog?.model_specs ?? []).map((spec: JsonRecord) => (
                <article key={spec.id} className="row-card">
                  <strong>{spec.name}</strong>
                  <span>{spec.kind}</span>
                  <span>{spec.description}</span>
                </article>
              ))}
            </section>
            <section className="panel">
              <div className="panel-header">
                <h2>Training Runs</h2>
              </div>
              {trainingRuns.map((run) => (
                <button key={run.id} className={`row-card selectable ${selectedRun?.id === run.id ? "selected" : ""}`} onClick={() => setSelectedRun({ kind: "training", id: run.id })}>
                  <strong>{run.id}</strong>
                  <span>{run.state}</span>
                  <span>{run.current_stage}</span>
                </button>
              ))}
            </section>
          </div>
        );
      case "Testing Console":
        return (
          <div className="panel-grid">
            <section className="panel">
              <div className="panel-header">
                <h2>Frozen Models</h2>
                <button onClick={startTesting} disabled={!models.length}>
                  Start Testing
                </button>
              </div>
              {models.map((model) => (
                <article key={model.id} className="row-card">
                  <strong>{model.name}</strong>
                  <span>{model.status}</span>
                  <span>{pretty(model.metrics?.sharpe)}</span>
                </article>
              ))}
            </section>
            <section className="panel">
              <div className="panel-header">
                <h2>Testing Runs</h2>
              </div>
              {testingRuns.map((run) => (
                <button key={run.id} className={`row-card selectable ${selectedRun?.id === run.id ? "selected" : ""}`} onClick={() => setSelectedRun({ kind: "testing", id: run.id })}>
                  <strong>{run.id}</strong>
                  <span>{run.state}</span>
                  <span>{run.current_stage}</span>
                </button>
              ))}
            </section>
          </div>
        );
      case "Risk Lab":
        return (
          <section className="panel">
            <div className="panel-header">
              <h2>Risk Metrics</h2>
            </div>
            <div className="metric-grid">
              {runMetrics
                .filter((metric) => metric.group_name === "risk")
                .map((metric) => <MetricCard key={metric.id} label={metric.name} value={pretty(metric.value)} />)}
            </div>
          </section>
        );
      case "Portfolio/Backtest":
        return (
          <section className="panel">
            <div className="panel-header">
              <h2>Portfolio Metrics</h2>
            </div>
            <div className="metric-grid">
              {runMetrics
                .filter((metric) => metric.group_name === "testing")
                .map((metric) => <MetricCard key={metric.id} label={metric.name} value={pretty(metric.value)} />)}
            </div>
          </section>
        );
      case "Execution Simulator":
        return (
          <section className="panel">
            <div className="panel-header">
              <h2>Execution Metrics</h2>
            </div>
            <div className="metric-grid">
              {runMetrics
                .filter((metric) => metric.group_name === "execution")
                .map((metric) => <MetricCard key={metric.id} label={metric.name} value={pretty(metric.value)} />)}
            </div>
          </section>
        );
      case "Model Registry":
        return (
          <section className="panel">
            <div className="panel-header">
              <h2>Model Registry</h2>
            </div>
            {models.map((model) => (
              <article key={model.id} className="row-card wide">
                <div>
                  <strong>{model.name}</strong>
                  <span>{model.id}</span>
                </div>
                <div>
                  <span>{model.status}</span>
                  <span>Sharpe {pretty(model.metrics?.sharpe)}</span>
                </div>
                <div className="inline-controls">
                  <button onClick={() => updateModelStatus(model.id, "promote")}>Promote</button>
                  <button className="ghost" onClick={() => updateModelStatus(model.id, "reject")}>
                    Reject
                  </button>
                </div>
              </article>
            ))}
          </section>
        );
      case "Monitoring":
      default:
        return (
          <section className="panel">
            <div className="panel-header">
              <h2>Monitoring</h2>
            </div>
            <div className="metric-grid">
              <MetricCard label="Run Success" value={pretty(monitoring?.run_success_rate)} />
              <MetricCard label="Retrain Triggers" value={pretty(monitoring?.retrain_trigger_count)} />
              <MetricCard label="Latest Sharpe" value={pretty(monitoring?.latest_metrics?.sharpe)} />
              <MetricCard label="Latest Rank IC" value={pretty(monitoring?.latest_metrics?.rank_ic)} />
            </div>
            <pre className="json-block">{JSON.stringify(monitoring, null, 2)}</pre>
          </section>
        );
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Local Quant Factory</p>
          <h1>Master Control Panel</h1>
          <p>
            Training and testing are separated, every step is logged, and the latest factors, metrics, traces, and
            execution diagnostics stay visible while runs progress.
          </p>
        </div>
        <div className="hero-metrics">
          <MetricCard label="Datasets" value={overview?.counts?.dataset_versions} />
          <MetricCard label="Feature Sets" value={overview?.counts?.feature_set_versions} />
          <MetricCard label="Models" value={overview?.counts?.model_versions} />
          <MetricCard label="Latest Slippage" value={monitoring?.latest_metrics?.avg_slippage_bps} />
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

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
              <h2>Run Control</h2>
            </div>
            <div className="run-summary">
              <span>{selectedRun?.kind ?? "No run selected"}</span>
              <strong>{selectedRun?.id ?? "Select a run"}</strong>
              <span>{selectedRunData?.state ?? "idle"}</span>
              <span>{selectedRunData?.current_stage ?? "n/a"}</span>
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
          </section>

          <section className="panel">
            <div className="panel-header">
              <h2>Live Events</h2>
            </div>
            <div className="event-list">
              {runEvents.slice(-16).map((event) => (
                <article key={event.id} className={`event-card severity-${event.severity}`}>
                  <small>{event.phase} / {event.stage}</small>
                  <strong>{event.message}</strong>
                  <span>{pretty(event.progress_pct)}%</span>
                </article>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <h2>Calculations</h2>
            </div>
            <div className="trace-list">
              {runTraces.slice(-6).map((trace) => (
                <article key={trace.id} className="trace-card">
                  <small>{trace.formula_id}</small>
                  <strong>{trace.label}</strong>
                  <pre>{JSON.stringify(trace.output, null, 2)}</pre>
                </article>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <h2>Artifacts</h2>
            </div>
            {runArtifacts.map((artifact) => (
              <article key={artifact.id} className="row-card">
                <strong>{artifact.artifact_type}</strong>
                <span>{artifact.path}</span>
              </article>
            ))}
          </section>
        </aside>
      </main>
    </div>
  );
}
