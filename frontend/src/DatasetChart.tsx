import { useMemo } from "react";
import type { JsonRecord } from "./types";

const WIDTH = 960;
const HEIGHT = 420;
const PAD_LEFT = 52;
const PAD_RIGHT = 58;
const PAD_TOP = 18;
const PAD_BOTTOM = 56;
const VOLUME_BAND = 62;
const NEWS_LANES = {
  ticker: HEIGHT - PAD_BOTTOM + 10,
  macro: HEIGHT - PAD_BOTTOM + 24,
  event: HEIGHT - PAD_BOTTOM + 38
};

const WINDOW_DAYS: Record<string, number | null> = {
  "3M": 90,
  "6M": 180,
  "1Y": 365,
  Full: null
};

const LAYER_COLORS = ["#ff8f70", "#7fd4ff", "#f4d35e", "#a2ff95", "#f7aef8", "#99b3ff"];

function toTime(value: unknown): number | null {
  if (typeof value !== "string") {
    return null;
  }
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : null;
}

function toNumber(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function linePath(
  points: Array<{ x: number; y: number }>,
): string {
  if (!points.length) {
    return "";
  }

  return points.map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(" ");
}

function formatNumber(value: number | null, digits = 2): string {
  if (value === null || !Number.isFinite(value)) {
    return "n/a";
  }
  return value.toFixed(digits);
}

function formatPercent(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "n/a";
  }
  return `${(value * 100).toFixed(2)}%`;
}

export function DatasetChart({
  payload,
  overlayState,
  windowMode
}: {
  payload: JsonRecord | null;
  overlayState: Record<string, boolean>;
  windowMode: string;
}) {
  const derived = useMemo(() => {
    if (!payload) {
      return null;
    }

    const rawPriceSeries = Array.isArray(payload.price_series) ? (payload.price_series as JsonRecord[]) : [];
    const rawPredictionSeries = Array.isArray(payload.prediction_series) ? (payload.prediction_series as JsonRecord[]) : [];
    const rawNewsEvents = Array.isArray(payload.news_events) ? (payload.news_events as JsonRecord[]) : [];
    const rawEventMarkers = Array.isArray(payload.event_markers) ? (payload.event_markers as JsonRecord[]) : [];
    const layerScoreColumns = Array.isArray(payload.layer_score_columns) ? (payload.layer_score_columns as JsonRecord[]) : [];

    const priceSeries = rawPriceSeries
      .map((item) => ({
        ...item,
        time: toTime(item.effective_at),
        closeValue: toNumber(item.close),
        lowValue: toNumber(item.low),
        highValue: toNumber(item.high),
        volumeValue: toNumber(item.volume)
      }))
      .filter((item) => item.time !== null && item.closeValue !== null);

    const predictionSeries = rawPredictionSeries
      .map((item) => ({
        ...item,
        time: toTime(item.effective_at),
        predictedValue: toNumber(item.predicted_return),
        forwardValue: toNumber(item.forward_return)
      }))
      .filter((item) => item.time !== null);

    const newsEvents = rawNewsEvents
      .map((item) => ({
        ...item,
        time: toTime(item.known_at)
      }))
      .filter((item) => item.time !== null);

    const eventMarkers = rawEventMarkers
      .map((item) => ({
        ...item,
        time: toTime(item.effective_at),
        numericValue: toNumber(item.value)
      }))
      .filter((item) => item.time !== null);

    const allTimes = [
      ...priceSeries.map((item) => item.time as number),
      ...predictionSeries.map((item) => item.time as number)
    ];
    if (!allTimes.length) {
      return null;
    }

    const maxTime = Math.max(...allTimes);
    const windowDays = WINDOW_DAYS[windowMode] ?? null;
    const minVisibleTime = windowDays ? maxTime - windowDays * 24 * 60 * 60 * 1000 : Math.min(...allTimes);

    const visiblePriceSeries = priceSeries.filter((item) => (item.time as number) >= minVisibleTime);
    const visiblePredictionSeries = predictionSeries.filter((item) => (item.time as number) >= minVisibleTime);
    const visibleNewsEvents = newsEvents.filter((item) => (item.time as number) >= minVisibleTime);
    const visibleEventMarkers = eventMarkers.filter((item) => (item.time as number) >= minVisibleTime);

    const visibleTimes = [
      ...visiblePriceSeries.map((item) => item.time as number),
      ...visiblePredictionSeries.map((item) => item.time as number),
      ...visibleNewsEvents.map((item) => item.time as number),
      ...visibleEventMarkers.map((item) => item.time as number)
    ];

    if (!visibleTimes.length) {
      return null;
    }

    const minTime = Math.min(...visibleTimes);
    const maxVisible = Math.max(...visibleTimes);
    const plotWidth = WIDTH - PAD_LEFT - PAD_RIGHT;
    const plotHeight = HEIGHT - PAD_TOP - PAD_BOTTOM - VOLUME_BAND;

    const xForTime = (value: number) => {
      if (maxVisible === minTime) {
        return PAD_LEFT + plotWidth / 2;
      }
      return PAD_LEFT + ((value - minTime) / (maxVisible - minTime)) * plotWidth;
    };

    const lowCandidates = visiblePriceSeries.map((item) => item.lowValue ?? item.closeValue ?? 0);
    const highCandidates = visiblePriceSeries.map((item) => item.highValue ?? item.closeValue ?? 0);
    const priceMin = Math.min(...lowCandidates);
    const priceMax = Math.max(...highCandidates);
    const yForPrice = (value: number) => {
      if (priceMax === priceMin) {
        return PAD_TOP + plotHeight / 2;
      }
      return PAD_TOP + plotHeight - ((value - priceMin) / (priceMax - priceMin)) * plotHeight;
    };

    const returnValues: number[] = [];
    if (overlayState.predicted) {
      returnValues.push(...visiblePredictionSeries.map((item) => item.predictedValue).filter((value): value is number => value !== null));
    }
    if (overlayState.actual) {
      returnValues.push(...visiblePredictionSeries.map((item) => item.forwardValue).filter((value): value is number => value !== null));
    }
    layerScoreColumns.forEach((column) => {
      const key = String(column.key ?? "");
      if (!key || !overlayState[key]) {
        return;
      }
      visiblePredictionSeries.forEach((item) => {
        const value = toNumber(item[key]);
        if (value !== null) {
          returnValues.push(value);
        }
      });
    });
    const returnMin = returnValues.length ? Math.min(...returnValues, 0) : -0.05;
    const returnMax = returnValues.length ? Math.max(...returnValues, 0) : 0.05;
    const yForReturn = (value: number) => {
      if (returnMax === returnMin) {
        return PAD_TOP + plotHeight / 2;
      }
      return PAD_TOP + plotHeight - ((value - returnMin) / (returnMax - returnMin)) * plotHeight;
    };

    const pricePath = linePath(
      visiblePriceSeries.map((item) => ({
        x: xForTime(item.time as number),
        y: yForPrice(item.closeValue as number)
      }))
    );

    const predictedPath = linePath(
      visiblePredictionSeries
        .filter((item) => item.predictedValue !== null)
        .map((item) => ({
          x: xForTime(item.time as number),
          y: yForReturn(item.predictedValue as number)
        }))
    );

    const actualPath = linePath(
      visiblePredictionSeries
        .filter((item) => item.forwardValue !== null)
        .map((item) => ({
          x: xForTime(item.time as number),
          y: yForReturn(item.forwardValue as number)
        }))
    );

    const layerPaths = layerScoreColumns
      .map((column, index) => {
        const key = String(column.key ?? "");
        return {
          key,
          label: String(column.label ?? key),
          color: LAYER_COLORS[index % LAYER_COLORS.length],
          path: linePath(
            visiblePredictionSeries
              .map((item) => {
                const value = toNumber(item[key]);
                if (value === null) {
                  return null;
                }
                return {
                  x: xForTime(item.time as number),
                  y: yForReturn(value)
                };
              })
              .filter((item): item is { x: number; y: number } => item !== null)
          )
        };
      })
      .filter((item) => item.path);

    return {
      layerPaths,
      layerScoreColumns,
      newsEvents: visibleNewsEvents,
      eventMarkers: visibleEventMarkers,
      plotHeight,
      plotWidth,
      priceMax,
      priceMin,
      pricePath,
      returnMax,
      returnMin,
      predictedPath,
      actualPath,
      visiblePredictionSeries,
      visiblePriceSeries,
      volumeMax: Math.max(...visiblePriceSeries.map((item) => item.volumeValue ?? 0), 1),
      xForTime,
      yForPrice,
      yForReturn
    };
  }, [overlayState, payload, windowMode]);

  if (!payload) {
    return <div className="chart-empty">Select a dataset and chart inputs to render the timeline.</div>;
  }

  if (!derived || !derived.visiblePriceSeries.length) {
    return <div className="chart-empty">No chartable rows were available for the selected window.</div>;
  }

  const zeroLine = derived.yForReturn(0);

  return (
    <div className="dataset-chart-shell">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="dataset-chart" role="img" aria-label="Dataset visualization chart">
        <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="rgba(8, 11, 16, 0.55)" />
        <rect x={PAD_LEFT} y={PAD_TOP} width={derived.plotWidth} height={derived.plotHeight} className="chart-plot-frame" />
        <line x1={PAD_LEFT} y1={zeroLine} x2={WIDTH - PAD_RIGHT} y2={zeroLine} className="chart-zero-line" />

        {overlayState.volume
          ? derived.visiblePriceSeries.map((item) => {
              const x = derived.xForTime(item.time as number);
              const barHeight = ((item.volumeValue ?? 0) / derived.volumeMax) * (VOLUME_BAND - 14);
              return (
                <rect
                  key={`volume-${item.effective_at}`}
                  x={x - 1.5}
                  y={HEIGHT - PAD_BOTTOM - barHeight}
                  width={3}
                  height={barHeight}
                  className="chart-volume-bar"
                >
                  <title>{`${item.effective_at}: volume ${formatNumber(item.volumeValue, 0)}`}</title>
                </rect>
              );
            })
          : null}

        {overlayState.price && derived.pricePath ? <path d={derived.pricePath} className="chart-price-line" /> : null}
        {overlayState.predicted && derived.predictedPath ? <path d={derived.predictedPath} className="chart-predicted-line" /> : null}
        {overlayState.actual && derived.actualPath ? <path d={derived.actualPath} className="chart-actual-line" /> : null}
        {derived.layerPaths
          .filter((item) => overlayState[item.key])
          .map((item) => <path key={item.key} d={item.path} stroke={item.color} className="chart-layer-line" />)}

        {overlayState.news
          ? derived.newsEvents
              .filter((item) => item.event_scope === "ticker")
              .map((item) => (
                <g key={`ticker-news-${item.event_id}`}>
                  <line
                    x1={derived.xForTime(item.time as number)}
                    y1={PAD_TOP}
                    x2={derived.xForTime(item.time as number)}
                    y2={NEWS_LANES.ticker}
                    className="chart-news-line"
                  />
                  <circle cx={derived.xForTime(item.time as number)} cy={NEWS_LANES.ticker} r={4} className="chart-news-dot ticker-news">
                    <title>{`${item.headline}\n${item.known_at}`}</title>
                  </circle>
                </g>
              ))
          : null}

        {overlayState.macro_news
          ? derived.newsEvents
              .filter((item) => item.event_scope === "macro")
              .map((item) => (
                <g key={`macro-news-${item.event_id}`}>
                  <line
                    x1={derived.xForTime(item.time as number)}
                    y1={PAD_TOP}
                    x2={derived.xForTime(item.time as number)}
                    y2={NEWS_LANES.macro}
                    className="chart-macro-line"
                  />
                  <circle cx={derived.xForTime(item.time as number)} cy={NEWS_LANES.macro} r={4} className="chart-news-dot macro-news">
                    <title>{`${item.headline}\n${item.known_at}`}</title>
                  </circle>
                </g>
              ))
          : null}

        {derived.eventMarkers
          .filter((item) => overlayState[item.category as string])
          .map((item) => (
            <g key={`${item.category}-${item.effective_at}-${item.label}`}>
              <line
                x1={derived.xForTime(item.time as number)}
                y1={PAD_TOP}
                x2={derived.xForTime(item.time as number)}
                y2={NEWS_LANES.event}
                className="chart-event-line"
              />
              <rect x={derived.xForTime(item.time as number) - 3} y={NEWS_LANES.event - 3} width={6} height={6} className={`chart-event-box ${String(item.category)}`}>
                <title>{`${item.label}\n${item.effective_at}`}</title>
              </rect>
            </g>
          ))}

        <text x={10} y={PAD_TOP + 8} className="chart-axis-label">{formatNumber(derived.priceMax)}</text>
        <text x={10} y={PAD_TOP + derived.plotHeight} className="chart-axis-label">{formatNumber(derived.priceMin)}</text>
        <text x={WIDTH - PAD_RIGHT + 8} y={PAD_TOP + 8} className="chart-axis-label right">{formatPercent(derived.returnMax)}</text>
        <text x={WIDTH - PAD_RIGHT + 8} y={PAD_TOP + derived.plotHeight} className="chart-axis-label right">{formatPercent(derived.returnMin)}</text>
      </svg>

      <div className="chart-legend">
        <span className="chart-legend-item"><i className="price" />Price</span>
        <span className="chart-legend-item"><i className="predicted" />Predicted Return</span>
        <span className="chart-legend-item"><i className="actual" />Realized Forward Return</span>
        <span className="chart-legend-item"><i className="ticker-news" />Ticker News</span>
        <span className="chart-legend-item"><i className="macro-news" />Macro News</span>
        <span className="chart-legend-item"><i className="earnings" />Earnings</span>
        <span className="chart-legend-item"><i className="macro-surprise" />Macro Surprise</span>
      </div>
    </div>
  );
}
