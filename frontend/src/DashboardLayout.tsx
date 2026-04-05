import { Children, useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent, ReactNode } from "react";

interface DashboardLayoutPanel {
  id: string;
  label: string;
}

interface DashboardLayoutEntry extends DashboardLayoutPanel {
  content: ReactNode;
}

interface DashboardLayoutProps {
  children: ReactNode;
  defaultRows?: string[][];
  layoutKey: string;
  panels: DashboardLayoutPanel[];
}

interface DashboardLayoutRow {
  panelIds: string[];
  widths: number[];
}

const STORAGE_PREFIX = "quant-platform-dashboard-layout:";
const MAX_PANELS_PER_ROW = 2;
const MIN_PANEL_SHARE = 0.28;

function buildDefaultRows(panelIds: string[], defaultRows?: string[][]): DashboardLayoutRow[] {
  const rows = defaultRows?.length
    ? defaultRows
    : panelIds.reduce<string[][]>((chunks, panelId, index) => {
        if (index % MAX_PANELS_PER_ROW === 0) {
          chunks.push([panelId]);
        } else {
          chunks[chunks.length - 1].push(panelId);
        }
        return chunks;
      }, []);

  return rows.map((panelRow) => ({
    panelIds: [...panelRow],
    widths: panelRow.length === 2 ? [0.5, 0.5] : [1]
  }));
}

function normalizeWidths(widths: number[] | undefined, panelCount: number): number[] {
  if (panelCount <= 1) {
    return [1];
  }

  if (!widths || widths.length !== 2) {
    return [0.5, 0.5];
  }

  const total = widths[0] + widths[1];
  const firstWidth = total > 0 ? widths[0] / total : 0.5;
  const clampedFirst = Math.min(1 - MIN_PANEL_SHARE, Math.max(MIN_PANEL_SHARE, firstWidth));
  return [clampedFirst, 1 - clampedFirst];
}

function sanitizeRows(rawRows: unknown, panelIds: string[], defaultRows?: string[][]): DashboardLayoutRow[] {
  const panelIdSet = new Set(panelIds);
  const seen = new Set<string>();
  const baseRows = Array.isArray(rawRows) ? rawRows : buildDefaultRows(panelIds, defaultRows);
  const cleanedRows: DashboardLayoutRow[] = [];

  for (const rawRow of baseRows) {
    if (!rawRow || typeof rawRow !== "object") {
      continue;
    }

    const candidateIds = Array.isArray((rawRow as DashboardLayoutRow).panelIds)
      ? (rawRow as DashboardLayoutRow).panelIds
      : Array.isArray(rawRow)
        ? rawRow
        : [];

    const panelRow = candidateIds
      .map((panelId) => String(panelId))
      .filter((panelId) => panelIdSet.has(panelId) && !seen.has(panelId))
      .slice(0, MAX_PANELS_PER_ROW);

    if (!panelRow.length) {
      continue;
    }

    panelRow.forEach((panelId) => seen.add(panelId));
    cleanedRows.push({
      panelIds: panelRow,
      widths: normalizeWidths((rawRow as DashboardLayoutRow).widths, panelRow.length)
    });
  }

  const missingPanels = panelIds.filter((panelId) => !seen.has(panelId));
  if (missingPanels.length) {
    buildDefaultRows(missingPanels).forEach((row) => cleanedRows.push(row));
  }

  if (!cleanedRows.length && panelIds.length) {
    return buildDefaultRows(panelIds, defaultRows);
  }

  return cleanedRows.map((row) => ({
    panelIds: [...row.panelIds],
    widths: normalizeWidths(row.widths, row.panelIds.length)
  }));
}

function movePanelBetweenRows(rows: DashboardLayoutRow[], panelId: string, direction: "up" | "down"): DashboardLayoutRow[] {
  const nextRows = rows.map((row) => ({
    panelIds: [...row.panelIds],
    widths: [...row.widths]
  }));
  const sourceRowIndex = nextRows.findIndex((row) => row.panelIds.includes(panelId));

  if (sourceRowIndex < 0) {
    return rows;
  }

  const targetRowIndex = direction === "up" ? sourceRowIndex - 1 : sourceRowIndex + 1;
  if (targetRowIndex < 0 || targetRowIndex >= nextRows.length) {
    return rows;
  }

  const sourceRow = nextRows[sourceRowIndex];
  const sourcePanelIndex = sourceRow.panelIds.indexOf(panelId);
  const targetRow = nextRows[targetRowIndex];

  if (sourcePanelIndex < 0) {
    return rows;
  }

  if (targetRow.panelIds.length < MAX_PANELS_PER_ROW) {
    sourceRow.panelIds.splice(sourcePanelIndex, 1);
    const targetInsertIndex = Math.min(sourcePanelIndex, targetRow.panelIds.length);
    targetRow.panelIds.splice(targetInsertIndex, 0, panelId);
    targetRow.widths = normalizeWidths(targetRow.widths, targetRow.panelIds.length);

    if (!sourceRow.panelIds.length) {
      nextRows.splice(sourceRowIndex, 1);
    } else {
      sourceRow.widths = normalizeWidths(sourceRow.widths, sourceRow.panelIds.length);
    }

    return nextRows.map((row) => ({
      panelIds: [...row.panelIds],
      widths: normalizeWidths(row.widths, row.panelIds.length)
    }));
  }

  const targetPanelIndex = Math.min(sourcePanelIndex, targetRow.panelIds.length - 1);
  [sourceRow.panelIds[sourcePanelIndex], targetRow.panelIds[targetPanelIndex]] = [
    targetRow.panelIds[targetPanelIndex],
    sourceRow.panelIds[sourcePanelIndex]
  ];

  return nextRows.map((row) => ({
    panelIds: [...row.panelIds],
    widths: normalizeWidths(row.widths, row.panelIds.length)
  }));
}

export function DashboardLayout({ children, defaultRows, layoutKey, panels }: DashboardLayoutProps) {
  const childPanels = useMemo(() => Children.toArray(children), [children]);
  const panelEntries = useMemo<DashboardLayoutEntry[]>(
    () =>
      panels.map((panel, index) => ({
        ...panel,
        content: childPanels[index] ?? null
      })),
    [childPanels, panels]
  );
  const panelMap = useMemo(() => new Map(panelEntries.map((panel) => [panel.id, panel])), [panelEntries]);
  const panelIds = useMemo(() => panels.map((panel) => panel.id), [panels]);
  const panelSignature = panelIds.join("|");
  const defaultRowsSignature = JSON.stringify(defaultRows ?? []);
  const storageKey = `${STORAGE_PREFIX}${layoutKey}`;
  const dragState = useRef<{ rowIndex: number; startWidth: number; rowWidth: number; startX: number } | null>(null);
  const [rows, setRows] = useState<DashboardLayoutRow[]>(() => {
    if (typeof window === "undefined") {
      return buildDefaultRows(panelIds, defaultRows);
    }

    try {
      const stored = window.localStorage.getItem(storageKey);
      return sanitizeRows(stored ? JSON.parse(stored) : null, panelIds, defaultRows);
    } catch {
      return buildDefaultRows(panelIds, defaultRows);
    }
  });

  useEffect(() => {
    setRows((currentRows) => sanitizeRows(currentRows, panelIds, defaultRows));
  }, [defaultRowsSignature, panelSignature]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(storageKey, JSON.stringify(rows));
  }, [rows, storageKey]);

  useEffect(() => {
    function handlePointerMove(event: PointerEvent) {
      const activeDrag = dragState.current;
      if (!activeDrag) {
        return;
      }

      const delta = event.clientX - activeDrag.startX;
      const widthDelta = activeDrag.rowWidth > 0 ? delta / activeDrag.rowWidth : 0;
      const nextWidth = Math.min(1 - MIN_PANEL_SHARE, Math.max(MIN_PANEL_SHARE, activeDrag.startWidth + widthDelta));

      setRows((currentRows) =>
        currentRows.map((row, rowIndex) =>
          rowIndex === activeDrag.rowIndex
            ? {
                ...row,
                widths: [nextWidth, 1 - nextWidth]
              }
            : row
        )
      );
    }

    function stopDragging() {
      dragState.current = null;
      document.body.classList.remove("panel-layout-is-resizing");
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopDragging);
    window.addEventListener("pointercancel", stopDragging);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopDragging);
      window.removeEventListener("pointercancel", stopDragging);
      document.body.classList.remove("panel-layout-is-resizing");
    };
  }, []);

  function beginResize(event: ReactPointerEvent<HTMLButtonElement>, rowIndex: number, firstPanelWidth: number) {
    const rowElement = event.currentTarget.parentElement;
    if (!rowElement) {
      return;
    }

    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragState.current = {
      rowIndex,
      startWidth: firstPanelWidth,
      rowWidth: rowElement.getBoundingClientRect().width,
      startX: event.clientX
    };
    document.body.classList.add("panel-layout-is-resizing");
  }

  function movePanel(panelId: string, direction: "up" | "down") {
    setRows((currentRows) => movePanelBetweenRows(currentRows, panelId, direction));
  }

  return (
    <div className="panel-layout">
      {rows.map((row, rowIndex) => {
        const rowPanels = row.panelIds
          .map((panelId) => panelMap.get(panelId))
          .filter((panel): panel is DashboardLayoutEntry => Boolean(panel));
        const isSplitRow = rowPanels.length === 2;
        const rowStyle = isSplitRow
          ? { gridTemplateColumns: `minmax(0, ${row.widths[0]}fr) 14px minmax(0, ${row.widths[1]}fr)` }
          : undefined;

        return (
          <div key={`${layoutKey}-${rowIndex}-${row.panelIds.join("-")}`} className={`panel-layout-row ${isSplitRow ? "is-split" : "is-single"}`} style={rowStyle}>
            {rowPanels.flatMap((panel, panelIndex) => {
              const canMoveUp = rowIndex > 0;
              const canMoveDown = rowIndex < rows.length - 1;
              const panelNode = (
                <div key={panel.id} className="panel-layout-cell">
                  <div className="panel-layout-toolbar">
                    <span>{panel.label}</span>
                    <div className="panel-layout-actions">
                      <button
                        type="button"
                        className="panel-layout-button"
                        onClick={() => movePanel(panel.id, "up")}
                        disabled={!canMoveUp}
                        title={`Move ${panel.label} to the row above`}
                      >
                        Up
                      </button>
                      <button
                        type="button"
                        className="panel-layout-button"
                        onClick={() => movePanel(panel.id, "down")}
                        disabled={!canMoveDown}
                        title={`Move ${panel.label} to the row below`}
                      >
                        Down
                      </button>
                    </div>
                  </div>
                  {panel.content}
                </div>
              );

              if (!isSplitRow || panelIndex !== 0) {
                return [panelNode];
              }

              return [
                panelNode,
                <button
                  key={`${panel.id}-separator`}
                  type="button"
                  className="panel-layout-separator"
                  aria-label={`Resize ${rowPanels[0]?.label ?? "left panel"} and ${rowPanels[1]?.label ?? "right panel"}`}
                  aria-orientation="vertical"
                  onPointerDown={(event) => beginResize(event, rowIndex, row.widths[0])}
                  title="Drag to resize panels in this row"
                >
                  <span />
                </button>
              ];
            })}
          </div>
        );
      })}
    </div>
  );
}
