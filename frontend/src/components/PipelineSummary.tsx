import { useState } from "react";
import { PipelineSummary as PS } from "../hooks/useStream";

interface Props {
  summary: PS;
}

function fmt(n: number): string {
  return n.toLocaleString();
}

function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms}ms`;
}

function fmtCost(usd: number): string {
  if (usd === 0) return "$0.00000";
  if (usd < 0.00001) return `$${usd.toFixed(7)}`;
  return `$${usd.toFixed(5)}`;
}

export function PipelineSummary({ summary }: Props) {
  const [open, setOpen] = useState(false);

  const tiles = [
    { label: "Total time",         value: fmtMs(summary.total_ms),                  color: "tile-time" },
    { label: "LLM calls",          value: String(summary.llm_calls),                color: "tile-llm" },
    { label: "Vector searches",    value: String(summary.vector_searches),           color: "tile-qdrant" },
    { label: "Chunks retrieved",   value: fmt(summary.chunks_retrieved),             color: "tile-qdrant" },
    { label: "Chunks reranked",    value: fmt(summary.chunks_reranked),              color: "tile-cohere" },
    { label: "Chunks used",        value: fmt(summary.chunks_used),                  color: "tile-cohere" },
    { label: "Prompt tokens",      value: fmt(summary.prompt_tokens),                color: "tile-llm" },
    { label: "Completion tokens",  value: fmt(summary.completion_tokens),            color: "tile-llm" },
    { label: "Cached tokens",      value: fmt(summary.cached_tokens),                color: "tile-cache" },
    { label: "Estimated cost",     value: fmtCost(summary.cost_usd),                 color: "tile-cost" },
  ];

  return (
    <div className="pipeline-summary">
      <button
        className="pipeline-summary-header"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="pipeline-summary-icon">⚡</span>
        <span className="pipeline-summary-label">Pipeline Summary</span>
        <span className="pipeline-summary-meta">
          {fmtMs(summary.total_ms)} · {summary.llm_calls} LLM call{summary.llm_calls !== 1 ? "s" : ""} · {fmtCost(summary.cost_usd)}
        </span>
        <span className={`pipeline-summary-chevron${open ? " open" : ""}`}>›</span>
      </button>

      {open && (
        <div className="pipeline-summary-tiles">
          {tiles.map((t) => (
            <div key={t.label} className={`pipeline-tile ${t.color}`}>
              <span className="pipeline-tile-value">{t.value}</span>
              <span className="pipeline-tile-label">{t.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
