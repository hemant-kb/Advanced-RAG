import { useState, useEffect, useRef } from "react";
import { Badge, DetailLine, ThinkingStep } from "../hooks/useStream";

// ── Badge rendering ─────────────────────────────────────────────
// Badges arrive as structured objects from the backend (see useStream.Badge).

const BADGE_STYLE: Record<Badge["badge"], { colorClass: string; icon: string }> = {
  llm:    { colorClass: "badge-llm",    icon: "🔵" },
  qdrant: { colorClass: "badge-qdrant", icon: "🟢" },
  cohere: { colorClass: "badge-cohere", icon: "🟣" },
};

function fmtTime(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms}ms`;
}

function fmtTok(n: number): string {
  return `${n.toLocaleString()} tok`;
}

function badgeFields(b: Badge): { key: string; value: string }[] {
  const num = (k: string) => Number(b[k] ?? 0);
  if (b.badge === "llm") {
    const fields = [
      { key: "in",  value: fmtTok(num("in")) },
      { key: "out", value: fmtTok(num("out")) },
    ];
    if (num("cached") > 0) fields.push({ key: "cached", value: fmtTok(num("cached")) });
    fields.push({ key: "cost", value: `~$${num("cost").toFixed(5)}` });
    fields.push({ key: "time", value: fmtTime(num("ms")) });
    return fields;
  }
  if (b.badge === "qdrant") {
    return [
      { key: "candidates", value: String(num("candidates")) },
      { key: "embed",      value: fmtTime(num("embed_ms")) },
      { key: "query",      value: fmtTime(num("qdrant_ms")) },
    ];
  }
  // cohere
  return [
    { key: "pairs", value: String(num("pairs")) },
    { key: "top",   value: String(num("top")) },
    { key: "time",  value: fmtTime(num("ms")) },
  ];
}

function BadgeChip({ badge }: { badge: Badge }) {
  const style = BADGE_STYLE[badge.badge];
  if (!style) return null;
  const model = badge.badge === "qdrant" ? String(badge.mode ?? "") : String(badge.model ?? "");
  const shortModel = model.split("/").pop() || model;
  return (
    <div className={`badge-chip ${style.colorClass}`}>
      <span className="badge-icon">{style.icon}</span>
      <span className="badge-kind">{badge.badge.toUpperCase()}</span>
      {shortModel && <span className="badge-model">{shortModel}</span>}
      {badgeFields(badge).map((f) => (
        <span key={f.key} className="badge-field">
          <span className="badge-field-key">{f.key}</span>
          <span className="badge-field-val">{f.value}</span>
        </span>
      ))}
    </div>
  );
}

function DetailRow({ line }: { line: DetailLine }) {
  if (typeof line !== "string") {
    return <BadgeChip badge={line} />;
  }
  const isHeader = line.startsWith("──");
  return (
    <div className={`thinking-step-detail-line${isHeader ? " detail-header" : ""}`}>
      {line}
    </div>
  );
}

interface Props {
  steps: ThinkingStep[];
  isThinking: boolean;   // true while streaming
  durationMs: number;    // 0 while streaming, set on done
}

export function ThinkingBlock({ steps, isThinking, durationMs }: Props) {
  const [open, setOpen] = useState(true);  // auto-open while thinking
  // Auto-expand steps that have details so user sees them without clicking.
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());

  // When details arrive on a step, auto-expand it.
  useEffect(() => {
    steps.forEach((step, i) => {
      if (step.details.length > 0) {
        setExpandedSteps((prev) => {
          if (prev.has(i)) return prev;
          const next = new Set(prev);
          next.add(i);
          return next;
        });
      }
    });
  }, [steps]);
  const stepsRef = useRef<HTMLDivElement>(null);

  // Auto-scroll steps list as new steps arrive
  useEffect(() => {
    if (isThinking && stepsRef.current) {
      stepsRef.current.scrollTop = stepsRef.current.scrollHeight;
    }
  }, [steps, isThinking]);

  // Auto-collapse when thinking finishes
  useEffect(() => {
    if (!isThinking && durationMs > 0) {
      setOpen(false);
    }
  }, [isThinking, durationMs]);

  if (steps.length === 0 && !isThinking) return null;

  const durationSec = durationMs > 0 ? (durationMs / 1000).toFixed(1) : null;

  function toggleStep(idx: number) {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  }

  return (
    <div className={`thinking-block${isThinking ? " thinking-active" : ""}`}>
      <button
        className="thinking-header"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className={`thinking-dot${isThinking ? " pulsing" : ""}`} />
        <span className="thinking-label">
          {isThinking ? "Thinking…" : `Thought for ${durationSec}s`}
        </span>
        <span className={`thinking-chevron${open ? " open" : ""}`}>›</span>
      </button>

      {open && (
        <div className="thinking-steps" ref={stepsRef}>
          {steps.map((step, i) => {
            const isDone = step.status === "done";
            const hasDetails = step.details.length > 0;
            const isExpanded = expandedSteps.has(i);

            return (
              <div key={i} className="thinking-step">
                <div className="thinking-step-row">
                  <span className={`thinking-step-icon${isDone ? " done" : " active"}`}>
                    {isDone ? "✓" : "⟳"}
                  </span>
                  <span className="thinking-step-text">{step.label}</span>
                  {hasDetails && (
                    <button
                      className="thinking-step-toggle"
                      onClick={() => toggleStep(i)}
                      title={isExpanded ? "Collapse details" : "Expand details"}
                    >
                      {isExpanded ? "▲" : "▼"}
                    </button>
                  )}
                </div>
                {hasDetails && isExpanded && (
                  <div className="thinking-step-details">
                    {step.details.map((line, j) => (
                      <DetailRow key={j} line={line} />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
