import { useCallback, useRef, useState } from "react";
import { chatUrl } from "../lib/api";

// Structured badge dict emitted by the backend (rag_graph → chat route).
// kind "llm" | "qdrant" | "cohere"; remaining keys are metric fields.
export interface Badge {
  badge: "llm" | "qdrant" | "cohere";
  model?: string;
  [key: string]: string | number | undefined;
}

export type DetailLine = string | Badge;

export interface ThinkingStep {
  label: string;
  node: string;          // node name from NODE_LABELS, "detail", or "start"
  parentNode?: string;   // set when node === "detail"
  output?: string;       // "done" for completed node steps
  status: "active" | "done";
  details: DetailLine[]; // detail lines/badges grouped under this step
}

export interface PipelineSummary {
  total_ms: number;
  llm_calls: number;
  vector_searches: number;
  chunks_retrieved: number;
  chunks_reranked: number;
  chunks_used: number;
  prompt_tokens: number;
  completion_tokens: number;
  cached_tokens: number;
  cost_usd: number;
  qdrant_ms: number;
  rerank_ms: number;
  embed_ms: number;
}

// One assistant turn's metadata, cached across session switches.
export interface TurnMeta {
  steps: ThinkingStep[];
  doneMs: number;
  summary: PipelineSummary | null;
}

export interface StreamState {
  isStreaming: boolean;
  partial: string;
  progress: string;
  thinkingSteps: ThinkingStep[];
  thinkingDoneMs: number;
  contextWarning: string;
  pipelineSummary: PipelineSummary | null;
}

// Append a detail line to its parent step: the step matching parentNode if
// given, otherwise the most recent non-detail step. Returns a new array.
function attachDetail(steps: ThinkingStep[], detail: DetailLine, parentNode?: string): ThinkingStep[] {
  let idx = parentNode ? steps.findIndex((s) => s.node === parentNode) : -1;
  if (idx < 0) {
    for (let i = steps.length - 1; i >= 0; i--) {
      if (steps[i].node !== "detail") { idx = i; break; }
    }
  }
  if (idx < 0) return steps;
  return steps.map((s, i) => (i === idx ? { ...s, details: [...s.details, detail] } : s));
}

type Handlers = {
  onToken?: (t: string) => void;
  onProgress?: (m: string) => void;
  onDone?: (answer: string, thinkingSteps: ThinkingStep[], thinkingDoneMs: number, pipelineSummary: PipelineSummary | null) => void;
  onError?: (m: string) => void;
  onContextLimit?: (message: string, warnOnly: boolean) => void;
};

export function useStream() {
  const [state, setState] = useState<StreamState>({
    isStreaming: false,
    partial: "",
    progress: "",
    thinkingSteps: [],
    thinkingDoneMs: 0,
    contextWarning: "",
    pipelineSummary: null,
  });
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(async (sessionId: string, message: string, handlers: Handlers = {}) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setState({ isStreaming: true, partial: "", progress: "", thinkingSteps: [], thinkingDoneMs: 0, contextWarning: "", pipelineSummary: null });

    try {
      const resp = await fetch(chatUrl(sessionId), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
        signal: ctrl.signal,
      });
      if (!resp.ok) {
        let msg = `HTTP ${resp.status}`;
        try { const j = await resp.json(); msg = j.detail ?? msg; } catch { /* keep default */ }
        throw new Error(msg);
      }
      if (!resp.body) throw new Error("No response body");

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let acc = "";
      let thinkingStepsAcc: ThinkingStep[] = [];
      let thinkingDoneMsAcc = 0;
      let pipelineSummaryAcc: PipelineSummary | null = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const block of lines) {
          const line = block.trim();
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            const evt = JSON.parse(payload);
            if (evt.type === "context_limit") {
              const warnOnly = !!evt.warn_only;
              setState((s) => ({ ...s, contextWarning: evt.message, isStreaming: warnOnly ? s.isStreaming : false }));
              handlers.onContextLimit?.(evt.message, warnOnly);
            } else if (evt.type === "thinking_step") {
              if (evt.node === "detail") {
                thinkingStepsAcc = attachDetail(thinkingStepsAcc, evt.badge ?? evt.step, evt.parent_node);
              } else {
                // Top-level node step: mark all previous steps as done, add new active step
                thinkingStepsAcc = [
                  ...thinkingStepsAcc.map((s) => ({ ...s, status: "done" as const })),
                  {
                    label: evt.step,
                    node: evt.node ?? "unknown",
                    parentNode: undefined,
                    output: undefined,
                    status: "active" as const,
                    details: [],
                  },
                ];
              }
              setState((s) => ({ ...s, thinkingSteps: thinkingStepsAcc }));
            } else if (evt.type === "thinking_step_output") {
              // Mark the matching node step as done
              thinkingStepsAcc = thinkingStepsAcc.map((step) => {
                if (step.node === evt.node) {
                  return { ...step, status: "done" as const, output: evt.output ?? "done" };
                }
                return step;
              });
              setState((s) => ({ ...s, thinkingSteps: thinkingStepsAcc }));
            } else if (evt.type === "thinking_done") {
              thinkingDoneMsAcc = evt.duration_ms || 0;
              // Mark all steps as done
              thinkingStepsAcc = thinkingStepsAcc.map((s) => ({ ...s, status: "done" as const }));
              setState((s) => ({ ...s, thinkingDoneMs: thinkingDoneMsAcc, thinkingSteps: thinkingStepsAcc }));
            } else if (evt.type === "token") {
              acc += evt.content;
              setState((s) => ({ ...s, partial: acc }));
              handlers.onToken?.(evt.content);
            } else if (evt.type === "progress") {
              setState((s) => ({ ...s, progress: evt.message }));
              handlers.onProgress?.(evt.message);
            } else if (evt.type === "pipeline_summary") {
              pipelineSummaryAcc = evt as PipelineSummary;
              setState((s) => ({ ...s, pipelineSummary: pipelineSummaryAcc }));
            } else if (evt.type === "done") {
              const answer = evt.answer || acc;
              setState((s) => ({ ...s, isStreaming: false, partial: "", progress: "", thinkingSteps: thinkingStepsAcc, thinkingDoneMs: thinkingDoneMsAcc, pipelineSummary: pipelineSummaryAcc }));
              handlers.onDone?.(answer, thinkingStepsAcc, thinkingDoneMsAcc, pipelineSummaryAcc);
            } else if (evt.type === "error") {
              setState((s) => ({ ...s, isStreaming: false, partial: "", progress: "" }));
              handlers.onError?.(evt.message);
            }
          } catch {
            /* ignore malformed line */
          }
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") {
        setState((s) => ({ ...s, isStreaming: false, partial: "", progress: "" }));
        handlers.onError?.(e.message);
      } else {
        setState((s) => ({ ...s, isStreaming: false, partial: "", progress: "" }));
      }
    }
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { ...state, send, abort };
}
