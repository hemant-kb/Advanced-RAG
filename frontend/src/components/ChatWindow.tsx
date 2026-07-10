import { ChangeEvent, KeyboardEvent, MutableRefObject, useEffect, useRef, useState } from "react";
import { Ghost } from "lucide-react";
import { ChatMessage, SessionInfo, autoNameSession, getHistory, uploadPdf } from "../lib/api";
import { useStream, ThinkingStep, PipelineSummary, TurnMeta } from "../hooks/useStream";
import { MessageBubble } from "./MessageBubble";
import { AttachIcon, FileTextIcon, InfoIcon, SendIcon, StopIcon, TableIcon } from "./icons";

interface StoredMessage extends ChatMessage {
  thinkingSteps?: ThinkingStep[];
  thinkingDoneMs?: number;
  pipelineSummary?: PipelineSummary | null;
}

interface Props {
  session: SessionInfo;
  onDocumentChange: () => void;
  onSessionRenamed: (id: string, name: string) => void;
  // Keyed `${sessionId}:${assistantMessageIndex}`; lives in App.tsx so thinking
  // steps / pipeline summaries survive ChatWindow remounts on session switch.
  turnMetaMapRef: MutableRefObject<Map<string, TurnMeta>>;
}

export function ChatWindow({ session, onDocumentChange, onSessionRenamed, turnMetaMapRef }: Props) {
  const [messages, setMessages] = useState<StoredMessage[]>([]);
  const [input, setInput] = useState("");
  const [isFirstMessage, setIsFirstMessage] = useState(true);
  const [uploadState, setUploadState] = useState<{
    phase: "idle" | "uploading" | "processing" | "complete" | "error";
    message: string;
  }>({ phase: "idle", message: "" });
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [contextLimitMsg, setContextLimitMsg] = useState<{ text: string; hard: boolean } | null>(null);
  const { isStreaming, partial, progress, thinkingSteps, thinkingDoneMs, contextWarning, send, abort } = useStream();
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setHistoryLoaded(false);
    setMessages([]);
    setContextLimitMsg(null);
    getHistory(session.id).then((h) => {
      // Re-hydrate thinking steps / pipeline summary from the persistent map
      // so they survive session switches (ref lives in App.tsx, never unmounts).
      const hydrated: StoredMessage[] = h.messages.map((m, i) => {
        const meta = turnMetaMapRef.current.get(`${session.id}:${i}`);
        return {
          ...m,
          thinkingSteps:   meta?.steps,
          thinkingDoneMs:  meta?.doneMs,
          pipelineSummary: meta?.summary ?? null,
        };
      });
      setMessages(hydrated);
      setIsFirstMessage(h.messages.length === 0);
      setHistoryLoaded(true);
    });
  }, [session.id]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, partial, progress]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, [input]);

  useEffect(() => {
    if (contextWarning) {
      const hard = !contextWarning.includes("getting long");
      setContextLimitMsg({ text: contextWarning, hard });
    }
  }, [contextWarning]);

  function handleKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  async function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";
    setUploadState({ phase: "uploading", message: `Uploading ${file.name}…` });
    try {
      const status = await uploadPdf(session.id, file, (message, progress) => {
        const phase = progress >= 1.0 ? "complete" : "processing";
        setUploadState({ phase, message });
      });
      if (status.status === "complete") {
        setUploadState({ phase: "complete", message: status.message });
        onDocumentChange();
      } else if (status.status === "error") {
        setUploadState({ phase: "error", message: status.message || "Upload failed" });
      } else {
        setUploadState({ phase: "processing", message: status.message || "Processing…" });
        onDocumentChange();
      }
    } catch (err: any) {
      setUploadState({ phase: "error", message: `Error: ${err.message}` });
    } finally {
      setTimeout(() => setUploadState({ phase: "idle", message: "" }), 5000);
    }
  }

  function submit() {
    const text = input.trim();
    if (!text || isStreaming) return;
    if (contextLimitMsg?.hard) return;

    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");

    const wasFirst = isFirstMessage;
    if (wasFirst) setIsFirstMessage(false);

    send(session.id, text, {
      onDone: (answer, steps, doneMs, pipelineSummary) => {
        setMessages((m) => {
          // Cache turn metadata keyed by sessionId:assistantMessageIndex
          // so it survives session switches (ref lives in App.tsx).
          const assistantIdx = m.length;
          turnMetaMapRef.current.set(`${session.id}:${assistantIdx}`, {
            steps,
            doneMs,
            summary: pipelineSummary ?? null,
          });
          return [
            ...m,
            { role: "assistant", content: answer, thinkingSteps: steps, thinkingDoneMs: doneMs, pipelineSummary },
          ];
        });

        if (wasFirst) {
          autoNameSession(session.id, text).then((name) => {
            if (name) onSessionRenamed(session.id, name);
          });
        }
      },
      onError: (msg) => {
        setMessages((m) => [...m, { role: "assistant", content: `⚠️ ${msg}` }]);
      },
      onContextLimit: (message, warnOnly) => {
        setContextLimitMsg({ text: message, hard: !warnOnly });
      },
    });
  }

  const canSend = !!input.trim() && !isStreaming && !contextLimitMsg?.hard;
  const showEmptyState = historyLoaded && messages.length === 0 && !isStreaming;

  return (
    <div className="chat-area">
      {/* Header */}
      <div className="chat-header">
        <span className="chat-title">{session.name}</span>
        {session.document_name && (
          <span className="doc-badge">📄 {session.document_name}</span>
        )}
      </div>

      {/* Context limit banner */}
      {contextLimitMsg && (
        <div className={`context-limit-banner${contextLimitMsg.hard ? " hard" : " warn"}`}>
          <span>{contextLimitMsg.hard ? "🚫" : "⚠️"}</span>
          <span>{contextLimitMsg.text}</span>
          {!contextLimitMsg.hard && (
            <button className="banner-dismiss" onClick={() => setContextLimitMsg(null)}>×</button>
          )}
        </div>
      )}

      {/* Messages */}
      <div className="messages" ref={scrollRef}>
        {showEmptyState && (
          <div className="empty-state">
            <div className="empty-logo">
              <Ghost size={28} strokeWidth={1.8} />
            </div>
            <h2>Upload a PDF and start asking questions</h2>
            <div className="empty-cards">
              <button className="empty-card" onClick={() => setInput("What are the key findings in this document?")}>
                <div className="empty-card-icon"><FileTextIcon /></div>
                <span className="empty-card-title">Key findings</span>
                <span className="empty-card-sub">Summarise the main points</span>
              </button>
              <button className="empty-card" onClick={() => setInput("What tables or data are in this document?")}>
                <div className="empty-card-icon"><TableIcon /></div>
                <span className="empty-card-title">Tables & data</span>
                <span className="empty-card-sub">Find numbers and figures</span>
              </button>
              <button className="empty-card" onClick={() => setInput("Explain the methodology used in this document")}>
                <div className="empty-card-icon"><InfoIcon /></div>
                <span className="empty-card-title">Methodology</span>
                <span className="empty-card-sub">Understand how it was done</span>
              </button>
            </div>
          </div>
        )}

        {messages.map((m, i) => {
          // Prefer data stored on the message object; fall back to the persistent map.
          const meta = turnMetaMapRef.current.get(`${session.id}:${i}`);
          return (
            <MessageBubble
              key={i}
              role={m.role as "user" | "assistant"}
              content={m.content}
              thinkingSteps={m.thinkingSteps ?? meta?.steps}
              thinkingDoneMs={m.thinkingDoneMs ?? meta?.doneMs}
              isThinking={false}
              pipelineSummary={m.pipelineSummary ?? meta?.summary}
            />
          );
        })}

        {isStreaming && (thinkingSteps.length > 0 || partial) && (
          <MessageBubble
            role="assistant"
            content={partial}
            streaming={!!partial}
            thinkingSteps={thinkingSteps}
            thinkingDoneMs={thinkingDoneMs}
            isThinking={true}
            pipelineSummary={null}
          />
        )}

        {isStreaming && !partial && thinkingSteps.length === 0 && (
          <div className="progress-row">
            <span className="spinner" />
            <span className="progress-text">{progress || "Thinking…"}</span>
          </div>
        )}
      </div>

      {/* Upload banner */}
      {uploadState.phase !== "idle" && (
        <div className={`upload-banner upload-banner-${uploadState.phase}`}>
          {uploadState.phase === "uploading" && <span className="upload-spinner" />}
          {uploadState.phase === "processing" && <span className="upload-spinner" />}
          {uploadState.phase === "complete" && <span>✓</span>}
          {uploadState.phase === "error" && <span>✗</span>}
          <span>{uploadState.message}</span>
        </div>
      )}

      {/* Input bar */}
      <div className="input-wrap">
        <div className="input-box">
          {/* Left: PDF attach */}
          <label
            className={`attach-btn${uploadState.phase === "uploading" || uploadState.phase === "processing" ? " uploading" : ""}`}
            title={uploadState.phase === "uploading" || uploadState.phase === "processing" ? "Uploading PDF…" : "Add PDF"}
          >
            <AttachIcon />
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              onChange={handleFileChange}
              disabled={uploadState.phase === "uploading" || uploadState.phase === "processing"}
            />
          </label>

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            className="input-textarea"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about your document…"
            disabled={isStreaming || !!contextLimitMsg?.hard}
            rows={1}
          />

          {/* Right: Send / Abort */}
          {isStreaming ? (
            <button className="send-btn abort" onClick={abort} title="Stop generating">
              <StopIcon />
            </button>
          ) : (
            <button
              className="send-btn"
              onClick={submit}
              disabled={!canSend}
              title="Send (Enter)"
            >
              <SendIcon />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
