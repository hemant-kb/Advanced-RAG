import { memo, useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { vs } from "react-syntax-highlighter/dist/esm/styles/prism";
import { ThinkingBlock } from "./ThinkingBlock";
import { PipelineSummary } from "./PipelineSummary";
import { ThinkingStep, PipelineSummary as PS } from "../hooks/useStream";

// Reactively detect current theme from <html> class
function useIsDark() {
  const [isDark, setIsDark] = useState(!document.documentElement.classList.contains("light"));
  useEffect(() => {
    const obs = new MutationObserver(() => {
      setIsDark(!document.documentElement.classList.contains("light"));
    });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return isDark;
}

interface CodeBlockProps {
  language: string;
  code: string;
  isDark: boolean;
}

function CodeBlock({ language, code, isDark }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  function copy() {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  const style = isDark ? vscDarkPlus : vs;

  return (
    <div className="code-block">
      <div className="code-block-header">
        <span className="code-lang">{language || "plaintext"}</span>
        <button className="code-copy-btn" onClick={copy}>
          {copied ? (
            <>
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" width="12" height="12">
                <polyline points="2 8 6 12 14 4" />
              </svg>
              Copied
            </>
          ) : (
            <>
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" width="12" height="12">
                <rect x="4" y="4" width="9" height="9" rx="1.5" />
                <path d="M3 10H2a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v1" />
              </svg>
              Copy
            </>
          )}
        </button>
      </div>
      <SyntaxHighlighter
        language={language || "text"}
        style={style}
        customStyle={{
          margin: 0,
          borderRadius: 0,
          fontSize: "13px",
          lineHeight: "1.6",
          padding: "14px 18px",
          background: isDark ? "#1e1e1e" : "#f6f8fa",
          border: "none",
        }}
        codeTagProps={{ style: { fontFamily: '"Söhne Mono", "Consolas", "Fira Code", "SF Mono", monospace' } }}
        showLineNumbers={code.split("\n").length > 10}
        lineNumberStyle={{ color: isDark ? "#555" : "#bbb", minWidth: "2.5em", userSelect: "none", paddingRight: "12px" }}
        wrapLongLines={false}
      >
        {code.trimEnd()}
      </SyntaxHighlighter>
    </div>
  );
}

interface Props {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  thinkingSteps?: ThinkingStep[];
  thinkingDoneMs?: number;
  isThinking?: boolean;
  pipelineSummary?: PS | null;
}

function normalizeMarkdown(text: string): string {
  // Ensure a blank line before list items so GFM parses them as <ul>/<ol>
  return text.replace(/([^\n])\n([ \t]*[-*+][ \t])/g, "$1\n\n$2")
             .replace(/([^\n])\n([ \t]*\d+\.[ \t])/g, "$1\n\n$2");
}

// Memoized: during token streaming only the live bubble's props change, so
// historical messages skip re-parsing markdown / re-highlighting code per token.
export const MessageBubble = memo(function MessageBubble({ role, content, streaming, thinkingSteps, thinkingDoneMs, isThinking, pipelineSummary }: Props) {
  const steps = thinkingSteps ?? [];
  const doneMs = thinkingDoneMs ?? 0;
  const thinking = isThinking ?? false;
  const isDark = useIsDark();

  const showThinking = steps.length > 0 || thinking;

  const markdownContent = (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ node, className, children, ...props }: any) {
          const match = /language-(\w+)/.exec(className || "");
          const codeStr = String(children).replace(/\n$/, "");
          const isBlock = codeStr.includes("\n") || !!match;
          if (isBlock) {
            return (
              <CodeBlock
                language={match ? match[1] : ""}
                code={codeStr}
                isDark={isDark}
              />
            );
          }
          return <code className="inline-code" {...props}>{children}</code>;
        },
      }}
    >
      {role === "assistant" ? normalizeMarkdown(content) : content}
    </ReactMarkdown>
  );

  return (
    <div className={`message ${role}`}>
      <div className="message-inner">
        <div className="bubble">
          {role === "assistant" && showThinking && (
            <ThinkingBlock
              steps={steps}
              isThinking={thinking}
              durationMs={doneMs}
            />
          )}

          {role === "assistant" ? (
            <>
              {!thinking && pipelineSummary && (
                <PipelineSummary summary={pipelineSummary} />
              )}
              {content && (
                <div className="answer-body">
                  {markdownContent}
                  {streaming && <span className="cursor" />}
                </div>
              )}
              {!content && streaming && (
                <div className="answer-body"><span className="cursor" /></div>
              )}
            </>
          ) : (
            /* User: plain content */
            content && (
              <>
                {markdownContent}
                {streaming && <span className="cursor" />}
              </>
            )
          )}
        </div>
      </div>
    </div>
  );
});
