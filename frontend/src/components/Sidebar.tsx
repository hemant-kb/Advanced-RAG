import { KeyboardEvent, useRef, useState } from "react";
import { Ghost } from "lucide-react";
import { SessionInfo, renameSession } from "../lib/api";
import { ConfirmDialog } from "./ConfirmDialog";
import { CloseIcon, HamburgerIcon, PencilIcon, PlusIcon, TrashIcon } from "./icons";

interface Props {
  sessions: SessionInfo[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onDeleteAll: () => void;
  onRename: (id: string, name: string) => void;
  isDark: boolean;
  onToggleTheme: () => void;
}

export function Sidebar({
  sessions,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onDeleteAll,
  onRename,
  isDark,
  onToggleTheme,
}: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<SessionInfo | null>(null);
  const [showDeleteAll, setShowDeleteAll] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function startRename(s: SessionInfo, e: React.MouseEvent) {
    e.stopPropagation();
    setEditingId(s.id);
    setEditValue(s.name);
    setTimeout(() => inputRef.current?.select(), 0);
  }

  async function commitRename(id: string) {
    const name = editValue.trim();
    if (name && name !== sessions.find((s) => s.id === id)?.name) {
      await renameSession(id, name);
      onRename(id, name);
    }
    setEditingId(null);
  }

  function handleRenameKey(e: KeyboardEvent<HTMLInputElement>, id: string) {
    if (e.key === "Enter") commitRename(id);
    if (e.key === "Escape") setEditingId(null);
  }

  return (
    <>
      <aside className={`sidebar${collapsed ? " sidebar-collapsed" : ""}`}>
        {/* Top bar */}
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <div className="brand-mark">
              <Ghost size={16} strokeWidth={2} />
            </div>
            {!collapsed && <span className="brand-name">Agentic RAG Assistant</span>}
          </div>
          <button
            className="hamburger-btn"
            onClick={() => setCollapsed((c) => !c)}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-label="Toggle sidebar"
          >
            <HamburgerIcon />
          </button>
        </div>

        {/* New chat */}
        <div className="sidebar-new-wrap">
          <button className="new-chat-btn" onClick={onNew} title="New chat">
            <PlusIcon className="new-chat-icon" />
            {!collapsed && <span>New chat</span>}
          </button>
        </div>

        {/* Sessions list */}
        {!collapsed && (
          <div className="sidebar-scroll">
            {sessions.length > 0 && (
              <div className="sidebar-section-header">
                <span className="sidebar-section-label">Recent</span>
                <button
                  className="clear-all-btn"
                  title="Clear all chats"
                  onClick={() => setShowDeleteAll(true)}
                >
                  <TrashIcon />
                  <span>Clear all</span>
                </button>
              </div>
            )}
            {sessions.map((s) => (
              <div
                key={s.id}
                className={`session-item${s.id === activeId ? " active" : ""}`}
                onClick={() => { if (editingId !== s.id) onSelect(s.id); }}
              >
                {editingId === s.id ? (
                  <input
                    ref={inputRef}
                    className="session-rename-input"
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => handleRenameKey(e, s.id)}
                    onBlur={() => commitRename(s.id)}
                    onClick={(e) => e.stopPropagation()}
                    autoFocus
                  />
                ) : (
                  <span className="title">{s.name}</span>
                )}
                {s.has_document && editingId !== s.id && (
                  <span className="doc-pill">PDF</span>
                )}
                {editingId !== s.id && (
                  <div className="session-item-actions">
                    <button
                      className="session-action-btn"
                      title="Rename"
                      onClick={(e) => startRename(s, e)}
                    >
                      <PencilIcon />
                    </button>
                    <button
                      className="session-action-btn danger"
                      title="Delete"
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteTarget(s);
                      }}
                    >
                      <CloseIcon />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Bottom */}
        <div className="sidebar-bottom">
          <button
            className="theme-toggle"
            onClick={onToggleTheme}
            title={isDark ? "Switch to light mode" : "Switch to dark mode"}
          >
            <span className="theme-toggle-icon">{isDark ? "☀" : "☾"}</span>
            {!collapsed && <span>{isDark ? "Light mode" : "Dark mode"}</span>}
          </button>
        </div>
      </aside>

      {/* Delete single chat */}
      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete chat?"
        description={deleteTarget ? `"${deleteTarget.name}" will be permanently deleted and cannot be recovered.` : ""}
        confirmLabel="Delete"
        danger
        onConfirm={() => {
          if (deleteTarget) onDelete(deleteTarget.id);
          setDeleteTarget(null);
        }}
        onCancel={() => setDeleteTarget(null)}
      />

      {/* Delete all chats */}
      <ConfirmDialog
        open={showDeleteAll}
        title="Clear all chats?"
        description={`All ${sessions.length} chat${sessions.length !== 1 ? "s" : ""} will be permanently deleted. This cannot be undone.`}
        confirmLabel="Clear all"
        danger
        onConfirm={() => {
          onDeleteAll();
          setShowDeleteAll(false);
        }}
        onCancel={() => setShowDeleteAll(false)}
      />
    </>
  );
}
