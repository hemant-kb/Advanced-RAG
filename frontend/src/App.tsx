import { useCallback, useEffect, useRef, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
import {
  SessionInfo,
  createSession,
  deleteSession,
  listSessions,
} from "./lib/api";
import { TurnMeta } from "./hooks/useStream";

export default function App() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  // Lives here so it survives ChatWindow remounts on session switch (key={active.id}).
  const turnMetaMapRef = useRef<Map<string, TurnMeta>>(new Map());
  const [isDark, setIsDark] = useState(() => {
    return localStorage.getItem("theme") !== "light";
  });

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.remove("light");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.add("light");
      localStorage.setItem("theme", "light");
    }
  }, [isDark]);

  const refresh = useCallback(async () => {
    const list = await listSessions();
    setSessions(list);
    setActiveId((cur) => {
      if (!cur && list.length > 0) return list[0].id;
      if (cur && !list.find((s) => s.id === cur)) return list[0]?.id ?? null;
      return cur;
    });
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleNew() {
    const s = await createSession();
    await refresh();
    setActiveId(s.id);
  }

  async function handleDelete(id: string) {
    await deleteSession(id);
    await refresh();
  }

  async function handleDeleteAll() {
    await Promise.all(sessions.map((s) => deleteSession(s.id)));
    setSessions([]);
    setActiveId(null);
  }

  function handleRename(id: string, name: string) {
    setSessions((prev) =>
      prev.map((s) => (s.id === id ? { ...s, name } : s))
    );
  }

  const active = sessions.find((s) => s.id === activeId) || null;

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={handleNew}
        onDelete={handleDelete}
        onDeleteAll={handleDeleteAll}
        onRename={handleRename}
        isDark={isDark}
        onToggleTheme={() => setIsDark((d) => !d)}
      />
      {active ? (
        <ChatWindow
          key={active.id}
          session={active}
          onDocumentChange={() => refresh()}
          onSessionRenamed={handleRename}
          turnMetaMapRef={turnMetaMapRef}
        />
      ) : (
        <div className="no-session">
          <h2>No active chat</h2>
          <p>Click "New chat" to get started.</p>
        </div>
      )}
    </div>
  );
}
