export interface SessionInfo {
  id: string;
  name: string;
  created_at: string;
  has_document: boolean;
  document_name?: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface UploadStatus {
  status: "pending" | "processing" | "complete" | "error";
  progress: number;
  message: string;
  chunks_created: number;
}

const API = "/api";

export async function listSessions(): Promise<SessionInfo[]> {
  return (await fetch(`${API}/sessions`)).json();
}

export async function createSession(name?: string): Promise<SessionInfo> {
  const r = await fetch(`${API}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return r.json();
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`${API}/sessions/${id}`, { method: "DELETE" });
}

export async function renameSession(id: string, name: string): Promise<SessionInfo> {
  const r = await fetch(`${API}/sessions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return r.json();
}

export async function getHistory(id: string): Promise<{ messages: ChatMessage[] }> {
  return (await fetch(`${API}/sessions/${id}/history`)).json();
}

export async function uploadPdf(
  id: string,
  file: File,
  onProgress?: (message: string, progress: number) => void,
): Promise<UploadStatus> {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${API}/sessions/${id}/upload`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) throw new Error(await r.text());

  // Stream SSE progress events until complete or error
  const reader = r.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalStatus: UploadStatus = { status: "processing", progress: 0, message: "", chunks_created: 0 };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() || "";
    for (const block of lines) {
      const line = block.trim();
      if (!line.startsWith("data:")) continue;
      try {
        const evt = JSON.parse(line.slice(5).trim());
        if (evt.type === "progress") {
          onProgress?.(evt.message, evt.progress ?? 0.5);
          finalStatus = { status: "processing", progress: evt.progress ?? 0.5, message: evt.message, chunks_created: 0 };
        } else if (evt.type === "complete") {
          finalStatus = { status: "complete", progress: 1.0, message: evt.message, chunks_created: evt.chunks_created ?? 0 };
          onProgress?.(evt.message, 1.0);
        } else if (evt.type === "error") {
          throw new Error(evt.message);
        }
      } catch (e: any) {
        if (e.message && !e.message.includes("JSON")) throw e;
      }
    }
  }
  return finalStatus;
}

export async function autoNameSession(id: string, firstMessage: string): Promise<string> {
  const r = await fetch(`${API}/sessions/${id}/auto-name`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: firstMessage }),
  });
  if (!r.ok) return "";
  const data = await r.json();
  return data.name || "";
}

export function chatUrl(id: string): string {
  return `${API}/sessions/${id}/chat`;
}
