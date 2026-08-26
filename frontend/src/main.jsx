import React, { useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import "./styles.css";

const API_URL = "http://127.0.0.1:8000";

async function readSseStream(response, onToken, onDone) {
  if (!response.ok || !response.body) {
    throw new Error(`Server error (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const processEvent = (rawEvent) => {
    const lines = rawEvent.split("\n");
    const eventName = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
    const data = lines.find((line) => line.startsWith("data:"))?.slice(5).trim();
    if (!data) return;

    const payload = JSON.parse(data);
    if (eventName === "token") onToken(payload.content || "");
    if (eventName === "done") onDone(payload);
    if (eventName === "error") throw new Error(payload.message || "Chat request failed.");
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    events.forEach(processEvent);
    if (done) break;
  }
  if (buffer.trim()) processEvent(buffer);
}

function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [uploadState, setUploadState] = useState({ status: "idle", detail: "No document loaded" });
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef(null);

  const uploadPdf = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError("");
    setUploadState({ status: "loading", detail: "Processing document..." });
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Upload failed.");
      setUploadState({ status: "ready", detail: `${result.pages} pages · ${result.chunks} chunks indexed` });
    } catch (uploadError) {
      setUploadState({ status: "error", detail: "Upload failed" });
      setError(uploadError.message);
    } finally {
      event.target.value = "";
    }
  };

  const sendQuestion = async (event) => {
    event?.preventDefault();
    const text = question.trim();
    if (!text || isSending) return;
    setError("");
    setQuestion("");
    setMessages((current) => [...current, { role: "user", content: text }, { role: "assistant", content: "", streaming: true }]);
    setIsSending(true);
    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      await readSseStream(
        response,
        (token) => setMessages((current) => current.map((message, index) => index === current.length - 1 ? { ...message, content: message.content + token } : message)),
        (result) => setMessages((current) => current.map((message, index) => index === current.length - 1 ? { ...message, content: result.answer || message.content, sourcePages: result.source_pages || [], streaming: false } : message)),
      );
    } catch (chatError) {
      setError(chatError.message || "Could not connect to the backend.");
      setMessages((current) => current.map((message, index) => index === current.length - 1 ? { ...message, content: "", streaming: false, failed: true } : message));
    } finally {
      setIsSending(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar"><div className="brand-mark">R</div><div><p className="eyebrow">Document intelligence</p><h1>RAG Chatbot</h1></div><span className="connection-dot">● Local workspace</span></header>
      <section className="workspace-grid">
        <aside className="document-panel">
          <p className="panel-kicker">Source document</p><h2>Ask your PDF</h2>
          <div className={`upload-box ${uploadState.status}`} onClick={() => fileInput.current?.click()} role="button" tabIndex="0">
            <input ref={fileInput} type="file" accept="application/pdf,.pdf" onChange={uploadPdf} />
            <span className="upload-icon">↑</span><strong>{uploadState.status === "loading" ? "Processing..." : "Upload a PDF"}</strong><small>PDF files only</small>
          </div>
          <div className={`file-status ${uploadState.status}`}><span className="status-pip" />{uploadState.detail}</div>
          <div className="boundary-note"><span>◎</span><p>Answers stay grounded in the uploaded document. Source pages come directly from retrieval.</p></div>
        </aside>
        <section className="chat-panel">
          <div className="chat-header"><div><p className="panel-kicker">Conversation</p><h2>Document Q&A</h2></div><span className="model-tag">PDF ONLY</span></div>
          <div className="messages" aria-live="polite">
            {messages.length === 0 && <div className="empty-state"><div className="empty-icon">✦</div><h3>What would you like to find?</h3><p>Upload a document, then ask a precise question about its contents.</p></div>}
            {messages.map((message, index) => <article className={`message ${message.role}`} key={`${index}-${message.content}`}><span className="message-label">{message.role === "user" ? "You" : "Assistant"}</span><p>{message.content || (message.streaming ? "Thinking..." : message.failed ? "No response received." : "")}</p>{message.sourcePages?.length > 0 && <small className="sources">Source pages {message.sourcePages.join(", ")}</small>}</article>)}
          </div>
          {error && <div className="error-banner">{error}</div>}
          <form className="composer" onSubmit={sendQuestion}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about your PDF..." disabled={isSending} /><button type="submit" disabled={isSending || !question.trim()} aria-label="Send question">{isSending ? "..." : "↗"}</button></form>
        </section>
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
