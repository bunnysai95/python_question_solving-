import { useEffect, useRef, useState } from "react";

import { api } from "../api";

export default function Chat() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi! Ask me anything." },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollerRef = useRef(null);

  useEffect(() => {
    // autoscroll to bottom on new message
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function sendMessage(e) {
    e.preventDefault();
    if (!input.trim() || busy) return;

    const token = localStorage.getItem("access_token");
    const next = [...messages, { role: "user", content: input.trim() }];
    setMessages(next);
    setInput("");
    setBusy(true);

    try {
  const res = await fetch(api("/api/chat"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ messages: next }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const msg = Array.isArray(err.detail)
          ? err.detail.map(d => `${(d.loc && d.loc.at(-1)) || "field"}: ${d.msg}`).join(" • ")
          : (err.detail || "Chat failed");
        throw new Error(msg);
      }

      const data = await res.json();
      setMessages(s => [...s, { role: "assistant", content: data.reply }]);
    } catch (e) {
      setMessages(s => [...s, { role: "assistant", content: `❌ ${e.message}` }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="chat-wrap">
      <div className="chat-log" ref={scrollerRef}>
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            <div className="content">{m.content}</div>
          </div>
        ))}
        {busy && <div className="bubble assistant"><div className="content">…</div></div>}
      </div>

      <form className="chat-input" onSubmit={sendMessage}>
        <input
          className="input"
          placeholder="Type your message…"
          value={input}
          onChange={e => setInput(e.target.value)}
        />
        <button className="btn" type="submit" disabled={busy || !input.trim()}>
          {busy ? "Sending…" : "Send"}
        </button>
      </form>
    </div>
  );
}
