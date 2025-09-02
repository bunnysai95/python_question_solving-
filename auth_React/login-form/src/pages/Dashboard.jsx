import { useEffect, useState } from "react";

const API_URL = import.meta.env?.VITE_API_URL ?? "http://localhost:8000";

export default function Dashboard() {
  const [me, setMe] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (!token) { setError("Not logged in"); return; }

    fetch(`${API_URL}/api/me`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(async (r) => (r.ok ? setMe(await r.json())
                               : setError((await r.json()).detail || "Failed to load profile")))
      .catch(() => setError("Network error"));
  }, []);

  if (error) return <div className="message">❌ {error}</div>;
  if (!me) return <div className="message">Loading…</div>;

  return (
    <>
      <h1 className="title">Hi {me.firstName} 👋</h1>
      <p className="subtitle">You’re signed in as <strong>{me.username}</strong></p>
      <button
        className="btn"
        onClick={() => { localStorage.removeItem("access_token"); window.location.href = "/"; }}
      >
        Log out
      </button>
    </>
  );
}
