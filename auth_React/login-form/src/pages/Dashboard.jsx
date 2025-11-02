import { useEffect, useState } from "react";
import { Link } from "react-router-dom"; // ✅ keep only this import

import { api } from "../api";

export default function Dashboard() {
  const [me, setMe] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (!token) { setError("Not logged in"); return; }

  fetch(api("/api/me"), {
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

      {/* buttons */}
      <div className="grid-2" style={{ marginTop: "1rem" }}>
        <Link to="/profile" className="btn" style={{ textAlign: "center" }}>
          Complete profile
        </Link>

        {/* 👇 add this chat link */}
        <Link to="/chat" className="btn" style={{ textAlign: "center" }}>
          Open chat
        </Link>

        <button
          className="btn"
          onClick={() => {
            localStorage.removeItem("access_token");
            window.location.href = "/";
          }}
        >
          Log out
        </button>
      </div>
    </>
  );
}
