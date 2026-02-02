import { useEffect, useState } from "react";
import { api } from "../api";

export default function ResearchList() {
  const [items, setItems] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      setError("");
      try {
        const token = localStorage.getItem("access_token");
        if (!token) {
          setError("Not logged in");
          return;
        }
        const res = await fetch(api("/api/research"), {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "Failed to load");
        }
        const json = await res.json();
        setItems(json || []);
      } catch (e) {
        setError(e.message || "Error");
      }
    }
    load();
  }, []);

  if (error) return <div className="message">❌ {error}</div>;
  if (!items) return <div className="message">Loading…</div>;

  return (
    <div className="research-container">
      <h1 className="title">Research submissions</h1>
      <p className="subtitle">All stored UX feedback</p>

      {items.length === 0 ? (
        <div className="empty-state">No submissions yet</div>
      ) : (
        <div className="table-wrapper">
          <table className="research-table" aria-describedby="research-list">
            <thead>
              <tr>
                <th>ID</th>
                <th>User Name</th>
                <th>Email</th>
                <th>Rating</th>
                <th>Comments</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={r.id}>
                  <td className="col-id">{r.id}</td>
                  <td className="col-name">{r.username ?? "-"}</td>
                  <td className="col-email">{r.email}</td>
                  <td className="col-rating">{r.rating}</td>
                  <td className="col-comments">{r.comments || "-"}</td>
                  <td className="col-date">{r.created_at ? new Date(r.created_at).toLocaleString() : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
