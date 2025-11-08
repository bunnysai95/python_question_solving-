import React, { useEffect, useState } from "react";

export default function TaskHistory() {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const res = await fetch(`/api/etl/tasks?limit=50`);
        if (!res.ok) throw new Error(`status:${res.status}`);
        const j = await res.json();
        setTasks(j);
      } catch (err) {
        setError(String(err));
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="page tasks">
      <h2>Task History</h2>
      {loading && <div>Loading...</div>}
      {error && <div className="error">{error}</div>}
      <table className="task-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Status</th>
            <th>Started</th>
            <th>Finished</th>
            <th>Error / Notes</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map((t) => (
            <tr key={t.id}>
              <td>{t.id}</td>
              <td>{t.name}</td>
              <td>{t.status}</td>
              <td>{t.started_at ? new Date(t.started_at).toLocaleString() : "-"}</td>
              <td>{t.finished_at ? new Date(t.finished_at).toLocaleString() : "-"}</td>
              <td style={{ maxWidth: 400, overflow: "auto" }}><pre>{t.error}</pre></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
