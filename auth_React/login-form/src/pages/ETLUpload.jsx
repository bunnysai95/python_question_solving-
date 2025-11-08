import React, { useEffect, useRef, useState } from "react";
import Papa from "papaparse";

export default function ETLUpload() {
  const [fileName, setFileName] = useState("");
  const [preview, setPreview] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [polling, setPolling] = useState(false);
  const pollRef = useRef(null);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function pollTask(taskId) {
    setPolling(true);
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/etl/tasks/${taskId}`);
        if (!res.ok) throw new Error(`status:${res.status}`);
        const j = await res.json();
        setStatus({ ok: true, data: j });
        if (j.status && ["success", "failed"].includes(j.status)) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setPolling(false);
        }
      } catch (err) {
        setStatus({ ok: false, error: String(err) });
        clearInterval(pollRef.current);
        pollRef.current = null;
        setPolling(false);
      }
    }, 2000);
  }

  async function handleFile(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFileName(f.name);
    setStatus(null);
    setLoading(true);

    // show a small preview using PapaParse (first 10 rows)
    Papa.parse(f, {
      header: true,
      preview: 10,
      skipEmptyLines: true,
      complete: (results) => setPreview(results.data),
    });

    try {
      const token = localStorage.getItem("access_token");
      const fd = new FormData();
      fd.append("file", f, f.name);

      const res = await fetch("/api/etl/upload", {
        method: "POST",
        headers: {
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: fd,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setStatus({ ok: false, code: res.status, detail: err });
      } else {
        const json = await res.json();
        setStatus({ ok: true, data: json });
        if (json.task_id) {
          pollTask(json.task_id);
        }
      }
    } catch (err) {
      setStatus({ ok: false, error: String(err) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page etl-upload">
      <h2>ETL Upload</h2>
      <p>Upload a CSV file. You can preview first rows in the browser; the file will be sent to the server for processing.</p>

      <label className="file-input">
        <input type="file" accept=".csv,text/csv" onChange={handleFile} />
      </label>

      {fileName && <div>Selected file: <strong>{fileName}</strong></div>}

      {preview && (
        <div className="preview">
          <h4>Preview (first rows)</h4>
          <pre style={{ maxHeight: 200, overflow: "auto" }}>{JSON.stringify(preview, null, 2)}</pre>
        </div>
      )}

      {loading && <div>Uploading...</div>}

      {polling && <div>Processing on server... (polling)</div>}

      {status && (
        <div className={`etl-status ${status.ok ? "ok" : "error"}`}>
          {status.ok ? (
            <pre>{JSON.stringify(status.data, null, 2)}</pre>
          ) : (
            <pre>{JSON.stringify(status.detail || status.error || status, null, 2)}</pre>
          )}
        </div>
      )}
    </div>
  );
}
