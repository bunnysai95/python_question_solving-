import React, { useEffect, useRef, useState } from "react"; 
// useState  → local component state
// useRef    → mutable value that persists across renders (interval ID)
// useEffect → lifecycle hook (mount / unmount side effects)
import Papa from "papaparse"; // PapaParse → client-side CSV parsing (used only for preview, not upload)
// ETLUpload component definition
export default function ETLUpload() {
  const [fileName, setFileName] = useState(""); // selected file name & filename == currnet file_name, useState= hook,setfileName= function to update file name
  const [preview, setPreview] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [polling, setPolling] = useState(false);
  const pollRef = useRef(null);

  useEffect(() => {
    return () => {
      // cleanup on unmount, clear polling interval, prevent memory leaks, only run on unmount
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);
// Polling function to check task status
  async function pollTask(taskId) { // taskId = unique identifier for the ETL task, async = non blocking function
    setPolling(true); // set polling state to true, purpose = indicate polling in progress
    if (pollRef.current) clearInterval(pollRef.current); // clear existing interval if any, purpose = avoid multiple intervals
    pollRef.current = setInterval(async () => { // set up new interval to poll every 2 seconds
      try {
        const res = await fetch(`/api/etl/tasks/${taskId}`); // fetch task status from API
        if (!res.ok) throw new Error(`status:${res.status}`); // throw error if response not ok
        const j = await res.json(); // parse JSON response
        setStatus({ ok: true, data: j }); // update status state with task data, data = task status info
        if (j.status && ["success", "failed"].includes(j.status)) { // check if task is complete
          clearInterval(pollRef.current); // clear polling interval
          pollRef.current = null; // reset pollRef
          setPolling(false); // set polling state to false
        }
      } catch (err) { // handle errors
        setStatus({ ok: false, error: String(err) }); // update status state with error message
        clearInterval(pollRef.current); // clear polling interval on error
        pollRef.current = null; // reset pollRef
        setPolling(false); // set polling state to false
      }
    }, 2000); // poll every 2 seconds 2000 milliseconds
  }

  async function handleFile(e) { // handle file selection event
    const f = e.target.files[0]; // get the selected file
    if (!f) return; // if no file selected, exit
    setFileName(f.name); // update fileName state with selected file name
    setStatus(null); // reset status state
    setLoading(true); // set loading state to true

    // show a small preview using PapaParse (first 10 rows)
    Papa.parse(f, { // parse CSV file for preview
      header: true, // treat first row as header
      preview: 10,  // limit to first 10 rows
      skipEmptyLines: true, // skip empty lines
      complete: (results) => setPreview(results.data), // update preview state with parsed data
    });
// send file to server important step 
    try { // upload the file to the server
      const token = localStorage.getItem("access_token"); // get access token from local storage
      const fd = new FormData(); // create FormData object for file upload
      fd.append("file", f, f.name); // append file to FormData

      const res = await fetch("/api/etl/upload", { // send POST request to upload endpoint
        method: "POST", // use POST method
        headers: { // include Authorization header if token exists
          ...(token ? { Authorization: `Bearer ${token}` } : {}), // Bearer token for authentication
        },
        body: fd, // set FormData as request body
      });

      if (!res.ok) { // handle non-ok responses
        const err = await res.json().catch(() => ({})); // try to parse error response as JSON
        setStatus({ ok: false, code: res.status, detail: err }); // update status state with error details
      } else {
        const json = await res.json(); // parse successful response as JSON
        setStatus({ ok: true, data: json }); // update status state with response data
        if (json.task_id) { // if task_id is present, start polling for task status
          pollTask(json.task_id); //  start polling with the given task ID
        }
      }
    } catch (err) { // handle network or other errors
      setStatus({ ok: false, error: String(err) }); // update status state with error message
    } finally { //  final cleanup
      setLoading(false); // set loading state to false
    }
  }

  return (
    <div className="page etl-upload"> {/* main ETL upload page container */}
      <h2>ETL Upload</h2>
      <h3><p>Upload a CSV file. You can preview first rows in the browser; the file will be sent to the server for processing.</p></h3>

      <label className="file-input"> {/* file input label */}
        <input type="file" accept=".csv,text/csv" onChange={handleFile} /> {/* file input element with CSV type */}
      </label>

      {fileName && <div>Selected file: <strong>{fileName}</strong></div>} {/* display selected file name */}

      {preview && (
        <div className="preview">
          <h4>Preview (first rows)</h4>
          <pre style={{ maxHeight: 200, overflow: "auto" }}>{JSON.stringify(preview, null, 2)}</pre>
        </div>
      )}

      {loading && <div>Uploading...</div>} {/* display uploading status */}

      {polling && <div>Processing on server... (polling)</div>} {/* display polling status */}

      {status && (
        <div className={`etl-status ${status.ok ? "ok" : "error"}`}> {/* status display container with dynamic class */}
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
