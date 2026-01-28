import { useEffect, useState } from "react"; // usestate=> stores data,useeffect=> run code when the page loads
import { Link } from "react-router-dom"; // Link=> navigation between pages
import { api } from "../api"; // helper to construct API URLs

// Dashboard component definition
export default function Dashboard() { // main dashboard page after login
  const [me, setMe] = useState(null); // state to hold user profile data
  const [error, setError] = useState(""); // state to hold error messages

  // Fetch user profile on component mount
  useEffect(() => {
    const token = localStorage.getItem("access_token"); //access token from local storage assigned to token
    if (!token) { setError("Not logged in"); // if no token, set error and exit
        return;
       } 

  fetch(api("/api/me"), { // fetch user profile from API
      headers: { Authorization: `Bearer ${token}` } // include token in request headers with Bearer scheme sent to backend
    })
      .then(async (r) => (r.ok ? setMe(await r.json()) // if response is ok, set user profile data
                               : setError((await r.json()).detail || "Failed to load profile"))) // else set error message from response
      .catch(() => setError("Network error")); // handle network errors
  }, []);

  if (error) return <div className="message">❌ {error}</div>; // display error message
  if (!me) return <div className="message">Loading…</div>; // display loading message while fetching profile

  return (
    <div>
      <h1 className="title">welcome {me.firstName}</h1><p className="subtitle">You’re signed in as <strong>{me.username}</strong></p>

      {/* buttons */}
      <div className="grid-2" style={{ marginTop: "1rem" }}>
        <Link to="/profile" className="btn" style={{ textAlign: "center" }}>
          Complete profile
        </Link>
        <Link to="/research" className="btn" style={{ textAlign: "center" }}>
          UX Research form
        </Link>
        <Link to="/research/list" className="btn" style={{ textAlign: "center" }}>
          View submissions
        </Link>
        <Link to="/etl" className="btn" style={{ textAlign: "center" }}>
          ETL Upload
        </Link>
        <Link to="/chat" className="btn" style={{ textAlign: "center" }}>
          Open chat
        </Link>
                <Link to="/Tasks" className="btn" style={{ textAlign: "center" }}>
          Task History
        </Link>
                <Link to="/About" className="btn" style={{ textAlign: "center" }}>
          About
        </Link>

        <button className="btn" onClick={() => {
            localStorage.removeItem("access_token");
            window.location.href = "/";
          }}> Log out </button>
      </div>
    </div>
  );
}
