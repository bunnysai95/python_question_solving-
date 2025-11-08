import { Link, useNavigate } from "react-router-dom";

export default function Navbar() {
  const navigate = useNavigate();

  function handleLogout() {
    localStorage.removeItem("access_token");
    // optional: remove any user info
    navigate("/");
  }

  return (
    <nav className="nav">
      <div className="nav-inner">
        <div className="nav-brand">MyApp</div>
        <ul className="nav-links">
          <li><Link to="/dashboard">Home</Link></li>
          <li><Link to="/etl">ETL Upload</Link></li>
          <li><Link to="/tasks">Task History</Link></li>
          <li><Link to="/about">About Me</Link></li>
          <li><Link to="/profile">Profile</Link></li>
          <li><button className="link-like" onClick={handleLogout}>Log out</button></li>
        </ul>
      </div>
    </nav>
  );
}
