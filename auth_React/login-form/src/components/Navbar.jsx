import { Link, useNavigate } from "react-router-dom"; // Link=> navigation between pages, useNavigate=> programmatic navigation

export default function Navbar() { // Navbar component definition
  const navigate = useNavigate(); // get navigate function

  function handleLogout() { // logout handler
    localStorage.removeItem("access_token"); // optional: remove any user info
    navigate("/"); // redirect to login page
  }

  return (
       <>
      <style>
        {`
          .navbar {
            background: rgba(10,20,40,0.9);
            padding: 12px 24px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
          }

          .nav-inner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: auto;
          }

          .brand {
            font-size: 1.4rem;
            font-weight: 600;
            color: white;
          }

          .links {
            display: flex;
            gap: 1.5rem;
            list-style: none;
            margin: 0;
            padding: 0;
          }

          .links a {
            color: #cbd5f5;
            text-decoration: none;
          }

          .logout {
            background: none;
            border: none;
            color: #ffb4b4;
            cursor: pointer;
          }
        `}
      </style>

    <nav className="navbar">
      <div className="nav-inner">
        <div className="brand"><Link to="/dashboard">MyApp</Link></div>
        <ul className="links">
          <li><Link to="/etl">ETL Upload</Link></li>
          <li><Link to="/tasks">Task History</Link></li>
          <li><Link to="/about">About Me</Link></li>
          <li><Link to="/profile">Profile</Link></li>
          <li><button className="link-like" onClick={handleLogout}>Log Out</button></li>
        </ul>
      </div>
    </nav>
    </>
  );
}
