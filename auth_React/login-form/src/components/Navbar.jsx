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
            background: linear-gradient(90deg, rgba(10,20,40,0.95) 0%, rgba(10,20,40,0.9) 100%);
            padding: 1rem 2rem;
            border-bottom: 1px solid rgba(59,130,246,0.15);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
            width: 100%;
          }

          .nav-inner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 100%;
            margin: auto;
          }

          .nav-left {
            display: flex;
            gap: 3rem;
            align-items: center;
            flex: 1;
          }

          .brand {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.5px;
          }

          .brand a {
            text-decoration: none;
          }

          .links {
            display: flex;
            gap: 2rem;
            list-style: none;
            margin: 0;
            padding: 0;
            align-items: center;
          }

          .links a {
            color: #8892a8;
            text-decoration: none;
            font-weight: 500;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            padding: 0.5rem 0;
            border-bottom: 2px solid transparent;
          }

          .links a:hover {
            color: #60a5fa;
            border-bottom-color: #60a5fa;
          }

          .logout {
            background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
            border: none;
            color: white;
            cursor: pointer;
            padding: 0.75rem 1.5rem;
            border-radius: 0.75rem;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            box-shadow: 0 8px 16px rgba(59,130,246,0.3);
          }

          .logout:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(59,130,246,0.4);
          }

          .logout:active {
            transform: translateY(0);
          }
        `}
      </style>

    <nav className="navbar">
      <div className="nav-inner">
        <div className="nav-left">
          <div className="brand"><Link to="/dashboard">MyApp</Link></div>
          <ul className="links">
            <li><Link to="/etl">ETL Upload</Link></li>
            <li><Link to="/tasks">Task History</Link></li>
            <li><Link to="/about">About Me</Link></li>
            <li><Link to="/profile">Profile</Link></li>
          </ul>
        </div>
        <button className="logout" onClick={handleLogout}>Log Out</button>
      </div>
    </nav>
    </>
  );
}
