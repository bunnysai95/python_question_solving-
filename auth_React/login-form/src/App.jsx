import { Outlet, Link } from "react-router-dom";

export default function App() {
  return (
    <div className="app">
      <main className="card" role="main">
        <Outlet />
      </main>

      <footer className="footer">
        <small>
          <Link to="/" className="text-link">Sign in</Link> ·{" "}
          <Link to="/register" className="text-link">Create account</Link>
        </small>
      </footer>
    </div>
  );
}
