import { Outlet } from "react-router-dom";
import Navbar from "./components/Navbar";

export default function App() {
  return (
    <div className="app">
      <Navbar />
      <main className="card" role="main">
        <Outlet />
      </main>

      <footer className="footer">
        <small>
          <a className="text-link" href="/">Sign in</a> ·{" "}
          <a className="text-link" href="/register">Create account</a>
        </small>
      </footer>
    </div>
  );
}
