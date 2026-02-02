import { Outlet } from "react-router-dom";
import Navbar from "./components/Navbar";

export default function App() {
  return (
    <div className="app">
      <Navbar />
      <main className="card" role="main">
        <Outlet />
      </main>
    </div>
  );
}
