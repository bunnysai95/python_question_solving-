import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";

const LoginSchema = z.object({
  username: z.string().min(3, "At least 3 characters").max(20).regex(/^[a-zA-Z0-9_]+$/, "Only letters, numbers, underscore"),
  password: z.string().min(8, "At least 8 characters")
    .regex(/[a-z]/, "Include a lowercase letter")
    .regex(/[A-Z]/, "Include an uppercase letter")
    .regex(/[0-9]/, "Include a number")
    .regex(/[^A-Za-z0-9]/, "Include a special character"),
});

const API_URL = import.meta.env?.VITE_API_URL ?? "http://localhost:8000/default";
import { api } from "../api";


export default function Login() {
  const navigate = useNavigate();            
  const [showPw, setShowPw] = useState(false);
  const [message, setMessage] = useState("");

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting, isValid, isDirty },
    reset,
  } = useForm({
    resolver: zodResolver(LoginSchema),
    mode: "onBlur",
    reValidateMode: "onChange",
    defaultValues: { username: "", password: "" },
  });

  async function onSubmit(values) {
    setMessage("");
    console.log("POST ->", api("/api/login"), values.username);
    try {
      const res = await fetch(api("/api/login"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Invalid username or password");
      }

  const data = await res.json();
      localStorage.setItem("access_token", data.access_token);

      setMessage("✅ Logged in!");
      reset({ username: values.username, password: "" });
      navigate("/dashboard");                // ✅ and call it here
    } catch (e) {
      setMessage(`❌ ${e.message || "Login failed. Try again."}`);
    }
  }

  return (
    <>
      <h1 className="title">Welcome</h1>
      <p className="subtitle">Sign in to continue</p>

      <form className="form" onSubmit={handleSubmit(onSubmit)} noValidate>
        <label htmlFor="username" className="label">Username</label>
        <input
          id="username"
          type="text"
          autoComplete="username"
          placeholder="e.g. sai_pranay"
          className={`input ${errors.username ? "input-error" : ""}`}
          aria-invalid={!!errors.username}
          aria-describedby="username-help"
          {...register("username")}
        />
        <div id="username-help" className="help">
          {errors.username ? errors.username.message : "3–20 chars: letters, numbers, _"}
        </div>

        <label htmlFor="password" className="label">Password</label>
        <div className={`pw-wrap ${errors.password ? "pw-error" : ""}`}>
          <input
            id="password"
            type={showPw ? "text" : "password"}
            autoComplete="current-password"
            placeholder="••••••••"
            className="input pw-input"
            aria-invalid={!!errors.password}
            aria-describedby="password-help"
            {...register("password")}
          />
          <button
            type="button"
            className="pw-toggle"
            onClick={() => setShowPw((s) => !s)}
            aria-label={showPw ? "Hide password" : "Show password"}
          >
            {showPw ? "Hide" : "Show"}
          </button>
        </div>
        <div id="password-help" className="help">
          {errors.password ? errors.password.message : "Min 8 chars with a-z, A-Z, 0-9, special"}
        </div>

        <button
          type="submit"
          className="btn"
          disabled={!isDirty || !isValid || isSubmitting}
          aria-busy={isSubmitting}
        >
          {isSubmitting ? "Signing in..." : "Sign In"}
        </button>

        <div className="below-cta">
          <span>New here?</span>{" "}
          <Link to="/register" className="text-link">Create account</Link>
        </div>

        {message && <div className="message" role="status">{message}</div>}
      </form>
    </>
  );
}
