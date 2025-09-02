import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";

// Helpers
const today = new Date();
const thirteenYearsAgo = new Date(today.getFullYear() - 13, today.getMonth(), today.getDate());

// Zod schema for registration
const RegisterSchema = z.object({
  firstName: z.string().min(2, "First name is too short").max(50),
  lastName: z.string().min(2, "Last name is too short").max(50),
  dob: z.string().refine((v) => {
    const d = new Date(v);
    return !isNaN(d.getTime()) && d <= thirteenYearsAgo;
  }, "You must be at least 13 years old"),
  username: z.string().min(3).max(20).regex(/^[a-zA-Z0-9_]+$/, "Only letters, numbers, underscore"),
  password: z.string().min(8, "At least 8 characters")
    .regex(/[a-z]/, "Include a lowercase letter")
    .regex(/[A-Z]/, "Include an uppercase letter")
    .regex(/[0-9]/, "Include a number")
    .regex(/[^A-Za-z0-9]/, "Include a special character"),
  confirmPassword: z.string().min(1, "Please confirm your password"),
  phone: z.string()
    .regex(/^\+?[0-9()\-\s]{7,20}$/, "Enter a valid phone number"),
}).refine((data) => data.password === data.confirmPassword, {
  path: ["confirmPassword"],
  message: "Passwords do not match",
});

// ✅ Add this line (base URL for your API)
const API_URL = import.meta.env?.VITE_API_URL ?? "http://localhost:8000";

export default function Register() {
  const [showPw, setShowPw] = useState(false);
  const [showPw2, setShowPw2] = useState(false);
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const { register, handleSubmit, formState: { errors, isSubmitting, isValid, isDirty } } = useForm({
    resolver: zodResolver(RegisterSchema),
    mode: "onBlur",
    reValidateMode: "onChange",
    defaultValues: {
      firstName: "", lastName: "", dob: "", username: "",
      password: "", confirmPassword: "", phone: ""
    }
  });

  // 🔁 Replace your onSubmit with this
  async function onSubmit(values) {
    setMessage("");
    try {
      // (optional) lightly normalize fields before sending
      const payload = {
        ...values,
        firstName: values.firstName.trim(),
        lastName: values.lastName.trim(),
        username: values.username.trim(),
        phone: values.phone?.trim() || null,   // backend allows null
      };

      const res = await fetch(`${API_URL}/api/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        // Common case from backend: 409 when username is taken
        throw new Error(err.detail || "Registration failed");
      }

      setMessage("✅ Account created. Redirecting to Sign in…");
      setTimeout(() => navigate("/"), 600);
    } catch (e) {
      setMessage(`❌ ${e.message || "Could not create account. Try again."}`);
    }
  }

  return (
    <>
      <h1 className="title">Create account</h1>
      <p className="subtitle">It’s quick and secure</p>

      <form className="form" onSubmit={handleSubmit(onSubmit)} noValidate>
        {/* First / Last name */}
        <div className="grid-2">
          <div>
            <label htmlFor="firstName" className="label">First name</label>
            <input id="firstName" className={`input ${errors.firstName ? "input-error" : ""}`} {...register("firstName")} />
            <div className="help">{errors.firstName?.message ?? " "}</div>
          </div>
          <div>
            <label htmlFor="lastName" className="label">Last name</label>
            <input id="lastName" className={`input ${errors.lastName ? "input-error" : ""}`} {...register("lastName")} />
            <div className="help">{errors.lastName?.message ?? " "}</div>
          </div>
        </div>

        {/* DOB */}
        <label htmlFor="dob" className="label">Date of birth</label>
        <input id="dob" type="date" className={`input ${errors.dob ? "input-error" : ""}`} {...register("dob")} />
        <div className="help">{errors.dob?.message ?? " "}</div>

        {/* Username */}
        <label htmlFor="username" className="label">Username</label>
        <input id="username" className={`input ${errors.username ? "input-error" : ""}`} placeholder="e.g. sai_pranay" {...register("username")} />
        <div className="help">{errors.username?.message ?? "3–20 chars: letters, numbers, _"}</div>

        {/* Password */}
        <label htmlFor="password" className="label">Password</label>
        <div className={`pw-wrap ${errors.password ? "pw-error" : ""}`}>
          <input id="password" type={showPw ? "text" : "password"} className="input pw-input" placeholder="••••••••" {...register("password")} />
          <button type="button" className="pw-toggle" onClick={() => setShowPw((s) => !s)}>{showPw ? "Hide" : "Show"}</button>
        </div>
        <div className="help">{errors.password?.message ?? "Min 8 with a-z, A-Z, 0-9, special"}</div>

        {/* Confirm Password */}
        <label htmlFor="confirmPassword" className="label">Confirm password</label>
        <div className={`pw-wrap ${errors.confirmPassword ? "pw-error" : ""}`}>
          <input id="confirmPassword" type={showPw2 ? "text" : "password"} className="input pw-input" placeholder="••••••••" {...register("confirmPassword")} />
          <button type="button" className="pw-toggle" onClick={() => setShowPw2((s) => !s)}>{showPw2 ? "Hide" : "Show"}</button>
        </div>
        <div className="help">{errors.confirmPassword?.message ?? "Re-enter the same password"}</div>

        {/* Phone */}
        <label htmlFor="phone" className="label">Phone number</label>
        <input id="phone" inputMode="tel" placeholder="+1 (555) 123-4567" className={`input ${errors.phone ? "input-error" : ""}`} {...register("phone")} />
        <div className="help">{errors.phone?.message ?? "Include country code if outside US"}</div>

        <button type="submit" className="btn" disabled={!isDirty || !isValid || isSubmitting} aria-busy={isSubmitting}>
          {isSubmitting ? "Creating..." : "Create account"}
        </button>

        <div className="below-cta">
          <span>Already have an account?</span>{" "}
          <Link to="/" className="text-link">Sign in</Link>
        </div>

        {message && <div className="message" role="status">{message}</div>}
      </form>
    </>
  );
}
