import { useState } from "react";
import { useForm } from "react-hook-form";
import { api } from "../api";
import { useNavigate, Link } from "react-router-dom";

export default function ResearchForm() {
  const { register, handleSubmit, formState: { errors, isSubmitting } } = useForm();
  const [message, setMessage] = useState("");
  const [saved, setSaved] = useState(false);
  const navigate = useNavigate();

  async function onSubmit(values) {
    setMessage("");
    try {
      const token = localStorage.getItem("access_token");
      if (!token) {
        setMessage("❌ Not logged in");
        return;
      }

      const payload = {
        firstName: values.firstName || "",
        lastName: values.lastName || "",
        email: values.email || "",
        phone: values.phone || "",
        address1: values.address1 || "",
        address2: values.address2 || "",
        city: values.city || "",
        region: values.region || "",
        postal: values.postal || "",
        country: values.country || "",
        rating: values.rating || 3,
        comments: values.comments || "",
      };

      const res = await fetch(api("/api/research"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Submission failed");
      }

      setMessage("✅ Submission saved");
      setSaved(true);
      setTimeout(() => navigate("/dashboard"), 800);
    } catch (e) {
      setMessage(`❌ ${e.message || "Something went wrong"}`);
    }
  }

  return (
    <div>
      <h1 className="title">User Experience Research Form</h1>
      <p className="subtitle">We appreciate your feedback</p>

      <form className="form" onSubmit={handleSubmit(onSubmit)}>
        <div className="grid-2">
          <div>
            <label className="label">First</label>
            <input className={`input ${errors.firstName ? "input-error" : ""}`} {...register("firstName", { required: true })} />
            <div className="help">{errors.firstName && "Required"}</div>
          </div>

          <div>
            <label className="label">Last</label>
            <input className={`input ${errors.lastName ? "input-error" : ""}`} {...register("lastName", { required: true })} />
            <div className="help">{errors.lastName && "Required"}</div>
          </div>
        </div>

        <label className="label">Email</label>
        <input className={`input ${errors.email ? "input-error" : ""}`} {...register("email", { required: true, pattern: /@/ })} />
        <div className="help">{errors.email && "Valid email required"}</div>

        <label className="label">Phone</label>
        <input className="input" placeholder="+1 555 123 4567" {...register("phone")} />

        <label className="label">Address</label>
        <input className="input" placeholder="Street Address" {...register("address1")} />
        <input className="input" placeholder="Street Address Line 2" {...register("address2")} />

        <div className="grid-2">
          <div>
            <label className="label">City</label>
            <input className="input" {...register("city")} />
          </div>
          <div>
            <label className="label">Region</label>
            <input className="input" {...register("region")} />
          </div>
        </div>

        <div className="grid-2">
          <div>
            <label className="label">Postal / Zip Code</label>
            <input className="input" {...register("postal")} />
          </div>
          <div>
            <label className="label">Country</label>
            <input className="input" {...register("country")} />
          </div>
        </div>

        <label className="label">Questions</label>
        <div className="radio-row">
          <label><input type="radio" value={5} {...register("rating")} /> Very Good</label>
          <label><input type="radio" value={4} {...register("rating")} /> Good</label>
          <label><input type="radio" value={3} defaultChecked {...register("rating")} /> Fair</label>
          <label><input type="radio" value={2} {...register("rating")} /> Poor</label>
          <label><input type="radio" value={1} {...register("rating")} /> Very Poor</label>
        </div>

        <label className="label">Comments</label>
        <textarea className="input" rows={4} {...register("comments")} />

        <button className="btn" type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Submitting..." : "Submit"}
        </button>

        {message && <div className="message" role="status">{message}</div>}
        {saved && (
          <div style={{ marginTop: "0.5rem" }}>
            <Link to="/research/list" className="btn">View all submissions</Link>
          </div>
        )}
      </form>
    </div>
  );
}
