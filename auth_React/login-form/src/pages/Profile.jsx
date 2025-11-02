import { useEffect, useMemo, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Link, useNavigate } from "react-router-dom";

import { api } from "../api";

// Very short list just for demo — extend as you like.
const COUNTRIES = [
  "Afghanistan","Australia","Brazil","Canada","China","France","Germany",
  "India","Indonesia","Italy","Japan","Mexico","Netherlands","New Zealand",
  "Nigeria","Pakistan","Russia","United Kingdom","United States"
];

function wordCount(text) {
  return (text || "")
    .trim()
    .replace(/\s+/g, " ")
    .split(" ")
    .filter(Boolean).length;
}

// ----- Zod schema -----
const ProfileSchema = z.object({
  firstName: z.string().min(2, "Too short").max(50),
  lastName: z.string().min(2, "Too short").max(50),
  dob: z.string().refine(v => {
    const d = new Date(v);
    return !isNaN(d.getTime());
  }, "Invalid date"),
  gender: z.enum(["male", "female"], { required_error: "Select gender" }),
  phone: z.string()
    .regex(/^\+?[0-9()\-\s]{7,20}$/, "Invalid phone")
    .optional()
    .or(z.literal("")), // allow blank
  address: z.string().min(5, "Too short"),
  pincode: z.string().regex(/^[0-9A-Za-z\- ]{4,10}$/, "Invalid pincode"),
  country: z.string().min(2, "Select a country"),
  aboutMe: z.string().refine(v => wordCount(v) >= 150, "Write at least 150 words"),
  acknowledge: z.literal(true, {
    errorMap: () => ({ message: "You must acknowledge" }),
  }),
  // optional file (0 or 1), max 5MB, only png/jpg/pdf if present
  file: z
    .any()
    .optional()
    .refine((f) => !f || f.length <= 1, "Upload one file")
    .refine(
      (f) => !f || !f[0] || f[0].size <= 5 * 1024 * 1024,
      "Max 5MB"
    )
    .refine(
      (f) =>
        !f ||
        !f[0] ||
        ["image/png", "image/jpeg", "application/pdf"].includes(f[0].type),
      "PNG, JPG or PDF only"
    ),
});

export default function Profile() {
  const navigate = useNavigate();
  const [message, setMessage] = useState("");
  const [openCountries, setOpenCountries] = useState(false);

  const {
    register,
    handleSubmit,
    setValue,
    watch,
    formState: { errors, isSubmitting, isValid, isDirty },
  } = useForm({
    resolver: zodResolver(ProfileSchema),
    mode: "onBlur",
    reValidateMode: "onChange",
    defaultValues: {
      firstName: "",
      lastName: "",
      dob: "",
      gender: undefined,
      phone: "",
      address: "",
      pincode: "",
      country: "",
      aboutMe: "",
      acknowledge: false,
      file: undefined,
    },
  });

  // show live word count
  const aboutMe = watch("aboutMe");
  const wc = useMemo(() => wordCount(aboutMe), [aboutMe]);

  // submit
  async function onSubmit(values) {
    setMessage("");
    try {
      const token = localStorage.getItem("access_token");
      if (!token) {
        setMessage("❌ Not logged in");
        return;
      }

      const fd = new FormData();
      fd.append("firstName", values.firstName);
      fd.append("lastName", values.lastName);
      fd.append("dob", values.dob);
      fd.append("gender", values.gender);
      fd.append("phone", values.phone || "");
      fd.append("address", values.address);
      fd.append("pincode", values.pincode);
      fd.append("country", values.country);
      fd.append("aboutMe", values.aboutMe);
      fd.append("acknowledge", String(values.acknowledge));
      if (values.file && values.file.length === 1) {
        fd.append("file", values.file[0]);
      }

  const res = await fetch(api("/api/profile"), {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: fd,
      });

      if (!res.ok) {
        // Try to decode backend error
        const err = await res.json().catch(() => ({}));
        const detail =
          typeof err?.detail === "string"
            ? err.detail
            : err?.detail
            ? JSON.stringify(err.detail)
            : "Submission failed";
        throw new Error(detail);
      }

      setMessage("✅ Profile saved!");
      // Go back to dashboard
      setTimeout(() => navigate("/dashboard"), 700);
    } catch (e) {
      setMessage(`❌ ${e.message || "Something went wrong"}`);
    }
  }

  return (
    <>
      <h1 className="title">Complete your profile</h1>
      <p className="subtitle">Tell us about yourself</p>

      <form className="form" onSubmit={handleSubmit(onSubmit)} noValidate>
        {/* First / Last name */}
        <div className="grid-2">
          <div>
            <label className="label" htmlFor="firstName">First name</label>
            <input
              id="firstName"
              autoComplete="given-name"
              className={`input ${errors.firstName ? "input-error" : ""}`}
              {...register("firstName")}
            />
            <div className="help">{errors.firstName?.message ?? " "}</div>
          </div>

          <div>
            <label className="label" htmlFor="lastName">Last name</label>
            <input
              id="lastName"
              autoComplete="family-name"
              className={`input ${errors.lastName ? "input-error" : ""}`}
              {...register("lastName")}
            />
            <div className="help">{errors.lastName?.message ?? " "}</div>
          </div>
        </div>

        {/* DOB */}
        <label className="label" htmlFor="dob">Date of birth</label>
        <input
          id="dob"
          type="date"
          autoComplete="bday"
          className={`input ${errors.dob ? "input-error" : ""}`}
          {...register("dob")}
        />
        <div className="help">{errors.dob?.message ?? " "}</div>

        {/* Gender */}
        <fieldset className={`fieldset ${errors.gender ? "pw-error" : ""}`}>
          <legend className="label">Gender</legend>
          <div className="radio-row">
            <input id="gender-male" type="radio" value="male" {...register("gender")} />
            <label htmlFor="gender-male">Male</label>
            <input id="gender-female" type="radio" value="female" {...register("gender")} />
            <label htmlFor="gender-female">Female</label>
          </div>
          <div className="help">{errors.gender?.message ?? " "}</div>
        </fieldset>

        {/* Phone */}
        <label className="label" htmlFor="phone">Phone</label>
        <input
          id="phone"
          autoComplete="tel"
          placeholder="+1 555 123 4567"
          className={`input ${errors.phone ? "input-error" : ""}`}
          {...register("phone")}
        />
        <div className="help">{errors.phone?.message ?? " "}</div>

        {/* Address */}
        <label className="label" htmlFor="address">Address</label>
        <input
          id="address"
          autoComplete="street-address"
          className={`input ${errors.address ? "input-error" : ""}`}
          {...register("address")}
        />
        <div className="help">{errors.address?.message ?? " "}</div>

        {/* Pincode / Country */}
        <div className="grid-2">
          <div>
            <label className="label" htmlFor="pincode">Pincode</label>
            <input
              id="pincode"
              autoComplete="postal-code"
              className={`input ${errors.pincode ? "input-error" : ""}`}
              {...register("pincode")}
            />
            <div className="help">{errors.pincode?.message ?? " "}</div>
          </div>

          <div>
            <label className="label" htmlFor="country">Country</label>
            <div
              className="country-wrap"
              onMouseEnter={() => setOpenCountries(true)}
              onMouseLeave={() => setOpenCountries(false)}
            >
              <input
                id="country"                                  // ✅ matches label
                type="text"
                autoComplete="country-name"                   // ✅ accessibility hint
                placeholder="Hover or focus to open, click to select"
                className={`input ${errors.country ? "input-error" : ""}`}
                onFocus={() => setOpenCountries(true)}
                {...register("country")}
              />
              {openCountries && (
                <div className="country-popover">
                  <ul>
                    {COUNTRIES.map((c) => (
                      <li
                        key={c}
                        onMouseDown={(e) => e.preventDefault()} // keep focus
                        onClick={() => {
                          setValue("country", c, { shouldValidate: true });
                          setOpenCountries(false);
                        }}
                      >
                        {c}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            <div className="help">{errors.country?.message ?? " "}</div>
          </div>
        </div>

        {/* File (optional) */}
        <label className="label" htmlFor="file">Upload file</label>
        <input
          id="file"                                          // ✅ matches label
          type="file"
          className={`input ${errors.file ? "input-error" : ""}`}
          accept=".png,.jpg,.jpeg,.pdf"
          {...register("file")}
        />
        <div className="help">{errors.file?.message ?? "PNG, JPG or PDF. Max 5MB."}</div>

        {/* About me */}
        <label className="label" htmlFor="aboutMe">About me</label>
        <textarea
          id="aboutMe"                                       // ✅ matches label
          autoComplete="off"
          rows={6}
          className={`input ${errors.aboutMe ? "input-error" : ""}`}
          placeholder="Write at least 150 words"
          {...register("aboutMe")}
        />
        <div className="help">
          {errors.aboutMe?.message ?? `${wc} / 150+ words`}
        </div>

        {/* Acknowledge */}
        <div className="checkbox-row">
          <input id="acknowledge" type="checkbox" {...register("acknowledge")} />
          <label htmlFor="acknowledge">I acknowledge my info</label>
        </div>
        <div className="help">{errors.acknowledge?.message ?? " "}</div>

        <button
          type="submit"
          className="btn"
          disabled={!isDirty || !isValid || isSubmitting}
          aria-busy={isSubmitting}
        >
          {isSubmitting ? "Submitting..." : "Submit"}
        </button>

        {message && <div className="message" role="status">{message}</div>}
      </form>
    </>
  );
}
