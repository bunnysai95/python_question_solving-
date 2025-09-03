import { useEffect, useMemo, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";

const API_URL = import.meta.env?.VITE_API_URL ?? "http://localhost:8000";

// quick country list
const COUNTRIES = [
  "Afghanistan","Australia","Brazil","Canada","China","France","Germany","India",
  "Indonesia","Italy","Japan","Mexico","Netherlands","New Zealand","Nigeria","Pakistan",
  "Russia","Saudi Arabia","Singapore","South Africa","South Korea","Spain","Sri Lanka",
  "Sweden","Switzerland","Turkey","United Arab Emirates","United Kingdom","United States","Vietnam"
];

// zod helpers
const wordCount = (s) => s.trim().split(/\s+/).filter(Boolean).length;

const ProfileSchema = z.object({
  firstName: z.string().min(2, "Too short").max(50),
  lastName: z.string().min(2, "Too short").max(50),
  dob: z.string().refine(v => !isNaN(new Date(v).getTime()), "Invalid date"),
  gender: z.enum(["male","female"], { required_error: "Select gender" }),
  phone: z.string().regex(/^\+?[0-9()\-\s]{7,20}$/, "Invalid phone").optional().or(z.literal("")),
  address: z.string().min(5, "Too short"),
  pincode: z.string().regex(/^[0-9A-Za-z\- ]{4,10}$/, "Invalid pincode"),
  country: z.string().min(2, "Select a country"),
  aboutMe: z.string().refine(v => wordCount(v) >= 150, "Write at least 150 words"),
  acknowledge: z.literal(true, { errorMap: () => ({ message: "You must acknowledge" }) }),
  file: z.any().refine((f) => f && f.length === 1, "Upload a file")
               .refine((f) => f && f[0]?.size <= 5 * 1024 * 1024, "Max 5MB")
               .refine((f) => {
                  const ok = ["image/png","image/jpeg","application/pdf"];
                  return f && ok.includes(f[0]?.type);
               }, "Only PNG, JPG or PDF"),
});

export default function Profile() {
  const [message, setMessage] = useState("");
  const [wc, setWc] = useState(0);

  const { 
    register, handleSubmit, setValue, watch, 
    formState: { errors, isValid, isSubmitting, isDirty }, 
    reset 
  } = useForm({
    resolver: zodResolver(ProfileSchema),
    mode: "onBlur",
    reValidateMode: "onChange",
    defaultValues: {
      firstName: "", lastName: "", dob: "", gender: undefined,
      phone: "", address: "", pincode: "", country: "",
      aboutMe: "", acknowledge: false, file: undefined
    }
  });

  // word count
  const aboutVal = watch("aboutMe");
  useEffect(() => { setWc(wordCount(aboutVal || "")); }, [aboutVal]);

  async function onSubmit(values) {
    setMessage("");
    try {
      const fd = new FormData();
      fd.append("firstName", values.firstName.trim());
      fd.append("lastName", values.lastName.trim());
      fd.append("dob", values.dob);
      fd.append("gender", values.gender);
      if (values.phone) fd.append("phone", values.phone.trim());
      fd.append("address", values.address.trim());
      fd.append("pincode", values.pincode.trim());
      fd.append("country", values.country);
      fd.append("aboutMe", values.aboutMe.trim());
      fd.append("acknowledge", String(values.acknowledge));
      fd.append("file", values.file[0]);

      const token = localStorage.getItem("access_token");
      const res = await fetch(`${API_URL}/api/profile`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` }, // don't set Content-Type
        body: fd,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to save profile");
      }

      await res.json();
      setMessage("✅ Profile saved!");
      reset({ ...values, file: undefined }); // keep values visible, clear file
    } catch (e) {
      setMessage(`❌ ${e.message}`);
    }
  }

  return (
    <>
      <h1 className="title">Complete your profile</h1>
      <p className="subtitle">Tell us about yourself</p>

      <form className="form" onSubmit={handleSubmit(onSubmit)} noValidate>
        {/* Names */}
        <div className="grid-2">
          <div>
            <label className="label" htmlFor="firstName">First name</label>
            <input id="firstName" className={`input ${errors.firstName ? "input-error":""}`} {...register("firstName")} />
            <div className="help">{errors.firstName?.message ?? " "}</div>
          </div>
          <div>
            <label className="label" htmlFor="lastName">Last name</label>
            <input id="lastName" className={`input ${errors.lastName ? "input-error":""}`} {...register("lastName")} />
            <div className="help">{errors.lastName?.message ?? " "}</div>
          </div>
        </div>

        {/* DOB */}
        <label className="label" htmlFor="dob">Date of birth</label>
        <input id="dob" type="date" className={`input ${errors.dob ? "input-error":""}`} {...register("dob")} />
        <div className="help">{errors.dob?.message ?? " "}</div>

        {/* Gender */}
        <label className="label">Gender</label>
        <div className="btn-group">
          <input type="radio" id="g-m" value="male" {...register("gender")} className="sr-only" />
          <label htmlFor="g-m" className="toggle">Male</label>

          <input type="radio" id="g-f" value="female" {...register("gender")} className="sr-only" />
          <label htmlFor="g-f" className="toggle">Female</label>
        </div>
        <div className="help">{errors.gender?.message ?? " "}</div>

        {/* Phone */}
        <label className="label" htmlFor="phone">Phone</label>
        <input id="phone" inputMode="tel" className={`input ${errors.phone ? "input-error":""}`} placeholder="+1 (555) 123-4567" {...register("phone")} />
        <div className="help">{errors.phone?.message ?? "Optional"}</div>

        {/* Address */}
        <label className="label" htmlFor="address">Address</label>
        <input id="address" className={`input ${errors.address ? "input-error":""}`} {...register("address")} />
        <div className="help">{errors.address?.message ?? " "}</div>

        {/* Pincode / Country */}
        <div className="grid-2">
          {/* Pincode */}
          <div>
            <label className="label" htmlFor="pincode">Pincode</label>
            <input id="pincode" className={`input ${errors.pincode ? "input-error" : ""}`} {...register("pincode")} />
            <div className="help">{errors.pincode?.message ?? " "}</div>
          </div>

          {/* Country dropdown (uses setValue) */}
          <div className="dropdown">
            <label className="label" htmlFor="country">Country</label>
            <input
              id="country"
              className={`input dropdown-input ${errors.country ? "input-error" : ""}`}
              placeholder="Hover to pick…"
              readOnly
              {...register("country")}
              onFocus={(e) => e.currentTarget.parentElement?.classList.add("open")}
              onBlur={(e) => e.currentTarget.parentElement?.classList.remove("open")}
            />
            <ul className="dropdown-list">
              {COUNTRIES.map((c) => (
                <li
                  key={c}
                  onMouseDown={(e) => {
                    e.preventDefault(); // set before blur
                    setValue("country", c, { shouldValidate: true, shouldDirty: true });
                  }}
                >
                  {c}
                </li>
              ))}
            </ul>
            <div className="help">{errors.country?.message ?? "Hover or focus to open, click to select"}</div>
          </div>
        </div>

        {/* Upload */}
        <label className="label" htmlFor="file">Upload file</label>
        <input id="file" type="file" accept="image/png,image/jpeg,application/pdf"
               className={`input ${errors.file ? "input-error":""}`} {...register("file")} />
        <div className="help">{errors.file?.message ?? "PNG, JPG or PDF. Max 5MB."}</div>

        {/* About me */}
        <label className="label" htmlFor="aboutMe">About me</label>
        <textarea id="aboutMe" rows="6" className={`input ${errors.aboutMe ? "input-error":""}`} placeholder="Write at least 150 words…" {...register("aboutMe")} />
        <div className="help">{errors.aboutMe?.message ?? `${wc} / 150+ words`}</div>

        {/* Acknowledge */}
        <label className="label">
          <input type="checkbox" {...register("acknowledge")} /> <span style={{marginLeft:8}}>I acknowledge my info</span>
        </label>
        <div className="help">{errors.acknowledge?.message ?? " "}</div>

        <button className="btn" type="submit" disabled={!isDirty || !isValid || isSubmitting} aria-busy={isSubmitting}>
          {isSubmitting ? "Saving…" : "Submit"}
        </button>

        {message && <div className="message" role="status">{message}</div>}
      </form>
    </>
  );
}
