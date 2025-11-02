// helper to build API URLs
export function api(path) {
  const base = import.meta.env?.VITE_API_URL;
  if (base && base !== "") {
    // ensure no double slashes
    return `${base.replace(/\/$/, "")}${path}`;
  }
  // when VITE_API_URL is not set, use relative paths so Vite proxy can handle /api
  return path;
}
