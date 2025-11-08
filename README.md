# Auth React / FastAPI - Dev & Docker

This repository contains a small React (Vite) frontend and a FastAPI backend. The project includes dev and production Docker configurations.

Quick dev run (uses Vite proxy):

1. Start dev stack (frontend runs Vite, backend runs uvicorn):

```bash
docker-compose up --build
```

Access the app at http://localhost:5173 and the backend at http://localhost:8000.

Notes:
- Frontend dev server proxies `/api` to the backend using `vite.config.js` so you can use relative paths like `fetch('/api/login')`.
- If you want to use the backend directly, set `VITE_API_URL` in `auth_React/login-form/.env`.

Production build with nginx:

1. Build and run the production stack (serves built static files via nginx on port 80):

```bash
docker-compose -f docker-compose.prod.yml up --build
```

Access the app at http://localhost and the backend at http://localhost:8000 (proxied by nginx for `/api`).

Files added:
- `auth_React/fast_api/requirements.txt` - backend requirements
- `auth_React/fast_api/Dockerfile` - backend Dockerfile (dev-friendly)
- `auth_React/login-form/Dockerfile` - frontend Dockerfile (dev)
- `auth_React/login-form/Dockerfile.prod` - frontend production build
- `auth_React/login-form/nginx.conf` - nginx config to serve static + proxy `/api`
- `auth_React/login-form/vite.config.js` - Vite proxy for dev
- `docker-compose.yml` - dev compose (Vite + uvicorn)
- `docker-compose.prod.yml` - production compose (nginx-bundled frontend + uvicorn)

If you want, I can:
- Start the dev compose and verify endpoints (curl /health and open homepage).
- Start the production compose and verify the nginx-served frontend and proxied API.
- Convert the frontend to always use relative paths and remove any remaining absolute URLs.

## Environment files and notes

- Frontend env: `auth_React/login-form/.env`
	- `VITE_API_URL` controls the base URL used by the frontend when constructing API calls.
	- If `VITE_API_URL` is empty (the default in `.env`), the frontend will make relative requests like `/api/login` which the Vite dev server proxies to the backend (recommended for local development).
	- To call the backend directly (for example when running the frontend outside of the Vite proxy), set `VITE_API_URL` to the backend URL, e.g. `http://localhost:8000`.

- Backend CORS: the backend must allow the origin used by the browser. When using Codespaces or GitHub dev previews, the preview origin (e.g. `https://...app.github.dev`) must be allowed in the backend CORS settings.

Backend tip: the backend will read `auth_React/fast_api/.env` (copy from `.env.example`).
- To enable permissive Codespaces preview origins for development, set `ALLOW_CODESPACES=true` in the backend `.env`. This will allow origins that match `*.app.github.dev` for quick previews. Keep this disabled in production and prefer explicit `CORS_ORIGINS` values.

Quick tip: If you change `.env`, restart the Vite dev server or backend so they pick up the new values.
# python_question_solving-