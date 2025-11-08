# main.py
from fastapi import FastAPI, HTTPException, status, Depends
import logging
from fastapi.middleware.cors import CORSMiddleware
from tortoise.contrib.fastapi import register_tortoise



from settings import settings
from models import User, Profile, TaskStatus
from schemas import (
    RegisterIn,
    UserOut,
    LoginIn,
    TokenOut,
    MeOut,
    ProfileOut,
    ChatRequest,
    ChatResponse,
    UpdateMeIn,
    ChangePasswordIn,
)
from security import hash_password, verify_password, create_access_token, get_current_username
# for chatbot-----
import asyncio
from typing import List


# forms after login page 
from typing import Literal
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4
import re

from fastapi import UploadFile, File, Form



app = FastAPI(title=settings.APP_NAME)

# Basic logging configuration for the app
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("app")

# Register global exception handlers (validation, HTTP errors, and uncaught exceptions)
from errors import http_exception_handler, validation_exception_handler, generic_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from middleware import RequestLoggingMiddleware

app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# request/response logging middleware
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
app.add_middleware(RequestLoggingMiddleware)

# CORS: allow origins from settings and GitHub Codespaces/dev URLs
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "CORS_ORIGINS", []),
    # allow any github.dev app preview host (http or https) only when enabled in settings
    allow_origin_regex=(r"^https?://.*\.app\.github\.dev$" if getattr(settings, "ALLOW_CODESPACES", False) else None),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ROUTES ---

@app.post("/api/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterIn):
    exists = await User.filter(username=payload.username).exists()
    if exists:
        raise HTTPException(status_code=409, detail="Username already taken")

    user = await User.create(
        username=payload.username,
        first_name=payload.firstName,
        last_name=payload.lastName,
        dob=payload.dob,
        phone=payload.phone,
        password_hash=hash_password(payload.password),
    )
    # Map to UserOut fields
    return UserOut(
        id=user.id,
        username=user.username,
        firstName=user.first_name,
        lastName=user.last_name,
        dob=user.dob,
        phone=user.phone,
    )

@app.post("/api/login", response_model=TokenOut)
async def login(payload: LoginIn):
    user = await User.get_or_none(username=payload.username)
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(sub=user.username)
    return TokenOut(access_token=token)

@app.get("/api/me", response_model=MeOut)
async def me(current_username: str = Depends(get_current_username)):
    user = await User.get_or_none(username=current_username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return MeOut(username=user.username, firstName=user.first_name, lastName=user.last_name)


@app.patch("/api/me", response_model=MeOut)
async def update_me(payload: UpdateMeIn, current_username: str = Depends(get_current_username)):
    user = await User.get_or_none(username=current_username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # apply updates if provided
    if payload.firstName is not None:
        user.first_name = payload.firstName
    if payload.lastName is not None:
        user.last_name = payload.lastName
    if getattr(payload, "phone", None) is not None:
        user.phone = payload.phone

    await user.save()
    return MeOut(username=user.username, firstName=user.first_name, lastName=user.last_name)


@app.post("/api/change-password")
async def change_password(payload: ChangePasswordIn, current_username: str = Depends(get_current_username)):
    user = await User.get_or_none(username=current_username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # verify current password
    if not verify_password(payload.currentPassword, user.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    # basic new password validation
    if len(payload.newPassword) < 8:
        raise HTTPException(status_code=422, detail="New password must be at least 8 characters long")

    user.password_hash = hash_password(payload.newPassword)
    await user.save()
    return {"ok": True, "message": "Password updated"}


# --- ETL endpoints: queue a background task and query status ---
from celery_app import celery_app
from uuid import uuid4


@app.post("/api/etl/upload")
async def upload_etl(file: UploadFile = File(...), current_username: str = Depends(get_current_username)):
    """Accept a CSV file upload (multipart/form-data) and enqueue a worker task to process it.
    The endpoint immediately returns a task_id (TaskStatus record) that can be polled.
    """
    # basic content-type check (allow text/csv and common fallbacks)
    allowed = {"text/csv", "application/csv", "text/plain"}
    # some clients may not set content_type for .csv, so we don't strictly enforce it

    # persist uploaded file to disk in uploads dir
    ext = Path(file.filename).suffix
    fname = f"{uuid4().hex}_{_safe_filename(file.filename)}"
    dest = UPLOADS_DIR / fname
    # stream write to avoid loading entire file to memory
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        await file.close()

    # create task status
    t = await TaskStatus.create(name="etl_upload", status="pending")

    # enqueue celery task (worker will initialize DB and update TaskStatus)
    try:
        # import here to avoid circular imports at module import time
        from celery_tasks import process_etl_file
        process_etl_file.apply_async(args=[t.id, str(dest)])
    except Exception as exc:
        # mark as failed and return 500
        t.status = "failed"
        t.error = f"enqueue_failed: {str(exc)}"
        t.finished_at = datetime.utcnow()
        await t.save()
        raise HTTPException(status_code=500, detail="Failed to enqueue ETL task")

    return {"status": "accepted", "task_id": t.id}


@app.get("/api/etl/tasks")
async def list_tasks(limit: int = 50):
    """Return most recent TaskStatus records (simple list endpoint used by Task History UI)."""
    q = TaskStatus.all().order_by("-id")
    if limit:
        q = q.limit(limit)
    tasks = await q
    out = []
    for t in tasks:
        out.append({
            "id": t.id,
            "name": t.name,
            "status": t.status,
            "started_at": t.started_at,
            "finished_at": t.finished_at,
            "error": t.error,
        })
    return out


@app.get("/api/etl/tasks/{task_id}")
async def get_task_status(task_id: int):
    t = await TaskStatus.get_or_none(id=task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "id": t.id,
        "name": t.name,
        "status": t.status,
        "started_at": t.started_at,
        "finished_at": t.finished_at,
        "error": t.error,
    }


# debug endpoint to trigger an uncaught exception (useful for testing error handler)
@app.get("/__test/raise")
def _raise():
    raise RuntimeError("test-exception")
@app.get("/health")
def health():
    return {"ok": True}

# --- DB INIT ---
register_tortoise(
    app,
    db_url=settings.DB_URL,
    modules={"models": ["models"]},
    generate_schemas=True,   # creates tables on first run
    add_exception_handlers=True,
)

def _safe_filename(name: str) -> str:
    # simple sanitizer: keep letters, numbers, dot, dash, underscore
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

@app.post("/api/profile", response_model=ProfileOut)
async def create_profile(
    firstName: str = Form(...),
    lastName: str = Form(...),
    dob: date = Form(...),
    gender: Literal["male","female"] = Form(...),
    phone: str | None = Form(default=None),
    address: str = Form(...),
    pincode: str = Form(...),
    country: str = Form(...),
    aboutMe: str = Form(...),
    acknowledge: bool = Form(...),
    file: UploadFile = File(...),
    current_username: str = Depends(get_current_username),
):
    # server-side validation
    words = len([w for w in aboutMe.strip().split() if w])
    if words < 150:
        raise HTTPException(status_code=422, detail="About me must be at least 150 words")
    if not acknowledge:
        raise HTTPException(status_code=422, detail="You must acknowledge your info")

    # save file
    ext = Path(file.filename).suffix
    fname = f"{uuid4().hex}_{_safe_filename(file.filename)}"
    dest = UPLOADS_DIR / fname
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    # upsert (create or update) the profile for this user
    user = await User.get_or_none(username=current_username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    from models import Profile
    existing = await Profile.get_or_none(user=user)
    if existing:
        p = existing
        p.first_name = firstName
        p.last_name  = lastName
        p.dob        = dob
        p.gender     = gender
        p.phone      = phone
        p.address    = address
        p.pincode    = pincode
        p.country    = country
        p.about_me   = aboutMe
        p.file_path  = str(dest)
        await p.save()
    else:
        p = await Profile.create(
            user=user,
            first_name=firstName, last_name=lastName, dob=dob, gender=gender,
            phone=phone, address=address, pincode=pincode, country=country,
            about_me=aboutMe, file_path=str(dest)
        )

    return ProfileOut(
        id=p.id,
        username=user.username,
        firstName=p.first_name,
        lastName=p.last_name,
        dob=p.dob,
        gender=p.gender,
        phone=p.phone,
        address=p.address,
        pincode=p.pincode,
        country=p.country,
        aboutMe=p.about_me,
        filePath=p.file_path,
    )


# ---- Chat provider (echo fallback or Groq) ----
async def generate_reply(messages: List[dict]) -> str:
    prov = settings.CHAT_PROVIDER.lower()
    if prov == "groq" and settings.GROQ_API_KEY:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        def _call():
            return client.chat.completions.create(
                model=settings.GROQ_MODEL, messages=messages, temperature=0.7
            ).choices[0].message.content
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _call)

    # fallback echo
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return "Echo: " + last_user

@app.post("/api/chat", response_model=ChatResponse)   # ⬅️ THIS is the route your UI calls
async def chat(req: ChatRequest, current_username: str = Depends(get_current_username)):
    print("Using provider:", settings.CHAT_PROVIDER)   # optional debug
    full_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    full_messages += [m.model_dump() for m in req.messages]
    reply = await generate_reply(full_messages)
    return ChatResponse(reply=reply)