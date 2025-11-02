# main.py
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from tortoise.contrib.fastapi import register_tortoise



from settings import settings
from models import User, Profile
from schemas import RegisterIn, UserOut, LoginIn, TokenOut, MeOut, ProfileOut, ChatRequest, ChatResponse
from security import hash_password, verify_password, create_access_token, get_current_username
# for chatbot-----
import asyncio
from typing import List


# forms after login page 
from typing import Literal
from datetime import date
from pathlib import Path
from uuid import uuid4
import re

from fastapi import UploadFile, File, Form



app = FastAPI(title=settings.APP_NAME)

# CORS: allow origins from settings and GitHub Codespaces/dev URLs
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "CORS_ORIGINS", []),
    # allow any github.dev app preview host (http or https)
    allow_origin_regex=r"^https?://.*\.app\.github\.dev$",
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