# main.py
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from tortoise.contrib.fastapi import register_tortoise

from settings import settings
from models import User
from schemas import RegisterIn, UserOut, LoginIn, TokenOut, MeOut
from security import hash_password, verify_password, create_access_token, get_current_username

app = FastAPI(title=settings.APP_NAME)

# CORS so Vite (http://localhost:5173) can call us during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^https://.*\.github\.dev$",
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
