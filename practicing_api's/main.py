from fastapi import FastAPI #, HTTPException, status,Depands
from fastapi.middleware.cors import CORSMiddleware
from tortoise.contrib.fastapi import register_tortoise


from settings import Settings
from models import User
from security import hash_password, verify_password, create_access_token, get_current_username
# from security import hash_password, verify_password, create_access_token, decode_access_token, get_current_username
from schemas import RegisterIn, UserOut, LoginIn, TokenOut, MeOut, ProfileOut
# from schemas import RegisterIn,UserOut, LoginIn, TokenOut,MeOut,ProfileOut

# forms after login page

from typing import Literal
from datatime import date
from pathlib import Path
from uuid import uuid4
import re

from fastapi import UploadFile, File, Form

app = FastAPI(title = settings.APP_NAME)

# CORS so Vite url calling 
app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex = r"^https://.*\.github\.dev$",
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# routes
@app.post("/api/register", response_model = UserOut, status_code = status.HTTP_201_CREATED)
async def register():
    exists = await User.filter(username = payload.username).exists()
    if exists:
        raise HTTPException(status_code = 409, detail = "username already taken")    
    user = await User.created(
        username = payload.username,
        first_name = payload.firstname,
        last_name = payload.lastname,
        dob = payload.dob,
        phone = payload.phone,
        password_hash = hash_password(payload.password),
    )

    return UserOut(
        id=user.id,
        username=user.username,
        firstName=user.first_name,
        lastName=user.last_name,
        dob=user.dob,
        phone=user.phone,
    )