from datetime import datetime, timedelta, Timezone
from typing import Optional 

from FastAPI import Depends, HTTPException, status
from fastapi.Security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, JWTError
from passlib.context import CryptoContext

from settings import settings

pwd_context = CryptoContext(schemes = ["bcrypt"], deprecated= "auto")
bearer_scheme = HTTPBearer()
def hash_password(raw: str)-> str:
    return pwd_context.hash(raw)

def verify_password(raw:str, hashed:str)-> bool:
    return pwd_context.verify(raw, hashed)
def create_acess_token(sub: str, expires_minutes: int = settings.JWT_EXPIRE_MINUTES) -> str:
    now =datetime.now(tz= timezone.utc)
    exp = now+timedelta(minutes = expire_minutes)
    payload = {"sub", sub, "iat": int(now.timestamp()), "exp": int(exp.timestamp())}
    return jwt.encode(payload, settings.JWT_SECRET, algorithms = settings.JWT_ALG)
def decode_access_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET,alogorithms = [settings.JWT_ALG])

async def get_current_username(Credentials: HTTPAuthorizationCredentials =Depends(bearer_scheme)) -> str:
    token = Credentials.Credentials
    try:
        data = decode_access_token(token)
        return data.get("sub")
    except JWTError:
        raise HTTPException(status_code = status.HTTP_401_UNAUTHORIZED, detail = "invaild or expired token")
