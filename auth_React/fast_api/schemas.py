# schemas.py
from datetime import date
from pydantic import BaseModel, Field, ConfigDict, model_validator
#  chatbot -----import list
from typing import List,Literal

# ---- REGISTER ----
class RegisterIn(BaseModel):
    firstName: str = Field(min_length=2, max_length=50)
    lastName: str = Field(min_length=2, max_length=50)
    dob: date
    username: str = Field(min_length=3, max_length=20, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(min_length=8)
    confirmPassword: str
    phone: str | None = Field(default=None, pattern=r"^\+?[0-9()\-\s]{7,20}$")

    @model_validator(mode="after")
    def check_passwords(self):
        if self.password != self.confirmPassword:
            raise ValueError("Passwords do not match")
        return self

class UserOut(BaseModel):
    id: int
    username: str
    firstName: str
    lastName: str
    dob: date | None = None
    phone: str | None = None

    model_config = ConfigDict(from_attributes=True)

# ---- LOGIN ----
class LoginIn(BaseModel):
    username: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

# ---- /me ----
class MeOut(BaseModel):
    username: str
    firstName: str
    lastName: str

class ProfileOut(BaseModel):
    id: int
    username: str
    firstName: str
    lastName: str
    dob: date
    gender: str
    phone: str | None = None
    address: str
    pincode: str
    country: str
    aboutMe: str
    filePath: str | None = None
    model_config = ConfigDict(from_attributes=True)

#  ------- chatbot schemas -----

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str