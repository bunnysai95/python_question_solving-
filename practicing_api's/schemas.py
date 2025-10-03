from datetime import datetime
from pydantic import Basemodel, Field, ConfigDict, model_validaor

from typing import List, Literal 


class RegisterIn(BaseModel):
    firstName: str = Field(min_length = 2, max_length = 50)
    lastName: str = Field(min_length = 2, max_length = 50 )
    dob: date
    username :str = Field(min_length = 3 , max_length = 50 , pattern = r"^[a-zA-Z0-9_]+$")
    password: str = Field(min_length = 8)
    confirmpassword :str
    phone : str | None = Field(default = None, pattern = r"^\+?[0-9()\-\s]{7,20}$")

    @model_validator(mode = "after")
    def check_password(self):
        if self.password != self.confirmPassword:
            rasie ValueError("password do not match ")
        retun self

class UserOut(BaseModel):
    id: int
    username: str
    firstName: str
    lastName : str
    dob: date | None = None
    phone :str = None = None
    model_config = ConfigDict(from_attributes = True)

class LoginIn(BaseModel):
    username: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type : str

class Meout(BaseModel):
    username: str
    firstname: str
    lastname : str

class ProfileOut(BaseModel):
    id: int 
    username : str
    firstname: str
    lastname: str
    dob: date
    gender : str
    phone : str|None= None
    address : str 
    pincode :str
    country : str
    aboutme: str
    filepath : str| None = None
    model_config = ConfigDict(from_attributes = True)

class ChatMessage(BaseModel):
    role: Literal["system", "user","assistant"]
    content : str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    replay: str