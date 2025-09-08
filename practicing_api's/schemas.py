from datetime import date
from pydantic import BaseModel, Field, ConfigDict, model_validator

# Register 
class RegisterIn(BaseModel):
    firstName: str = Field(min_length= 2, max_length = 50)
    lastName: str = Field(min_length =2 , max_length = 50)
    dob: date 
    username: str = Field(min_length = 3 , max_length = 20, pattern = r"^[a-zA-Z0-9_]+$")
    password: str  =Field(min_length = 8)
    confirmPassword: str
    phone: str |None = Field(default = None, pattern = r"^/+?[0-9()\-\s]{7,20}$")

    @model_validator(model= "after")
    def check_password(self):
        if self.password != self.confirmPassword:
            raise ValueError("passwords do not match")
        return self

class UserOut(BaseModel):
    pass
# login
class LoginIn(BaseModel):
    pass
class TokenOut(BaseModel):
    pass
# me
class MeOut(BaseModel):
    pass
class ProfileOut(BaseModel):
    pass
