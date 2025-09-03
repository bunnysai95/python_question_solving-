# models.py
from tortoise import fields
from tortoise.models import Model

class User(Model):
    id = fields.IntField(pk=True)
    username = fields.CharField(20, unique=True, index=True)
    first_name = fields.CharField(50)
    last_name = fields.CharField(50)
    dob = fields.DateField(null=True)
    phone = fields.CharField(20, null=True)
    password_hash = fields.CharField(128)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    
    class Meta:
        table = "users"

class Profile(Model):
    id = fields.IntField(pk=True)
    user = fields.OneToOneField("models.User", related_name="profile", on_delete=fields.CASCADE)

    first_name = fields.CharField(50)
    last_name  = fields.CharField(50)
    dob        = fields.DateField()
    gender     = fields.CharField(10)                 # "male" | "female"
    phone      = fields.CharField(20, null=True)
    address    = fields.TextField()
    pincode    = fields.CharField(20)
    country    = fields.CharField(80)
    about_me   = fields.TextField()
    file_path  = fields.CharField(255, null=True)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)


    class Meta:
        table = "profiles"
