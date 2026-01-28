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


class TaskStatus(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(100)
    status = fields.CharField(20, default="pending")  # pending | working | success | failed
    started_at = fields.DatetimeField(null=True)
    finished_at = fields.DatetimeField(null=True)
    error = fields.TextField(null=True)

    class Meta:
        table = "task_status"


class Research(Model):
    id = fields.IntField(pk=True)
    user = fields.ForeignKeyField("models.User", related_name="researches", null=True, on_delete=fields.SET_NULL)
    first_name = fields.CharField(50)
    last_name = fields.CharField(50)
    email = fields.CharField(200)
    phone = fields.CharField(50, null=True)
    address1 = fields.TextField(null=True)
    address2 = fields.TextField(null=True)
    city = fields.CharField(100, null=True)
    region = fields.CharField(100, null=True)
    postal = fields.CharField(50, null=True)
    country = fields.CharField(100, null=True)
    rating = fields.IntField(default=3)
    comments = fields.TextField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "research"
