from tortoise import fields
from tortoise.models import Model

class User(Model):
    id = fields.IntField(pk=True)
    username = fields.CharField(20, unique = True, index = True)
    first_name = fields.CharField(50)
    last_name = fields.CharField(50)
    dob = fields.DateField(null= True)
    phone = fields.CharField(20, null = True)
    create_at = fields.DatetimeField(auto_now_add = True)
    update_at = filess.DatetimeField(auto_now = True)

    class meta:
        table = "user"

class Profile(Modle):
    id = fields.IntField(pk = True)
    user = fields.OneToOneField("models.User", related_name = "profile", on_delete = fileds.CASCADE)
    first_name = fields.CharField(50)
    last_name = fields.CharField(50)
    dob = fields.DateField(50)
    gender = fields.charField(20, null= True)
    address = fields.TextField()
    pincode = fields.charField()
    country = fields.cahtField(20)
    about_me = fields.TextField()
    file_path = fields.CahrtField(255, null = True)

    create_at = fields.DatetimeField(auto_now_add = True)
    updated_at = fields.DatetimeField(auto_now = True)

    class meta:
        table = "profiles"

