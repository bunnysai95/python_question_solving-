from tortoise.models import Model
from tortoise import fields
from tortoise.contrib.pydantic import pydantic_model_creator
from decimal import Decimal

# Define the Product and Supplier models
class Product(Model):
    id = fields.IntField(pk = True)
    name = fields.CharField(max_length= 30, null = False)
    quantity_in_stocks= fields.IntField(default = 0 )
    quantity_sold = fields.IntField(default = 0 )
    unit_price = fields.DecimalField(max_digits=10, 
                                     decimal_places=2,
                                     default = (0.00),
                                     )
    revenue = fields.DecimalField(max_digits=20,
                                  decimal_places=2,
                                  default = (0.00),
                                  )
    supplied_by = fields.ForeignKeyField("models.Supplier",
                                          related_name="goods_supplied")

# Define the Supplier model
class Supplier(Model):
    id = fields.IntField(pk =True)
    name = fields.CharField(max_length = 30 )
    company = fields.CharField(max_length = 30 )
    email = fields.CharField(max_length = 100, null = True)
    phone = fields.CharField(max_length = 15, null = True)


# create pydantic models for product
product_pydantic = pydantic_model_creator(Product, name="Product"),
product_pydanticIn = pydantic_model_creator(Product, name= "ProductIn", exclude_readonly=True)

# create pydantic models for Supplier
supplier_pydantic = pydantic_model_creator(Supplier, name="Supplier")
supplier_pydanticIn = pydantic_model_creator(Supplier, name ="SupplierIn",exclude_readonly = True)

