from fastapi import FastAPI, HTTPException
from tortoise.contrib.fastapi import register_tortoise
from models import (supplier_pydantic, supplier_pydanticIn,Supplier,product_pydantic,product_pydanticIn,Product)
from decimal import Decimal
from typing import List
# email
# from fastapi import BackgroundTasks, FastAPI
from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
from pydantic import BaseModel, EmailStr
# from starlette.responses import JSONResponse


#dotenv

from dotenv import load_dotenv
import os 

# credentials
load_dotenv()
EMAIL = os.getenv("EMAIL")
PASS = os.getenv("PASS")
if not EMAIL or not PASS:
    raise ValueError("Missing email or password environment variables")

# adding cors header 
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# adding cors middleware
origins = [
    'http://localhost:3000'
]
# add middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
# this function will show the api documentation
@app.get("/")
def index():
    return {"message": "use api doc /(docs) usage "}

# this function will add a new supplier
@app.post("/supplier")
async def add_supplier(supplier_info: supplier_pydanticIn):
    supplier_obj = await Supplier.create(**supplier_info.dict(exclude_unset= True))
    response = await supplier_pydantic.from_tortoise_orm(supplier_obj)
    return {"status": "ok","data": response}

# will show all suppliers 
@app.get('/supplier')
async def get_all_supplier():
    response = await supplier_pydantic.from_queryset(Supplier.all())
    return {"status": "ok", "data": response}
# this will give suplier with id 
@app.get('/supplier/{supplier_id}')
async def get_supplier(supplier_id: int):
    response = await supplier_pydantic.from_queryset_single(Supplier.get(id=supplier_id))
    return {"status": "ok", "data":response}

# this will update the supplier with id 
@app.put('/supplier/{supplier_id}')
async def update_supplier(supplier_id: int, update_info: supplier_pydanticIn):
    supplier = await Supplier.get(id = supplier_id)
    update_info = update_info.dict(exclude_unset=True)
    supplier.name = update_info['name']
    supplier.company = update_info['company']
    supplier.email = update_info['email']
    supplier.phone = update_info['phone']
    await supplier.save()
    response = await supplier_pydantic.from_tortoise_orm(supplier)
    return {"status": "ok", "data": response}


# this fuction is to delete 
@app.delete('/supplier/{supplier_id}')
async def delete_supplier(supplier_id: int):
    await Supplier.filter(id=supplier_id).delete()
    return {"status": "ok"}

# product info 
@app.post('/product/{supplier_id}')
async def add_product(supplier_id: int , products_details: product_pydanticIn):
    supplier = await Supplier.get(id = supplier_id)
    products_details = products_details.dict(exclude_unset=True)
    products_details['revenue'] += products_details['quantity_sold']*products_details['unit_price']
    product_obj = await Product.create(**products_details,supplied_by= supplier)
    response = await product_pydantic.from_tortoise_orm(product_obj)
    return {"status": "ok", "data": response}


@app.get('/product')
async def all_products():
    response = await product_pydantic.from_queryset(Product.all())
    return {"status": "ok", "data": response}
@app.get('/product/{id}')
async def specific_product(id:int):
    response = await product_pydantic.from_queryset_single(Product.get(id = id))
    return {"status": "ok", "data": response}
@app.put('/product/{id}')
async def update_product(id: int, update_info: product_pydanticIn):
    product = await Product.get(id = id)
    update_info = update_info.dict(exclude_unset=True)
    product.name = update_info['name']
    product.quantity_in_stocks = update_info['quantity_in_stocks']
    product.revenue += update_info['quantity_sold']*update_info['unit_price']
    product.quantity_sold += update_info['quantity_sold']
    product.unit_price = update_info['unit_price']
    await product.save()
    response = await product_pydantic.from_tortoise_orm(product)
    return {"status": "ok", "data": response }

@app.delete('/product/{id}')
async def delete_product(id: int):  
    await Product.filter(id=id).delete()
    return {"status": "ok", "message": f"deleted product with id {id}"}

# email configration ==================
class EmailSchema(BaseModel):
    email: List[EmailStr]

class EmailContent(BaseModel):
    Message: str
    subject: str


conf = ConnectionConfig(
    MAIL_USERNAME = EMAIL,
    MAIL_PASSWORD = PASS,
    MAIL_FROM = EMAIL,
    MAIL_PORT = 465,
    MAIL_SERVER ="smtp.gmail.com",       # <-- correct smtp hsot for gmail
    MAIL_STARTTLS = False,                   
    MAIL_SSL_TLS = True,
    USE_CREDENTIALS = True,
    VALIDATE_CERTS = True,
    # MAIL_FROM_NAME = "Sai Mateti" # optional 
)

@app.post("/email/{product_id}")
async def send_email(product_id: int, content: EmailContent):
    product = await Product.get(id = product_id)
    supplier = await product.supplied_by
    supplier_email = [supplier.email] if supplier and supplier.email else None
    
    if not supplier_email:
        return JSONResponse(status_code=400, content={"message": "Supplier email not found."})


    html = """
    <p>sai mateti business LTD </p>
    <br>
    <p>{content.subject}</p>
    <br>
    <p>{content.Message}</p>
    <br>
    <h6> Best regards </h6>
    <h6> Sai Mateti </h6>

    """
    message = MessageSchema(
    subject= content.subject,
    recipients=supplier_email,
    body=html,
    subtype=MessageType.html)

    fm = FastMail(conf)
    await fm.send_message(message)
    return {"status": "ok"}

             


#  this will register the database tortoise
register_tortoise(
    app, 
    db_url = "sqlite://database.sqlite3",
    modules= {"models": ["models"]},
    generate_schemas = True,
    add_exception_handlers = True
)
