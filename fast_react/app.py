from fastapi import FastAPI
from tortoise.contrib.fastapi import register_tortoise
from models import (supplier_pydantic, supplier_pydanticIn,Supplier)

app = FastAPI()
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









#  this will register the database
register_tortoise(
    app, 
    db_url = "sqlite://database.sqlite3",
    modules= {"models": ["models"]},
    generate_schemas = True,
    add_exception_handlers = True
)
