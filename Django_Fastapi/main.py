from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    text: str
    is_done: bool = False
    

items = [ ]

@app.get("/") # defines the root endpoint path
def root():
    return {"Hello": "this build with FastApi"}

@app.post("/items")
def create_item(item: Item):
    items.append(item)
    return items
@app.get("/items", response_model=list[Item])
def list_items(limit: int= 10):
    return items[0:limit]

@app.get("/items/{item_id}", response_model=Item)
def read_items(item_id: int = None) -> Item:
    if item_id is None or item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail="ITEM not found")