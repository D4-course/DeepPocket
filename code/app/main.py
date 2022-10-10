from typing import Union
from fastapi import FastAPI
from __init__ import *

app = FastAPI()

@app.get("/")
def read_roo0t():
    #return printf(4)
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
