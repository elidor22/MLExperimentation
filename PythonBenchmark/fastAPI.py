from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.encoders import jsonable_encoder
from regressor import Predictor

pred = Predictor()
class Item(BaseModel):

    val1: float
    val2: float
    val3: float
    val4: float
    val5: float
    val6: float
    val7: float
    val8: float


app = FastAPI()


@app.post("/items/")
async def create_item(item: Item):
    res = pred.predict(item.val1,item.val2,item.val3,item.val4,item.val5,item.val6,item.val7,item.val8)
    #Convert calculated value to float, then create a dict and return it as a normal json response
    #Also round to 6 bit
    res = round(float(res),6)
    dictt = {"Prediction":res}
    return jsonable_encoder(dictt)