from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel 

from fastapi.encoders import jsonable_encoder
from regressor import Predictor

pred = Predictor()
class Item(BaseModel):

    input: list


app = FastAPI()


@app.post("/predictRisk/")
async def create_item(item: Item):
    #res = pred.predict(item.val1,item.val2,item.val3,item.val4,item.val5,item.val6,item.val7,item.val8)
    #Convert calculated value to float, then create a dict and return it as a normal json response
    #Also round to 6 bit
    res = pred.predict(item.input)
    #res = 5.123
    res = res
    dictt = {"Predictions":res}
    return dictt
