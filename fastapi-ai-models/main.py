from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str

@app.post("/predict/")
async def predict(item: Item):
    prediction = predict_sentiment(item.description)
    return {
            "name": item.name, 
            "description": item.description, 
            "predicted_label": prediction
            } 

def predict_sentiment(text):
    return "positive" if text.lower().count("good") > text.lower().count("bad") else "negative"


