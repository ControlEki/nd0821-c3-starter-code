"""API for Censusu data prediction."""

import sys

sys.path.append("./ml")

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from model import api_inference


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        ]
    })

app = FastAPI(
    title="Census Bureau Data API",
    description="Predicts salary category of population.",
    version="1.0.0",
    openapi_version="3.1.0",
)

@app.get("/")
async def print_greeting():
    return {"message": "Welcome to our salary category prodediction API."}

@app.post("/predict")
async def make_prediction(inference_data: InputData):
    
    # Run inference on the processed data
    prediction = api_inference(inference_data)
    print(prediction)
    return {"prediction": prediction}
