import uvicorn
import pandas as pd
import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field

liner_model = 'models/linear-house-price.pkl'
with open(liner_model, 'rb') as f:
    pickled_linear_model = pickle.load(f)

# Initialize the Fast API APP
app = FastAPI(
    title="House Price Prediction",
    description="A Sample demo application to predict the house price",
    version="0.1"
)

# Enabling CORS to allow all IP to hit while developing and integrating
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class jsonRequest(BaseModel):
    crim    : float = Field(...)
    zn      : int   = Field(...)
    indus   : float = Field(...)
    chas    : int   = Field(...)
    nox     : int   = Field(...)
    rm      : float = Field(...)
    age     : float = Field(...)
    dis     : float = Field(...)
    rad     : int   = Field(...)
    tax     : int   = Field(...)
    ptratio : float = Field(...)
    blask   : float = Field(...)
    lstat   : float = Field(...)

@app.post("/single_predict")
def predict(item:jsonRequest):
    labels = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]
    features = [[item.crim, item.zn, item.indus, item.chas, item.nox, item.rm, item.age,item.dis, item.rad, item.tax, item.ptratio, item.black, item.lstat]]
    to_predict = pd.DataFrame(features, columns=labels)
    prediction = pickled_linear_model.predict(to_predict)
    return {"predicted value:": int(prediction)}


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    dataframe = pd.read_csv(file.file)
    prediction = pickled_linear_model.predict(dataframe)
    print(prediction)
    return {"prediction": str(prediction)}


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000,reload=True)
