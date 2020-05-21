import uvicorn
import pandas as pd
import pickle
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

liner_model = 'models/linear-house-price.pkl'
with open(liner_model, 'rb') as file:
    pickled_linear_model = pickle.load(file)


@app.post("/single_predict/{crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,black,lstat}")
def predict(crim: float, zn: int, indus: float, chas: int, nox: float, rm: float, age: float, dis: float, rad: int,
            tax: int, ptratio: float, black: float, lstat: float):
    labels = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]
    features = [[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]]
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
