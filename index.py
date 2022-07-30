from operator import index
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin
import pickle
import os
import pandas as pd

basedir = os.path.abspath(os.path.dirname(__file__))


app = Flask(__name__)

CORS_ALLOW_ORIGIN = "*,*"
CORS_EXPOSE_HEADERS = "*,*"
CORS_ALLOW_HEADERS = "content-type,*"
cors = CORS(
    app,
    origins=CORS_ALLOW_ORIGIN.split(","),
    allow_headers=CORS_ALLOW_HEADERS.split(","),
    expose_headers=CORS_EXPOSE_HEADERS.split(","),
    supports_credentials=True,
)


file1 = open(os.path.join(basedir,"models_anxiety.pkl"), "rb")
anxiety = pickle.load(file1)
file1.close()

file1 = open(os.path.join(basedir,"models_depression.pkl"), "rb")
depression = pickle.load(file1)
file1.close()

file1 = open(os.path.join(basedir,"models_stress.pkl"), "rb")
stress = pickle.load(file1)
file1.close()


@app.route("/")
@cross_origin()
def home():
    return "hello"


@app.route("/predict", methods=["POST", "GET"])
def predict():

    def predict(model, dataframe):
        result = model.predict_proba(dataframe)
        return result[0][1]

    if request.method == "POST":
        health_input = request.get_json(force=True)
        
        df = pd.DataFrame(health_input,index=[0])
        df = df[['Age', 'Female', 'Male', 'BCS', 'BIT', 'Engineering', 'other', 'No',
       'Yes', 'year 1', 'year 2', 'year 3', 'year 4', 'high', 'low', 'mid']]
        dataToSend = {}
        dataToSend["depression"] = predict(depression, df)
        dataToSend["anxiety"] = predict(anxiety, df)
        dataToSend["stress"] = predict(stress, df)
        return dataToSend


if __name__ == "__main__":
    app.run(debug=True)
