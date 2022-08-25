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


file = open(os.path.join(basedir, "finalized_model.sav"), "rb")
model = pickle.load(file)
file.close()

file1 = open(os.path.join(basedir, "models_depression.pkl"), "rb")
depression = pickle.load(file1)
file1.close()

file1 = open(os.path.join(basedir, "models_stress.pkl"), "rb")
stress = pickle.load(file1)
file1.close()


@app.route("/")
@cross_origin()
def home():
    return "hello"


@app.route("/predict", methods=["POST", "GET"])
def predict():
    def predict(model, dataframe):
        result = model.predict_proba(np.array(dataframe))
        return result[0]

    if request.method == "POST":
        health_input = request.get_json(force=True)
        df = pd.DataFrame(health_input, index=[0])
        df = df[
            [
                "Age",
                "Gender",
                "CGPA",
                "free_time",
                "difficult_relax",
                "social_media",
                "look_forward",
                "overreact",
                "dry_mouth",
                "trembling_suffocation",
                "least_enthusiasm",
                "feel_scared",
                "games_OTT_platforms",
                "offended",
                "stay_negative",
            ]
        ]
        dataToSend = {}
        prediction = predict(model, df)
        dataToSend["healthy"] = prediction[0]
        dataToSend["stressed"] = prediction[1]
        dataToSend["critical"] = prediction[2]
        return dataToSend


if __name__ == "__main__":
    app.run(debug=True)
