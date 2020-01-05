from flask import Flask, render_template      
from keras.models import load_model
import flask
import pickle
import re
import numpy as np
import json

def clean_text(text):
    clean_text = text.lower()
    regex = re.compile('[^a-zA-Z ]')
    clean_text = regex.sub(' ', clean_text)
    return clean_text

vect = pickle.load( open( "tfidf.pickle", "rb" ) )
labels_series = pickle.load( open( "labels.pickle", "rb" ) )
model = load_model("dense.h5")

print(vect)
print(model.summary())
print(labels_series)


app = Flask(__name__)
@app.route("/")
def home():
    return "this is home, try requesting /classify/TEXT to classify new text"
    

@app.route("/classify/<comment>")
def classify(comment):
    response = {}
    response['comment'] = comment

    features = vect.transform([clean_text(comment)])
    preds = model.predict(features)[0]
    probas = {}
    for i, label in enumerate(labels_series):
        probas[label] = float(preds[i])

    response['ratings'] = probas
    response['success'] = True

    return flask.jsonify(response)


if __name__ == "__main__":
    app.run(debug=False, threaded=False)



