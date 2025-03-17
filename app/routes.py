from app import app
from flask import render_template, flash, redirect, url_for, request
import requests
from io import BytesIO
from PIL import Image



HF_API_URL = "https://microtransactions-cocbclassifyfastapi.hf.space"
static_classes = ['air_defense', 'air_sweeper', 'archer_tower', 'bomb_tower', 'cannon', 'hidden_tesla', 'inferno_tower', 'mortar', 'wizard_tower', 'xbow']


# Implicitly allows get
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('predict_form.html')
    
    if request.method == 'POST':

        image = request.files['image'] # return flask FileStorage obj .read will read binary content

        image_bytes = image.read() # Raw bytes, Investigate this

        # Send to HF API see misc notes in flask notes
        response = requests.post(
            HF_API_URL + "/predict/",
            files={"file": image_bytes}  # Match the parameter name in FastAPI
        )

        if response.status_code == 200:
            result = response.json() # This method auto converts json into py dict

            probabilities = result['pred_probabilities'] # gets the list of prob

            max_prob_index = probabilities.index(max(probabilities)) # get index of max

            predicted_class = static_classes[max_prob_index]

            predicted_class_probability = probabilities[max_prob_index] * 100


            return render_template('result.html', result=result, predicted_class=predicted_class, confidence=predicted_class_probability)
        else:
            flash(f"Error: API returned status code {response.status_code}")
            return redirect(url_for('predict'))