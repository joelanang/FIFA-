import joblib
import zipfile
import os

with zipfile.ZipFile('final_model.zip', 'r') as zip_ref:
    zip_ref.extractall()

model_path = os.path.join(os.getcwd(), 'final_model', 'final_model.pkl')
ensemble = joblib.load(model_path)


from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        attributes = [
            'movement_reactions', 'mentality_composure', 'passing',
            'release_clause_eur', 'dribbling', 'wage_eur', 'power_shot_power',
            'value_eur', 'mentality_vision', 'attacking_short_passing',
            'physic', 'skill_long_passing', 'age'
        ]

        input_data = {}
        for attr in attributes:
            value = float(request.form.get(attr, 0))
            input_data[attr] = value

        prediction = ensemble.predict([list(input_data.values())])
        prediction = round(prediction[0])

        return render_template("result.html", prediction=prediction, confidence=97.24)

    return render_template("form.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)