import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '../model/model.pkl')  # or pipeline.pkl if using pipeline
SCALER_PATH = os.path.join(BASE_DIR, '../model/scaler.pkl')

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature config
FEATURES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
    "sex", "smoking", "time"
]
CONTINUOUS = [
    "age", "creatinine_phosphokinase", "ejection_fraction", "platelets",
    "serum_creatinine", "serum_sodium", "time"
]
BINARY = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

VALID_RANGES = {
    "age": (18, 120),
    "creatinine_phosphokinase": (10, 10000),
    "ejection_fraction": (10, 80),
    "platelets": (100, 500),
    "serum_creatinine": (0.1, 10),
    "serum_sodium": (100, 160),
    "time": (1, 365)
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            values = []
            for feature in FEATURES:
                raw_val = request.form.get(feature)

                if feature in BINARY:
                    value = 1.0 if raw_val in ["1", "Yes", "Male"] else 0.0
                else:
                    value = float(raw_val)

                if feature in VALID_RANGES:
                    min_val, max_val = VALID_RANGES[feature]
                    if not (min_val <= value <= max_val):
                        raise ValueError(f"{feature} must be between {min_val} and {max_val}")

                values.append(value)

            input_df = pd.DataFrame([values], columns=FEATURES)
            input_df[CONTINUOUS] = scaler.transform(input_df[CONTINUOUS])

            pred = model.predict(input_df)[0]
            prediction = "Will Survive" if pred == 0 else "At Risk"

        except Exception as e:
            prediction = f"Error: Invalid input. {str(e)}"

        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return render_template("prediction_snippet.html", prediction=prediction)

    return render_template("index.html", features=FEATURES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
