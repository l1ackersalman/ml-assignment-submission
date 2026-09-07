"""
Predicts subscription likelihood for a single new customer record.

Usage: edit the `sample_customer` dict below and run:
    python predict.py
"""

import joblib
import pandas as pd

MODEL_PATH = "./models/logreg_pipeline.joblib"

sample_customer = {
    "age": 41,
    "balance": 1200,
    "day": 15,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "poutcome": "unknown",
}


def main():
    pipeline = joblib.load(MODEL_PATH)
    df = pd.DataFrame([sample_customer])

    prob = pipeline.predict_proba(df)[0, 1]
    pred = "yes" if prob >= 0.5 else "no"

    print(f"Predicted subscription: {pred}  (probability: {prob:.2%})")


if __name__ == "__main__":
    main()
