"""
prediction.py
-------------
Loads the trained loan-approval model and provides a simple prediction
interface for new applicants.

Usage (as a script):
    python prediction.py

Usage (as a module):
    from prediction import predict_loan_approval
    result = predict_loan_approval({
        "Gender": "Male", "Married": "Yes", "Dependents": "0",
        "Education": "Graduate", "Self_Employed": "No",
        "ApplicantIncome": 5000, "CoapplicantIncome": 2000,
        "LoanAmount": 130, "Loan_Amount_Term": 360,
        "Credit_History": 1.0, "Property_Area": "Urban",
    })
"""

import os
import pickle
import pandas as pd

from feature_engineering import add_engineered_features
from data_preprocessing import CATEGORICAL_COLS

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "trained_model.pkl")

REQUIRED_FIELDS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area",
]


def load_artifact(path: str = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_loan_approval(applicant: dict, artifact: dict = None) -> dict:
    """
    Predict loan approval for a single applicant record.

    Parameters
    ----------
    applicant : dict
        Must contain: Gender, Married, Dependents, Education,
        Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, Property_Area
    artifact : dict, optional
        Pre-loaded model artifact. If not given, loaded from disk.

    Returns
    -------
    dict with keys: prediction (0/1), label (str), probability (float)
    """
    if artifact is None:
        artifact = load_artifact()

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]

    missing = [f for f in REQUIRED_FIELDS if f not in applicant]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    row = pd.DataFrame([applicant])
    row = add_engineered_features(row)

    categorical_present = [c for c in CATEGORICAL_COLS if c in row.columns]

    # NOTE: pd.get_dummies on a single-row DataFrame is unsafe here — with only
    # one row, every categorical column has exactly one unique value, so
    # drop_first=True drops the *only* category and produces zero dummy
    # columns for it. Reindexing against feature_names would then silently
    # fill every one-hot column with 0, corrupting the prediction (e.g. a
    # "Married_Yes" applicant would be encoded as if Married_Yes=0).
    #
    # Instead, build the one-hot columns explicitly against the exact
    # feature names learned at training time.
    row_aligned = pd.DataFrame(0, index=row.index, columns=feature_names)
    for col in feature_names:
        if col in row.columns:
            row_aligned[col] = row[col].values
    for cat_col in categorical_present:
        value = str(row.iloc[0][cat_col])
        dummy_col = f"{cat_col}_{value}"
        if dummy_col in feature_names:
            row_aligned.loc[row.index[0], dummy_col] = 1

    row_scaled = scaler.transform(row_aligned)

    pred = int(model.predict(row_scaled)[0])
    proba = float(model.predict_proba(row_scaled)[0, 1])

    return {
        "prediction": pred,
        "label": "Approved" if pred == 1 else "Rejected",
        "probability": round(proba, 4),
    }


if __name__ == "__main__":
    artifact = load_artifact()
    print(f"Loaded model: {artifact['model_name']} "
          f"(test accuracy = {artifact['test_accuracy']:.4f})\n")

    sample_applicants = [
        {
            "Gender": "Male", "Married": "Yes", "Dependents": "0",
            "Education": "Graduate", "Self_Employed": "No",
            "ApplicantIncome": 5000, "CoapplicantIncome": 2000,
            "LoanAmount": 130, "Loan_Amount_Term": 360,
            "Credit_History": 1.0, "Property_Area": "Urban",
        },
        {
            "Gender": "Female", "Married": "No", "Dependents": "1",
            "Education": "Not Graduate", "Self_Employed": "Yes",
            "ApplicantIncome": 2000, "CoapplicantIncome": 0,
            "LoanAmount": 150, "Loan_Amount_Term": 360,
            "Credit_History": 0.0, "Property_Area": "Rural",
        },
    ]

    for i, applicant in enumerate(sample_applicants, start=1):
        result = predict_loan_approval(applicant, artifact)
        print(f"Applicant {i}: {applicant}")
        print(f"  -> Prediction: {result['label']} "
              f"(approval probability = {result['probability']:.2%})\n")
