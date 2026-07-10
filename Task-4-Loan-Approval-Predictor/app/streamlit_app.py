"""
streamlit_app.py
------------------
A simple Streamlit web app for the Loan Approval Predictor.

Run with:
    streamlit run app/streamlit_app.py

The app loads the trained model artifact (outputs/trained_model.pkl),
collects applicant details through a form, and shows the predicted
approval decision along with the model's confidence.
"""

import os
import sys
import pickle

import pandas as pd
import streamlit as st

# Make src/ importable regardless of the working directory streamlit is launched from
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(SRC_DIR)

from feature_engineering import add_engineered_features  # noqa: E402
from data_preprocessing import CATEGORICAL_COLS  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "trained_model.pkl")


@st.cache_resource
def load_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(applicant: dict, artifact: dict) -> dict:
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]

    row = pd.DataFrame([applicant])
    row = add_engineered_features(row)

    categorical_present = [c for c in CATEGORICAL_COLS if c in row.columns]

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
    return {"prediction": pred, "label": "Approved" if pred == 1 else "Rejected",
            "probability": proba}


def main():
    st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

    st.title("🏦 Loan Approval Predictor")
    st.write(
        "Fill in the applicant's details below to predict whether their loan "
        "application is likely to be **approved** or **rejected**."
    )

    try:
        artifact = load_artifact()
    except FileNotFoundError:
        st.error(
            "No trained model found. Please run `python src/model_training.py` "
            "first to train and save a model."
        )
        return

    st.caption(
        f"Using model: **{artifact['model_name']}** "
        f"(test accuracy: {artifact['test_accuracy'] * 100:.1f}%)"
    )

    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        with col2:
            applicant_income = st.number_input(
                "Applicant Income (monthly)", min_value=0, value=5000, step=100
            )
            coapplicant_income = st.number_input(
                "Coapplicant Income (monthly)", min_value=0, value=0, step=100
            )
            loan_amount = st.number_input(
                "Loan Amount (in thousands)", min_value=0, value=130, step=5
            )
            loan_term = st.selectbox(
                "Loan Amount Term (days)", [360, 180, 240, 120, 300, 60, 84, 36, 12], index=0
            )
            credit_history = st.selectbox(
                "Credit History", ["Has credit history", "No credit history"]
            )

        submitted = st.form_submit_button("Predict Loan Approval")

    if submitted:
        applicant = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": 1.0 if credit_history == "Has credit history" else 0.0,
            "Property_Area": property_area,
        }

        result = predict(applicant, artifact)

        st.divider()
        if result["prediction"] == 1:
            st.success(f"✅ Prediction: **{result['label']}**")
        else:
            st.error(f"❌ Prediction: **{result['label']}**")

        st.metric("Approval Probability", f"{result['probability'] * 100:.1f}%")
        st.progress(min(max(result["probability"], 0.0), 1.0))

        with st.expander("See submitted applicant details"):
            st.json(applicant)


if __name__ == "__main__":
    main()
