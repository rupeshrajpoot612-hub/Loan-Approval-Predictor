"""
data_preprocessing.py
----------------------
Loads and cleans the Loan Prediction dataset.

Handles:
  - Dropping the non-predictive Loan_ID identifier
  - Imputing missing values (mode for categorical, median for numeric)
  - Basic type fixes (Dependents "3+" -> 3, Credit_History as category)
"""

import os
import pandas as pd

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loan_dataset.csv")

TARGET_COL = "Loan_Status"
ID_COL = "Loan_ID"

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Credit_History", "Property_Area",
]
NUMERIC_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
]


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers, fix dtypes, and impute missing values."""
    df = df.copy()

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # "3+" -> 3 so Dependents can be treated numerically if desired later
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", "3")

    # Impute categorical columns with the mode
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Impute numeric columns with the median
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


if __name__ == "__main__":
    df = load_data()
    print("Raw shape:", df.shape)
    print("Missing values before cleaning:\n", df.isnull().sum())

    df_clean = clean_data(df)
    print("\nMissing values after cleaning:\n", df_clean.isnull().sum().sum(), "total")
    print("\nCleaned sample:\n", df_clean.head())
