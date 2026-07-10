"""
feature_engineering.py
------------------------
Builds derived features on top of the cleaned Loan Prediction dataset,
then encodes and scales everything into model-ready train/test arrays.

Engineered features:
  - TotalIncome        = ApplicantIncome + CoapplicantIncome
  - LoanAmountToIncome = LoanAmount / (TotalIncome + 1)   (affordability ratio)
  - EMI                = LoanAmount / Loan_Amount_Term    (rough monthly burden)
  - HasCoapplicant     = 1 if CoapplicantIncome > 0 else 0

These features often carry more predictive signal than the raw income
and loan amount columns individually, since approval realistically
depends on how large a loan is *relative to* a household's income.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data_preprocessing import load_data, clean_data, TARGET_COL, CATEGORICAL_COLS


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmountToIncome"] = df["LoanAmount"] / (df["TotalIncome"] + 1)
    df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"].replace(0, pd.NA)
    df["EMI"] = df["EMI"].fillna(df["EMI"].median())
    df["HasCoapplicant"] = (df["CoapplicantIncome"] > 0).astype(int)

    return df


def encode_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """One-hot encode categoricals, label-encode the target, split, and scale."""
    y_raw = df[TARGET_COL]
    X_raw = df.drop(columns=[TARGET_COL])

    categorical_present = [c for c in CATEGORICAL_COLS if c in X_raw.columns]
    X = pd.get_dummies(X_raw, columns=categorical_present, drop_first=True)

    label_encoder = None
    if not pd.api.types.is_numeric_dtype(y_raw):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y_raw), index=y_raw.index, name=TARGET_COL)
    else:
        y = y_raw

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns), label_encoder


def preprocess(test_size: float = 0.2, random_state: int = 42):
    """Full pipeline: load -> clean -> engineer features -> encode -> split -> scale."""
    df = load_data()
    df = clean_data(df)
    df = add_engineered_features(df)
    return encode_and_split(df, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, feature_names, label_encoder = preprocess()
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Number of features after encoding: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print(f"Class balance (train):\n{y_train.value_counts(normalize=True)}")
