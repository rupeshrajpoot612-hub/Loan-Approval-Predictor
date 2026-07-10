"""
model_training.py
------------------
Trains and compares three classifiers on the Loan Prediction dataset:
Logistic Regression, Random Forest, and XGBoost.

Saves:
  - screenshots/accuracy.png            (model accuracy comparison)
  - screenshots/confusion_matrix.png    (confusion matrix of the best model)
  - screenshots/feature_importance.png  (feature importance of the best model)
  - outputs/trained_model.pkl           (best model + scaler + feature names + label encoder)
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score,
)

from feature_engineering import preprocess

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
        ),
    }


def train_and_compare(X_train, X_test, y_train, y_test):
    models = get_candidate_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {"model": model, "accuracy": acc}
        print(f"{name:22s} accuracy = {acc:.4f}")
    return results


def plot_accuracy_comparison(results, save_path):
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] * 100 for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accuracies, color=["#4C72B0", "#DD8452", "#55A868"])
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison — Loan Approval Prediction")
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                  f"{acc:.1f}%", ha="center", fontweight="bold")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, best_name, save_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    labels = ["Rejected", "Approved"]
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black",
                      fontsize=14, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(best_model, best_name, feature_names, save_path):
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        return  # no importance to plot

    order = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    plt.figure(figsize=(9, 6))
    plt.barh(sorted_features[::-1], sorted_importances[::-1], color="#55A868")
    plt.xlabel("Importance")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    X_train, X_test, y_train, y_test, scaler, feature_names, label_encoder = preprocess()

    results = train_and_compare(X_train, X_test, y_train, y_test)

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = results[best_name]["model"]
    best_acc = results[best_name]["accuracy"]
    print(f"\nBest model: {best_name} ({best_acc:.4f} accuracy)")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    plot_accuracy_comparison(results, os.path.join(SCREENSHOTS_DIR, "accuracy.png"))
    plot_confusion_matrix(y_test, y_pred, best_name,
                           os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png"))
    plot_feature_importance(best_model, best_name, feature_names,
                             os.path.join(SCREENSHOTS_DIR, "feature_importance.png"))

    artifact = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "model_name": best_name,
        "test_accuracy": best_acc,
        "all_results": {n: r["accuracy"] for n, r in results.items()},
    }
    model_path = os.path.join(OUTPUTS_DIR, "trained_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved trained model to {model_path}")

    return results, best_name


if __name__ == "__main__":
    main()
