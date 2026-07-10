# Loan Approval Predictor

## Intern Details

- Intern ID: **[FILL IN YOUR INTERN ID]**
- Full Name: **[FILL IN YOUR NAME]**
- Duration: **[FILL IN DURATION, e.g. 8 Weeks]**

> ✏️ *Replace the three placeholders above with your actual details before submitting.*

## Project Scope

This project builds an end-to-end machine learning pipeline that predicts
whether a loan application should be **approved** or **rejected**, based
on applicant details such as gender, marital status, education, income,
loan amount, credit history, and property area. The pipeline covers data
preprocessing, feature engineering, training and comparing three
classification models (Logistic Regression, Random Forest, XGBoost),
and evaluating the best model with accuracy, a confusion matrix, and
feature importance. A simple **Streamlit web app** is also included as
an interactive prediction interface.

## Workflow

```
Dataset
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Prediction
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib / Seaborn
- Streamlit (web app)
- Jupyter Notebook

## Folder Structure

```
Task-4-Loan-Approval-Predictor/
│
├── data/
│   └── loan_dataset.csv            # Public Loan Prediction dataset (491 records)
│
├── notebooks/
│   └── analysis.ipynb              # Full EDA + feature engineering + training walkthrough (pre-run)
│
├── src/
│   ├── data_preprocessing.py       # Loads data, drops ID column, imputes missing values
│   ├── feature_engineering.py      # Builds derived features, encodes, scales, splits
│   ├── model_training.py           # Trains & compares models, saves plots + trained model
│   └── prediction.py               # Loads the saved model and predicts on new applicants
│
├── app/
│   └── streamlit_app.py            # Bonus: interactive Streamlit prediction interface
│
├── screenshots/
│   ├── accuracy.png                # Model accuracy comparison bar chart
│   ├── confusion_matrix.png        # Confusion matrix of the best model
│   └── feature_importance.png      # Feature importance of the best model
│
├── outputs/
│   └── trained_model.pkl           # Pickled best model + scaler + feature names
│
├── documentation/
│   └── project_report.pdf          # Written project report
│
├── requirements.txt
└── README.md
```

- **data/** — the raw Loan Prediction CSV.
- **notebooks/** — an already-executed notebook walking through every step of the pipeline with visuals.
- **src/** — the actual production pipeline: preprocessing, feature engineering, training, and prediction, each in its own reusable module.
- **app/** — the bonus Streamlit web app for interactive predictions.
- **screenshots/** — generated evaluation charts (accuracy, confusion matrix, feature importance).
- **outputs/** — the final trained model artifact, ready to load for predictions.
- **documentation/** — the written project report (PDF).

## Dataset & Features

Public Loan Prediction dataset (491 records) with the following key features:

| Feature | Description |
|---|---|
| Gender | Male / Female |
| Married | Yes / No |
| Dependents | Number of dependents (0, 1, 2, 3+) |
| Education | Graduate / Not Graduate |
| Self_Employed | Yes / No |
| ApplicantIncome | Applicant's monthly income |
| CoapplicantIncome | Co-applicant's monthly income |
| LoanAmount | Requested loan amount (in thousands) |
| Loan_Amount_Term | Term of the loan in days |
| Credit_History | Whether the applicant has a credit history (1) or not (0) |
| Property_Area | Urban / Semiurban / Rural |
| **Loan_Status** | **Target: whether the loan was approved (1) or not (0)** |

Missing values are present in Gender, Married, Dependents, Self_Employed,
LoanAmount, Loan_Amount_Term, and Credit_History, and are imputed
automatically (mode for categorical columns, median for numeric ones).

### Engineered Features
- **TotalIncome** = ApplicantIncome + CoapplicantIncome
- **LoanAmountToIncome** = LoanAmount / (TotalIncome + 1) — affordability ratio
- **EMI** = LoanAmount / Loan_Amount_Term — rough monthly repayment burden
- **HasCoapplicant** = 1 if CoapplicantIncome > 0 else 0

## Results

**Accuracy:**

| Model | Test Accuracy |
|---|---|
| **Logistic Regression (Best)** | **84.8%** |
| Random Forest | 83.8% |
| XGBoost | 78.8% |

**Confusion Matrix:** see `screenshots/confusion_matrix.png`

**Graphs:**
- Model accuracy comparison — `screenshots/accuracy.png`
- Feature importance — `screenshots/feature_importance.png`

Credit History and Total Income are consistently the strongest
predictors of loan approval, matching real-world lending intuition.

## Screenshots

(Add your own screenshots of the running Streamlit app here after
running it locally, e.g. `screenshots/app_screenshot.png`.)

- `screenshots/accuracy.png`
- `screenshots/confusion_matrix.png`
- `screenshots/feature_importance.png`

## How to Run

```bash
pip install -r requirements.txt

cd src
python model_training.py
python prediction.py
```

### Bonus: Run the Streamlit web app

```bash
streamlit run app/streamlit_app.py
```

This launches an interactive form in your browser where you can enter
an applicant's details and get an instant Approved/Rejected prediction
with a confidence score.

## Output

Running `model_training.py` produces:
- `screenshots/accuracy.png`, `screenshots/confusion_matrix.png`, `screenshots/feature_importance.png`
- `outputs/trained_model.pkl`

Running `prediction.py` prints predictions for sample applicants, e.g.:

```
Applicant 1: Male, Married, Graduate, Income 5000+2000, Credit History 1.0, Urban
  -> Prediction: Approved (approval probability = 88.5%)

Applicant 2: Female, Not Married, Not Graduate, Income 2000, Credit History 0.0, Rural
  -> Prediction: Rejected (approval probability = 0.2%)
```

## Future Enhancements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for all three models
- Cross-validation instead of a single train/test split for more robust accuracy estimates
- Handling class imbalance explicitly (e.g. SMOTE or class-weighting)
- Adding SHAP values for more detailed, per-prediction feature explanations
- Deploying the Streamlit app publicly (e.g. Streamlit Community Cloud) for live demos
- Expanding the model comparison with additional algorithms (e.g. LightGBM, CatBoost)
