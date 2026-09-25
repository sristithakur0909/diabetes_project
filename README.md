# Diabetic Prediction System — Logistic Regression

Naviotech Solution Pvt. Ltd. — Machine Learning Internship
**Author:** Sristi Thakur

Predicts whether a patient is likely diabetic from routine diagnostic
measurements, using a Logistic Regression classifier trained on the
[kandij/diabetes-dataset](https://www.kaggle.com/datasets/kandij/diabetes-dataset).

## Project Structure

```
diabetes_project/
├── ml/
│   ├── diabetes.csv         # training dataset (Kaggle: kandij/diabetes-dataset)
│   ├── train.py             # full training + evaluation pipeline
│   ├── save_model.py        # trains final model and exports it for the API
│   └── results.json         # saved metrics from the evaluation run
├── backend/
│   ├── app.py                # Flask REST API serving the model
│   ├── requirements.txt
│   ├── model.joblib          # trained Logistic Regression model
│   ├── scaler.joblib         # fitted StandardScaler
│   └── feature_order.joblib  # expected input feature order
├── frontend/
│   └── index.html            # single-page form that calls the API
└── README.md
```

## Dataset

- 768 records, 8 input features, binary target `Outcome` (1 = Diabetic, 0 = Not Diabetic)
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
  DiabetesPedigreeFunction, Age
- Source: Kaggle — `kandij/diabetes-dataset`

## Model

- Algorithm: Logistic Regression (scikit-learn)
- Preprocessing: StandardScaler on all 8 features
- Split: 80% train (614) / 20% test (154), stratified
- Validation: 5-fold stratified cross-validation on the training set

## Results

See `ml/results.json` for the full metrics from the training run, including
accuracy, precision, recall, F1-score, the confusion matrix and the learned
feature coefficients.

## Running locally

```bash
cd ml
python train.py          # reproduce the evaluation
python save_model.py     # retrain and export model.joblib / scaler.joblib

cd ../backend
pip install -r requirements.txt
python app.py             # serves on http://localhost:5000
```

Open `frontend/index.html` in a browser (update `API_URL` at the top of the
script to point at your backend once deployed).

## Deployment

- **Backend** — deploy `backend/` to [Render](https://render.com) as a Python
  web service (`gunicorn app:app`, build command `pip install -r requirements.txt`).
- **Frontend** — deploy `frontend/` to [Vercel](https://vercel.com) as a static
  site, after updating `API_URL` in `index.html` to your live backend URL.

> Update the live URLs below once deployed, and in the project report/PPT:
> - Live Application: https://diabetes-project-ruddy.vercel.app
> - Backend API: https://diabetes-prediction-api-hcj6.onrender.com
