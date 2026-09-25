import pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "diabetes.csv"))
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 5-fold stratified CV on training data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)
cv_scores = cross_val_score(model, X_train_s, y_train, cv=skf, scoring="accuracy")

model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

coefs = dict(zip(X.columns, model.coef_[0].round(4)))

results = {
    "n_total": len(df),
    "n_features": X.shape[1],
    "n_train": len(X_train),
    "n_test": len(X_test),
    "cv_scores": cv_scores.round(4).tolist(),
    "cv_mean": round(cv_scores.mean()*100, 2),
    "cv_std": round(cv_scores.std()*100, 2),
    "accuracy": round(acc*100, 2),
    "precision": round(prec*100, 2),
    "recall": round(rec*100, 2),
    "f1": round(f1*100, 2),
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "coefficients": coefs,
    "intercept": round(model.intercept_[0], 4)
}
print(json.dumps(results, indent=2))
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
