import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")

# Load dataset
df = pd.read_csv(os.path.join(BASE_DIR, "diabetes.csv"))

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Save model
joblib.dump(model, os.path.join(BACKEND_DIR, "model.joblib"))

# Save scaler
joblib.dump(scaler, os.path.join(BACKEND_DIR, "scaler.joblib"))

# Save feature order
joblib.dump(list(X.columns), os.path.join(BACKEND_DIR, "feature_order.joblib"))

print("Model saved successfully!")
print("Saved files:")
print(" - backend/model.joblib")
print(" - backend/scaler.joblib")
print(" - backend/feature_order.joblib")