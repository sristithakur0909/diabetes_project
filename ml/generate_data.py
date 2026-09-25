import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 768
pos_rate = 0.349
n_pos = int(round(n * pos_rate))
n_neg = n - n_pos
outcome = np.array([1]*n_pos + [0]*n_neg)
rng.shuffle(outcome)

def feat(mean0, std0, mean1, std1, lo, hi, integer=False):
    vals = np.where(outcome==1,
                     rng.normal(mean1, std1, n),
                     rng.normal(mean0, std0, n))
    vals = np.clip(vals, lo, hi)
    if integer:
        vals = np.round(vals)
    return vals

pregnancies = feat(3.3, 3.0, 4.9, 3.6, 0, 17, integer=True)
glucose     = feat(110.0, 24.0, 142.0, 31.0, 44, 199)
bp          = feat(68.0, 18.0, 70.8, 21.2, 24, 122)
skin        = feat(19.7, 14.9, 22.2, 17.7, 0, 99)
insulin     = feat(68.8, 98.9, 100.3, 138.7, 0, 846)
bmi         = feat(30.3, 7.0, 35.4, 7.3, 18.2, 67.1)
dpf         = feat(0.43, 0.30, 0.55, 0.38, 0.078, 2.42)
age         = feat(31.2, 11.0, 37.1, 10.9, 21, 81, integer=True)

df = pd.DataFrame({
    "Pregnancies": pregnancies.astype(int),
    "Glucose": glucose.round(1),
    "BloodPressure": bp.round(1),
    "SkinThickness": skin.round(1),
    "Insulin": insulin.round(1),
    "BMI": bmi.round(1),
    "DiabetesPedigreeFunction": dpf.round(3),
    "Age": age.astype(int),
    "Outcome": outcome
})
df.to_csv("/home/claude/diabetes_project/ml/diabetes.csv", index=False)
print(df.shape)
print(df["Outcome"].value_counts())
print(df.describe())
