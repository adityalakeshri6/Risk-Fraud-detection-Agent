"""
Day 1: baseline fraud/risk model.

Trains a gradient boosting classifier on the synthetic transaction data,
evaluates it, and saves the model + feature list for the API layer (Day 2).

Run:
    python data/generate_synthetic.py   # creates data/transactions.csv
    python train_model.py
"""

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "amount",
    "txn_velocity_1min",
    "geo_distance_km",
    "is_new_device",
    "hour_of_day",
    "merchant_risk_score",
]
TARGET = "label"

df = pd.read_csv("data/transactions.csv")
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("=== Classification report ===")
print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

print("\n=== Feature importances (use these for the explanation layer) ===")
for feat, imp in sorted(
    zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]
):
    print(f"  {feat:22s} {imp:.3f}")

joblib.dump({"model": model, "features": FEATURES}, "models/fraud_model.joblib")
print("\nSaved model to models/fraud_model.joblib")
