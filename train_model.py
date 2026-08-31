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

# --- Threshold analysis ---
# 0.5 is an arbitrary default. Evaluate a range of thresholds on the same
# held-out test set and pick one based on the actual precision/recall
# tradeoff, rather than assuming 0.5 is correct.
from sklearn.metrics import precision_score, recall_score, f1_score

print("\n=== Threshold analysis (held-out test set) ===")
print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
threshold_results = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred_t = (y_proba >= t).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    threshold_results.append((t, p, r, f1))
    print(f"{t:>10.2f} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")

# Select the threshold with the best F1 (balances precision and recall)
# among the ones evaluated. This is a starting policy, not a claim that
# it is optimal for a real payments business — see README for caveats.
best_threshold = max(threshold_results, key=lambda x: x[3])[0]
print(f"\nSelected threshold (best F1 on test set): {best_threshold}")

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "threshold": best_threshold,
    },
    "models/fraud_model.joblib",
)
print("\nSaved model to models/fraud_model.joblib")
