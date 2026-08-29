"""
Generates a synthetic transaction dataset with interpretable, fraud-relevant
features. We use synthetic data instead of the public Kaggle credit card
dataset on purpose: that dataset's features (V1-V28) are PCA-anonymized and
NOT human-readable, which breaks the "explain why this was flagged" pitch.
These features are things you can actually narrate in a demo.

Run: python generate_synthetic.py
Output: transactions.csv in this folder
"""

import os

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 20000
FRAUD_RATE = 0.03  # 3% fraud, realistic-ish imbalance

n_fraud = int(N_SAMPLES * FRAUD_RATE)
n_legit = N_SAMPLES - n_fraud


def make_legit(n):
    return pd.DataFrame({
        "amount": np.round(np.random.lognormal(mean=6.5, sigma=1.0, size=n), 2),
        "txn_velocity_1min": np.random.poisson(0.3, size=n),       # txns in last 60s
        "geo_distance_km": np.abs(np.random.normal(5, 8, size=n)), # distance from last known location
        "is_new_device": np.random.choice([0, 1], size=n, p=[0.92, 0.08]),
        "hour_of_day": np.random.normal(14, 4, size=n).clip(0, 23).astype(int),
        "merchant_risk_score": np.random.beta(2, 8, size=n),       # 0-1, most merchants low risk
        "label": 0,
    })


def make_fraud(n):
    return pd.DataFrame({
        "amount": np.round(np.random.lognormal(mean=7.5, sigma=1.3, size=n), 2),
        "txn_velocity_1min": np.random.poisson(3.5, size=n),
        "geo_distance_km": np.abs(np.random.normal(300, 200, size=n)),
        "is_new_device": np.random.choice([0, 1], size=n, p=[0.35, 0.65]),
        "hour_of_day": np.random.normal(3, 3, size=n).clip(0, 23).astype(int),
        "merchant_risk_score": np.random.beta(5, 3, size=n),
        "label": 1,
    })


df = pd.concat([make_legit(n_legit), make_fraud(n_fraud)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transactions.csv")
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows ({df['label'].sum()} fraud) to {out_path}")
