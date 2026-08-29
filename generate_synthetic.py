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
        "txn_velocity_1min": np.random.poisson(0.5, size=n),        # txns in last 60s
        "geo_distance_km": np.abs(np.random.normal(20, 40, size=n)),# distance from last known location
        "is_new_device": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
        "hour_of_day": np.random.normal(14, 5, size=n).clip(0, 23).astype(int),
        "merchant_risk_score": np.random.beta(2, 6, size=n),        # 0-1, most merchants low risk
        "label": 0,
    })


def make_fraud(n):
    return pd.DataFrame({
        "amount": np.round(np.random.lognormal(mean=7.0, sigma=1.4, size=n), 2),
        "txn_velocity_1min": np.random.poisson(2.0, size=n),
        "geo_distance_km": np.abs(np.random.normal(120, 150, size=n)),
        "is_new_device": np.random.choice([0, 1], size=n, p=[0.55, 0.45]),
        "hour_of_day": np.random.normal(7, 6, size=n).clip(0, 23).astype(int),
        "merchant_risk_score": np.random.beta(4, 4, size=n),
        "label": 1,
    })


df = pd.concat([make_legit(n_legit), make_fraud(n_fraud)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transactions.csv")
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows ({df['label'].sum()} fraud) to {out_path}")
