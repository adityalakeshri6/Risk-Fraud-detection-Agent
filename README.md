# Fraud/Risk Detection Agent

Built for the Razorpay AI Buildathon submission — scores transactions for
fraud risk in real time and explains the "why" in plain English instead of
just returning a number.

## Why synthetic data instead of Kaggle's Credit Card Fraud dataset

The popular Kaggle dataset (`creditcard.csv`) has features `V1`-`V28` that
are PCA-anonymized — you can't explain "V17 was high" to a human. Since the
whole pitch is an agent that *explains* why a transaction is risky, this
project uses a synthetic dataset with interpretable features instead:
`amount`, `txn_velocity_1min`, `geo_distance_km`, `is_new_device`,
`hour_of_day`, `merchant_risk_score`.

## Quick start

```bash
pip install -r requirements.txt
python data/generate_synthetic.py   # writes data/transactions.csv
python train_model.py               # trains + saves models/fraud_model.joblib
export ANTHROPIC_API_KEY=your_key_here   # optional — falls back to a template without it
uvicorn app:app --reload
```

Then open `http://localhost:8000` — the demo UI loads directly.

## Architecture

1. **Signal layer** — six interpretable transaction features (see above)
2. **Scoring layer** — a `GradientBoostingClassifier` (scikit-learn) outputs
   a 0-1 risk score
3. **Explanation layer** — the top contributing features + score are sent
   to the Claude API, which generates a short, specific, human-readable
   explanation of the flag

`app.py` exposes this as a FastAPI `POST /score` endpoint and also serves
the demo frontend at `/`.

## Model iteration log

**v1 (Day 1):** ROC-AUC ~1.00 — too perfect. `geo_distance_km` alone
accounted for 94% of feature importance, meaning the model was basically a
single-feature threshold in disguise, not a real multi-signal fraud model.

**v2 (current):** added realistic overlap between the fraud/legit synthetic
distributions so no single feature fully separates the classes. Result:
ROC-AUC 0.986, fraud recall 68% / precision 91%, and importance now spread
across `geo_distance_km` (57%), `merchant_risk_score` (18%),
`hour_of_day` (12%), and the rest. A believable model, not a toy.

## What broke (Day 4 stress test)

Ran edge cases against `/score`: extreme amounts, zero values, missing
fields, out-of-range values, wrong types, and negative amounts.

**Worked correctly already:** missing fields, out-of-range
`hour_of_day`/`merchant_risk_score`, and wrong types were all correctly
rejected with clear `422` errors — Pydantic validation was doing its job.

**Bug found:** negative transaction amounts (`amount: -500`) were silently
accepted and scored as a normal low-risk transaction. A fraud scorer that
happily scores nonsensical input isn't trustworthy input handling for a
payments company.

**Fix:** added a `ge=0` constraint to the `amount` field, so negative
amounts are now rejected at the validation layer before reaching the model.
Verified the fix rejects negative amounts while leaving normal transactions
unaffected.

## Frontend

`static/index.html` — a single-page demo ("The Risk Ledger"). Submit a
transaction on the left, see it appear as a stamped ledger entry on the
right (CLEARED / FLAGGED), with the risk score, top contributing factors,
and plain-English explanation.
