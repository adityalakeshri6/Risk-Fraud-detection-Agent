# Fraud/Risk Detection Agent — Day 1

Baseline model for the Razorpay AI Buildathon submission.

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
```

## Current result

ROC-AUC ~1.00 on the held-out set — expected, since the synthetic fraud/legit
distributions are cleanly separated right now. `geo_distance_km` dominates
feature importance (0.94), meaning the model is basically only using
location right now.

**This is your Day 4 "what broke" story**: a model that's *too* easy is a
red flag in a real pitch — it means the synthetic data isn't realistic
enough yet, or the model is leaning on one signal. Before Day 4, consider:
- Adding overlapping noise between fraud/legit distributions so no single
  feature fully separates them
- Checking the confusion matrix, not just AUC, once the model is less perfect
- Explicitly calling this out in your pitch video as something you caught
  and fixed — panels want to see you notice a "too good to be true" model,
  not just report a good number

## Next steps (Day 2)

- Wrap `models/fraud_model.joblib` in a FastAPI endpoint
- Feed the top contributing features into the Claude API to generate a
  plain-English explanation per transaction
