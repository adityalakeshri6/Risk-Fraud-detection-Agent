import os

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="Fraud/Risk Detection Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = joblib.load("models/fraud_model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]

# Rough human-readable baselines, used to describe how unusual a value is.
# In a real system these would come from the merchant/user's own history.
TYPICAL = {
    "amount": 800,
    "txn_velocity_1min": 0.5,
    "geo_distance_km": 20,
    "is_new_device": 0,
    "hour_of_day": 14,
    "merchant_risk_score": 0.25,
}


class Transaction(BaseModel):
    amount: float = Field(..., description="Transaction amount in INR")
    txn_velocity_1min: int = Field(..., description="Transactions from this user in the last 60s")
    geo_distance_km: float = Field(..., description="Distance from user's last known location, km")
    is_new_device: int = Field(..., ge=0, le=1)
    hour_of_day: int = Field(..., ge=0, le=23)
    merchant_risk_score: float = Field(..., ge=0, le=1)


def build_explanation(features: dict, contributions: list, score: float) -> str:
    """
    Turns the top contributing features into a plain-English explanation.
    Uses the Claude API if ANTHROPIC_API_KEY is set; otherwise falls back
    to a simple templated explanation so the endpoint still works without
    a key (useful for local testing before you wire up the API).
    """
    top = contributions[:3]
    facts = "; ".join(f"{name}={features[name]} (typical: {TYPICAL[name]})" for name, _ in top)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return f"[template] Flagged at {score:.0%} risk. Top factors: {facts}."

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"A transaction was scored {score:.0%} risk of fraud by a model. "
        f"The top contributing factors, with the typical/expected value for comparison, are: {facts}. "
        "In 1-2 short sentences, explain to a risk analyst why this transaction looks risky. "
        "Be specific and concrete, no hedging, no disclaimers."
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


@app.post("/score")
def score_transaction(txn: Transaction):
    row = pd.DataFrame([txn.model_dump()])[FEATURES]
    risk_score = float(model.predict_proba(row)[0][1])

    importances = model.feature_importances_
    contributions = sorted(zip(FEATURES, importances), key=lambda x: -x[1])

    explanation = build_explanation(txn.model_dump(), contributions, risk_score)

    return {
        "risk_score": round(risk_score, 4),
        "flag": "high_risk" if risk_score > 0.5 else "low_risk",
        "top_factors": [name for name, _ in contributions[:3]],
        "explanation": explanation,
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}
