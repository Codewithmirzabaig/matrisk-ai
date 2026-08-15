"""FastAPI service for health and financial-risk inference."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.financial.credit_risk import calculate_credit_risk

app=FastAPI(title="MatRisk AI",version="1.0.0")

class CreditRequest(BaseModel):
    pd: float=Field(ge=0,le=1)
    loan_outstanding_m: float=Field(ge=0)
    condition_rating: float=Field(ge=1,le=9)

@app.get("/health")
def health() -> dict[str,str]: return {"status":"healthy","version":"1.0.0"}

@app.post("/credit-risk")
def credit_risk(request: CreditRequest) -> dict:
    return calculate_credit_risk(**request.model_dump()).__dict__

