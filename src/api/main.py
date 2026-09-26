# main.py
# FastAPI endpoint for complaint triage.
# Run: uvicorn src.api.main:app --reload --port 8000

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_PATH = "Saksham-30/fintech-distilbert"
MAX_LEN = 256

app = FastAPI(
    title="Fintech Complaint Triage",
    description="Classifies complaints into fraud, billing_dispute, payment_issue, account_issue",
    version="1.0.0",
)

# allow the frontend (running on a different origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


class ComplaintRequest(BaseModel):
    text: str = Field(..., min_length=5, description="The complaint text to classify")


class ComplaintResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_PATH}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=ComplaintResponse)
def predict(req: ComplaintRequest):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(probs.argmax().item())
    category = model.config.id2label[pred_id]
    confidence = float(probs[pred_id].item())

    all_scores = {
        model.config.id2label[i]: round(float(probs[i].item()), 4)
        for i in range(len(probs))
    }

    return ComplaintResponse(
        category=category,
        confidence=round(confidence, 4),
        all_scores=all_scores,
    )