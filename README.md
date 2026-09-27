# Fintech Complaint Triage

An NLP classifier that reads fintech complaints and routes them to the right team. Built with DistilBERT, deployed with FastAPI + Docker, and served through a custom frontend.

**Live demo:** https://fintech-triage-frontend.vercel.app
**API:** https://fintech-complaint-triage.onrender.com

## Why

Support teams at fintechs get thousands of complaints a day. Someone has to read each one and figure out where it goes — fraud team, billing, tech ops, or general support. Misroute a fraud dispute and you've got a regulatory problem on your hands. This project automates the first pass.

## What it does

Input: a complaint text.

Output:
```json
{
  "category": "payment_issue",
  "confidence": 0.76,
  "all_scores": {
    "account_issue": 0.065,
    "billing_dispute": 0.080,
    "fraud": 0.094,
    "payment_issue": 0.762
  }
}
```

Four classes:
- `fraud` — unauthorized transactions, identity theft, scams
- `billing_dispute` — wrong charges, fee disputes, refunds
- `payment_issue` — failed payments, money deducted but not received
- `account_issue` — login problems, account closure, KYC issues

## Data

Source: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).

The raw file is ~8.7 GB with 17.7M rows. Most rows have no written narrative, so I filtered to ~3.85M rows with text, then took a stratified sample by product type down to ~27k labeled examples.

Labels came from the CFPB `Issue` field, mapped to our 4 classes via keyword rules in `src/features/label_mapping.py`. This is where most of the noise in the dataset comes from — see the "Limitations" section.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.71 | 0.71 |
| TF-IDF + LinearSVC | 0.72 | 0.70 |
| TF-IDF + SGD | 0.72 | 0.71 |
| **DistilBERT (fine-tuned)** | **0.75** | **0.73** |

DistilBERT was fine-tuned on a Colab T4 for 3 epochs (max length 256, batch size 32, learning rate 2e-5).

Per-class F1 for the final model:

| Class | F1 |
|---|---|
| account_issue | 0.73 |
| billing_dispute | 0.80 |
| fraud | 0.64 |
| payment_issue | 0.77 |

## What I learned

Fine-tuning DistilBERT only got us +2 points over a TF-IDF baseline. That surprised me at first, but the reason is pretty clear: the labels are noisy.

Two things going on:

1. The CFPB `Issue` field doesn't always match the narrative. A complaint filed under "Closing your account" might actually describe fraud. So even a strong model can't do much better than the labels allow.
2. `fraud` and `payment_issue` overlap heavily. "Unauthorized transaction" could be either one depending on context.

If I were doing this in production, the next step wouldn't be a bigger model — it would be cleaning up the labels.

## Architecture

```
┌─────────────────────────┐
│  Frontend (Vercel)      │
│  HTML + CSS + JS        │
└────────────┬────────────┘
             │ HTTPS POST /predict
             ▼
┌─────────────────────────┐
│  API (Render)           │
│  FastAPI + Docker       │
│  fp16 DistilBERT        │
└────────────┬────────────┘
             │ Downloads model
             ▼
┌─────────────────────────┐
│  Model (HuggingFace Hub)│
│  Saksham-30/fintech-    │
│  distilbert-fp16        │
└─────────────────────────┘
```

- Model stored on HuggingFace Hub (~134 MB, fp16)
- FastAPI backend containerized with Docker, deployed on Render
- Frontend on Vercel
- UptimeRobot pings the API every 5 minutes to keep the container warm

## Setup

```bash
git clone https://github.com/Sakshamswami30/Fintech-Complaint-Triage.git
cd Fintech-Complaint-Triage

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux

pip install -r requirements.txt
```

The trained model isn't checked into the repo. It's downloaded from HuggingFace at startup.

## Running the API locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Swagger docs: http://localhost:8000/docs

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My payment failed but the money was deducted from my account"}'
```

## Docker

```bash
docker-compose up --build
```

The image uses the CPU-only PyTorch build to keep it under 2 GB.

## Repo layout

```
src/
  data/         # CFPB loading + sampling
  features/     # Label mapping rules
  models/       # Baseline, DistilBERT training, fp16 conversion
  api/          # FastAPI service
notebooks/      # EDA and label exploration
models/         # Saved artifacts (not tracked)
Dockerfile
docker-compose.yml
```

## Limitations

- **Label noise.** Roughly 5-10% of examples are probably mislabeled. The CFPB taxonomy changed over the years and doesn't map cleanly to our 4 classes.
- **Small training set.** 22k training examples is on the low end for BERT fine-tuning.
- **Low confidence means low confidence.** Predictions under 0.6 should probably go to a human.
- **Cold starts.** The Render free tier sleeps after 15 minutes of inactivity. First request after sleep takes 60–90 seconds. UptimeRobot mitigates this.
- **English only.**

## TODO

- [ ] Relabel a subset with an LLM and see if F1 improves
- [ ] Add confidence threshold with human-in-the-loop fallback
- [ ] Batch prediction endpoint
- [ ] ONNX export to speed up inference
- [ ] API tests

## Stack

Python 3.12, PyTorch, HuggingFace Transformers, scikit-learn, FastAPI, Docker, Render, Vercel.