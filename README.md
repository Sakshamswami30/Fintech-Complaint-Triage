# Fintech Complaint Triage

Classifies fintech customer complaints into 4 categories (fraud, billing dispute, payment issue, account issue) and serves predictions via a FastAPI endpoint.

## Problem

Fintech support teams receive thousands of complaints daily across email, chat, and app forms. Manual triage is slow, expensive, and error-prone — and misrouting a fraud complaint can mean regulatory penalties. This project builds an NLP classifier that reads a complaint and routes it to the right team.

## Data

Source: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) (~17.7M rows, ~8.7 GB raw).

I filtered to rows with a written narrative, kept 7 products aligned with the target classes, and labeled each complaint by mapping the CFPB `Issue` field to one of four classes:

| Class | Description | Team |
|---|---|---|
| `fraud` | Unauthorized transactions, identity theft, scams | Fraud Ops |
| `billing_dispute` | Wrong charges, fee disputes, refund requests | Billing |
| `payment_issue` | Failed transactions, money deducted but not received | Tech Ops |
| `account_issue` | Login, KYC, account closure, access problems | Customer Support |

Final dataset: **27,692 labeled complaints**, roughly balanced (2.4:1 max/min ratio).

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.71 | 0.71 |
| TF-IDF + LinearSVC | 0.72 | 0.70 |
| TF-IDF + SGD | 0.72 | 0.71 |
| DistilBERT (fine-tuned) | **0.75** | **0.73** |

DistilBERT was fine-tuned on a Colab T4 for 3 epochs (max length 256, batch size 32, learning rate 2e-5).

Per-class F1:

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
2. `fraud` and `payment_issue` overlap heavily. "Unauthorized transaction" could be either one depending on context. The confusion matrix shows the model flipping between these two often.

If I were doing this in production, the next step wouldn't be a bigger model — it would be cleaning up the labels.

## Project Structure

```
fintech-complaint-triage/
├── src/
│   ├── data/               # CFPB loading and sampling
│   ├── features/           # Label mapping rules
│   ├── models/             # Baseline, comparison, DistilBERT training
│   └── api/                # FastAPI service
├── notebooks/              # EDA and label exploration
├── models/                 # Trained model artifacts (not tracked)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Sakshamswami30/Fintech-Complaint-Triage.git
cd Fintech-Complaint-Triage

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux

pip install -r requirements.txt
```

The trained DistilBERT model isn't tracked in the repo (~250 MB). You'll need to either:

- Download it separately and place it in `models/distilbert/`, or
- Retrain from scratch: `python src/models/train_distilbert.py` (needs a GPU — I used Colab)

## Running the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Swagger docs available at http://localhost:8000/docs

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My payment failed but the money was deducted from my account"}'
```

Response:

```json
{
  "category": "payment_issue",
  "confidence": 0.7649,
  "all_scores": {
    "account_issue": 0.0617,
    "billing_dispute": 0.0649,
    "fraud": 0.1085,
    "payment_issue": 0.7649
  }
}
```

## Running with Docker

```bash
docker-compose up --build
```

API available at http://localhost:8000.

The image is ~3.7 GB because of PyTorch. Torch is installed from the CPU-only index to avoid pulling CUDA libraries — saves about 2 GB.

## Stack

- Python 3.12
- PyTorch, HuggingFace Transformers
- scikit-learn (baseline)
- FastAPI, Uvicorn
- Docker

## Limitations

- Trained on a 27k sample, not the full CFPB dataset
- Label noise (~5–10%) from the CFPB `Issue` field
- Low-confidence predictions (< 0.6) should go to human review in production
- English only
- Not tested outside CFPB data

## TODO

- [ ] Relabel a subset with an LLM and see if F1 improves
- [ ] Add a confidence threshold and a "needs_review" flag
- [ ] Batch prediction endpoint
- [ ] ONNX export for faster inference
- [ ] Write tests for the API