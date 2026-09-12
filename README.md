# FinTech Complaint Triage

End-to-end NLP system that classifies fintech customer complaints into 4 categories and routes them to the appropriate team with priority and SLA.

## Problem
Fintech companies receive thousands of complaints daily. Manual triage is slow, error-prone, and regulated. Misrouting a fraud dispute leads to fines and customer loss.

## Solution
An NLP model that reads complaint text and outputs:
- **Category:** `payment_failure` | `fraud_dispute` | `refund_request` | `account_issue`
- **Confidence:** 0–1
- **Priority:** P0 (fraud) to P3 (general)
- **Routing:** Auto-assigned team + SLA

## Status
🚧 In Progress — Phase 0 (Setup)

## Stack
- Python 3.11
- scikit-learn, PyTorch, HuggingFace Transformers (planned)
- FastAPI, Docker (planned)
- MLflow, Evidently (planned)

## Project Structure
...
