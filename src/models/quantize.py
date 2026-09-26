# quantize.py
# Converts the DistilBERT model to float16 to reduce memory usage.
# Run: python src/models/quantize.py

import os
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

SRC = "Saksham-30/fintech-distilbert"
DST = "models/distilbert-fp16"

print(f"loading {SRC}")
model = DistilBertForSequenceClassification.from_pretrained(SRC)
tokenizer = DistilBertTokenizerFast.from_pretrained(SRC)

print("converting to float16...")
model = model.half()

print(f"saving to {DST}")
os.makedirs(DST, exist_ok=True)
model.save_pretrained(DST)
tokenizer.save_pretrained(DST)

total = sum(
    os.path.getsize(os.path.join(DST, f))
    for f in os.listdir(DST)
)
print(f"done. total size: {total / 1024 / 1024:.1f} MB")