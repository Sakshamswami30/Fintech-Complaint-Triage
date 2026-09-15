# train_distilbert.py
# Fine-tunes DistilBERT for 4-class complaint classification.
# CPU-friendly: small train set, short sequences, 3 epochs.
# Run: python src/models/train_distilbert.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

DATA_PATH = "data/processed/complaints_clean.csv"
MODEL_DIR = "models/distilbert"
SEED = 42
TEST_SIZE = 0.2
MAX_LEN = 96
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-5
SAMPLE_SIZE = 4000

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"loading {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"rows: {len(df):,}")

labels = sorted(df["category"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
print(f"labels: {label2id}")

df["label"] = df["category"].map(label2id)

train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
)

# small train set — faster epochs beat more data on CPU
if len(train_df) > SAMPLE_SIZE:
    train_df = train_df.sample(n=SAMPLE_SIZE, random_state=SEED)

print(f"train: {len(train_df):,}  test: {len(test_df):,}")

print("\nloading tokenizer")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors=None,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


train_ds = ComplaintDataset(train_df["clean_text"], train_df["label"], tokenizer, MAX_LEN)
test_ds = ComplaintDataset(test_df["clean_text"], test_df["label"], tokenizer, MAX_LEN)

print("loading model")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro")
    acc = (preds == labels).mean()
    return {"accuracy": acc, "macro_f1": f1}


args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=25,
    seed=SEED,
    report_to="none",
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

print("\ntraining...")
trainer.train()

print("\nevaluating on test set")
preds = trainer.predict(test_ds)
y_pred = np.argmax(preds.predictions, axis=-1)
y_true = preds.label_ids

print("\n" + "=" * 60)
print(classification_report(y_true, y_pred, target_names=labels))
print(confusion_matrix(y_true, y_pred))

os.makedirs(MODEL_DIR, exist_ok=True)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"\nsaved to {MODEL_DIR}")