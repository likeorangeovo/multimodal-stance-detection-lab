"""
Day14 demo: load a tiny BERT classifier and run one forward pass.

This is only for learning the HuggingFace sequence classification flow.
Day15 will add real fine-tuning and F1 evaluation.
"""

import csv
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "semeval2016_task6" / "semeval2016-task6-trainingdata.txt"

MODEL_NAME = "prajjwal1/bert-tiny"

label2id = {
    "AGAINST": 0,
    "FAVOR": 1,
    "NONE": 2,
}
id2label = {
    0: "AGAINST",
    1: "FAVOR",
    2: "NONE",
}


# 1. Read a few SemEval samples.
with TRAIN_PATH.open("r", encoding="latin-1", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

samples = rows[:4]

print("Sample data:")
for row in samples:
    print(row["Target"], "|", row["Tweet"][:70], "...", "|", row["Stance"])


# 2. Load tokenizer and BERT classification model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    label2id=label2id,
    id2label=id2label,
)


# 3. Convert target + tweet into BERT text-pair inputs.
targets = [row["Target"] for row in samples]
tweets = [row["Tweet"] for row in samples]
labels = torch.tensor([label2id[row["Stance"]] for row in samples])

batch = tokenizer(
    targets,
    tweets,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

print("\nBERT input shapes:")
print("input_ids:", batch["input_ids"].shape)
print("attention_mask:", batch["attention_mask"].shape)
print("token_type_ids:", batch["token_type_ids"].shape)


# 4. Run one forward pass. The model is not fine-tuned yet.
model.eval()
with torch.no_grad():
    outputs = model(**batch, labels=labels)
    probs = torch.softmax(outputs.logits, dim=1)
    preds = torch.argmax(probs, dim=1)

print("\nForward pass:")
print("loss:", outputs.loss.item())
print("logits shape:", outputs.logits.shape)

print("\nPredictions before fine-tuning:")
for row, pred_id in zip(samples, preds.tolist()):
    print("gold:", row["Stance"], "| pred:", id2label[pred_id])

print("\nNote: predictions are random-ish because the classifier head is not trained yet.")
