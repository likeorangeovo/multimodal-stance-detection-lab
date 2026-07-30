"""
Day17: swap the encoder to DistilRoBERTa for a quick stance classification demo.

The training/evaluation flow is the same as Day15:
Target + Tweet -> encoder -> AGAINST / FAVOR / NONE
"""

import csv
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "semeval2016_task6"
TRAIN_PATH = DATA_DIR / "semeval2016-task6-trainingdata.txt"
TEST_PATH = DATA_DIR / "testdata-gold" / "SemEval2016-Task6-subtaskA-testdata-gold.txt"

MODEL_NAME = "distilbert/distilroberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5

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


def read_data(path):
    with path.open("r", encoding="latin-1", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_dataset(rows, tokenizer):
    targets = [row["Target"] for row in rows]
    tweets = [row["Tweet"] for row in rows]
    labels = torch.tensor([label2id[row["Stance"]] for row in rows])

    encoded = tokenizer(
        targets,
        tweets,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        labels,
    )


def f1_score_for_label(gold, pred, label_id):
    tp = sum(g == label_id and p == label_id for g, p in zip(gold, pred))
    fp = sum(g != label_id and p == label_id for g, p in zip(gold, pred))
    fn = sum(g == label_id and p != label_id for g, p in zip(gold, pred))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return f1


def evaluate(model, dataloader, device, rows=None):
    model.eval()
    gold_labels = []
    pred_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

            gold_labels.extend(labels.tolist())
            pred_labels.extend(preds)

    correct = sum(g == p for g, p in zip(gold_labels, pred_labels))
    accuracy = correct / len(gold_labels)

    f1_against = f1_score_for_label(gold_labels, pred_labels, label2id["AGAINST"])
    f1_favor = f1_score_for_label(gold_labels, pred_labels, label2id["FAVOR"])
    f1_none = f1_score_for_label(gold_labels, pred_labels, label2id["NONE"])
    macro_f1 = (f1_against + f1_favor + f1_none) / 3
    semeval_f1 = (f1_against + f1_favor) / 2

    wrong_examples = []

    if rows is not None:
        for row, gold, pred in zip(rows, gold_labels, pred_labels):
            if gold != pred:
                wrong_examples.append({
                    "target": row["Target"],
                    "tweet": row["Tweet"],
                    "gold": id2label[gold],
                    "pred": id2label[pred],
                })

    return {
        "accuracy": accuracy,
        "f1_against": f1_against,
        "f1_favor": f1_favor,
        "f1_none": f1_none,
        "macro_f1": macro_f1,
        "semeval_f1": semeval_f1,
        "wrong_examples": wrong_examples,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("model:", MODEL_NAME)
    print("device:", device)

    train_rows = read_data(TRAIN_PATH)
    test_rows = read_data(TEST_PATH)
    print("train samples:", len(train_rows))
    print("test samples:", len(test_rows))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = build_dataset(train_rows, tokenizer)
    test_dataset = build_dataset(test_rows, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        label2id=label2id,
        id2label=id2label,
    )
    model.to(device)

    train_label_ids = [label2id[row["Stance"]] for row in train_rows]
    counts = Counter(train_label_ids)
    class_weights = torch.tensor([
        len(train_label_ids) / (3 * counts[0]),
        len(train_label_ids) / (3 * counts[1]),
        len(train_label_ids) / (3 * counts[2]),
    ], dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        metrics = evaluate(model, test_loader, device, test_rows)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"train loss: {avg_loss:.4f}")
        print(f"accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 AGAINST: {metrics['f1_against']:.4f}")
        print(f"F1 FAVOR: {metrics['f1_favor']:.4f}")
        print(f"F1 NONE: {metrics['f1_none']:.4f}")
        print(f"macro F1: {metrics['macro_f1']:.4f}")
        print(f"SemEval F1: {metrics['semeval_f1']:.4f}")
        print(f"wrong examples: {len(metrics['wrong_examples'])}")

    print("\nDay17 model swap demo done.")


if __name__ == "__main__":
    main()
