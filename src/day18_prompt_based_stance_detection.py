"""
Day18: prompt-based stance detection with a simple masked template.

Idea:
Target + Tweet -> prompt with one mask token -> DistilRoBERTa masked language model

The model does not have a task-specific classification head here. Instead, we
ask it to fill the mask with a small set of stance words:

AGAINST -> "against"
FAVOR   -> "favor"
NONE    -> "neutral"
"""

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "semeval2016_task6"
TEST_PATH = DATA_DIR / "testdata-gold" / "SemEval2016-Task6-subtaskA-testdata-gold.txt"

DEFAULT_MODEL_NAME = "distilbert/distilroberta-base"
MAX_LENGTH = 160
BATCH_SIZE = 16

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
label_words = {
    "AGAINST": "against",
    "FAVOR": "favor",
    "NONE": "neutral",
}


def read_data(path):
    with path.open("r", encoding="latin-1", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_prompt(row, mask_token):
    target = row["Target"]
    tweet = row["Tweet"]
    return (
        f"Tweet: {tweet}\n"
        f"Target: {target}\n"
        f"The stance toward the target is {mask_token}."
    )


def get_verbalizer_token_ids(tokenizer):
    verbalizer_token_ids = {}

    for label, word in label_words.items():
        token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f'Verbalizer "{word}" for {label} is split into tokens: {token_ids}. '
                "Choose a single-token word for this tokenizer."
            )
        verbalizer_token_ids[label] = token_ids[0]

    return verbalizer_token_ids


def f1_score_for_label(gold, pred, label_id):
    tp = sum(g == label_id and p == label_id for g, p in zip(gold, pred))
    fp = sum(g != label_id and p == label_id for g, p in zip(gold, pred))
    fn = sum(g == label_id and p != label_id for g, p in zip(gold, pred))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def predict_batch(model, tokenizer, prompts, verbalizer_token_ids, device, max_length):
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    mask_positions = encoded["input_ids"].eq(tokenizer.mask_token_id)
    if not torch.all(mask_positions.sum(dim=1) == 1):
        raise ValueError("Every prompt must contain exactly one mask token.")

    with torch.no_grad():
        outputs = model(**encoded)

    batch_indices = torch.arange(encoded["input_ids"].size(0), device=device)
    mask_indices = mask_positions.long().argmax(dim=1)
    mask_logits = outputs.logits[batch_indices, mask_indices, :]

    label_scores = []
    label_names = list(verbalizer_token_ids.keys())
    for label in label_names:
        token_id = verbalizer_token_ids[label]
        label_scores.append(mask_logits[:, token_id])

    scores = torch.stack(label_scores, dim=1)
    pred_indices = torch.argmax(scores, dim=1).cpu().tolist()
    return [label2id[label_names[index]] for index in pred_indices], scores.cpu()


def evaluate(model, tokenizer, rows, device, batch_size, max_length):
    verbalizer_token_ids = get_verbalizer_token_ids(tokenizer)
    label_names = list(verbalizer_token_ids.keys())
    gold_labels = [label2id[row["Stance"]] for row in rows]
    pred_labels = []
    score_examples = []
    wrong_examples = []

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        prompts = [build_prompt(row, tokenizer.mask_token) for row in batch_rows]
        batch_preds, batch_scores = predict_batch(
            model,
            tokenizer,
            prompts,
            verbalizer_token_ids,
            device,
            max_length,
        )
        pred_labels.extend(batch_preds)

        if len(score_examples) < 3:
            for row, pred, scores in zip(batch_rows, batch_preds, batch_scores):
                score_examples.append({
                    "target": row["Target"],
                    "tweet": row["Tweet"],
                    "gold": row["Stance"],
                    "pred": id2label[pred],
                    "scores": {
                        label: float(score)
                        for label, score in zip(label_names, scores.tolist())
                    },
                })
                if len(score_examples) == 3:
                    break

    correct = sum(g == p for g, p in zip(gold_labels, pred_labels))
    accuracy = correct / len(gold_labels)

    f1_against = f1_score_for_label(gold_labels, pred_labels, label2id["AGAINST"])
    f1_favor = f1_score_for_label(gold_labels, pred_labels, label2id["FAVOR"])
    f1_none = f1_score_for_label(gold_labels, pred_labels, label2id["NONE"])
    macro_f1 = (f1_against + f1_favor + f1_none) / 3
    semeval_f1 = (f1_against + f1_favor) / 2

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
        "score_examples": score_examples,
        "wrong_examples": wrong_examples,
    }


def print_prompt_demo(tokenizer, rows):
    print("\nPrompt template demo:")
    for row in rows[:2]:
        print("-" * 60)
        print(build_prompt(row, tokenizer.mask_token))
        print("Gold:", row["Stance"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Masked language model checkpoint used after prompt construction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of test examples to evaluate. Use 0 for the full test set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for prompt inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Maximum token length for the prompt.",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Print label-word scores for the first few examples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_data(TEST_PATH)
    if args.limit > 0:
        rows = rows[:args.limit]

    print("Day18 prompt-based stance detection")
    print("model:", args.model_name)
    print("device:", device)
    print("test samples:", len(rows))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    print_prompt_demo(tokenizer, rows)

    metrics = evaluate(
        model,
        tokenizer,
        rows,
        device,
        args.batch_size,
        args.max_length,
    )

    print("\nMetrics:")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 AGAINST: {metrics['f1_against']:.4f}")
    print(f"F1 FAVOR: {metrics['f1_favor']:.4f}")
    print(f"F1 NONE: {metrics['f1_none']:.4f}")
    print(f"macro F1: {metrics['macro_f1']:.4f}")
    print(f"SemEval F1: {metrics['semeval_f1']:.4f}")
    print(f"wrong examples: {len(metrics['wrong_examples'])}")

    if args.show_scores:
        print("\nLabel-word scores:")
        for example in metrics["score_examples"]:
            print("-" * 60)
            print("Target:", example["target"])
            print("Gold:", example["gold"], "| Pred:", example["pred"])
            for label, score in example["scores"].items():
                print(f"{label}: {score:.4f}")

    print("\nFirst wrong examples:")
    for example in metrics["wrong_examples"][:5]:
        print("-" * 60)
        print("Target:", example["target"])
        print("Tweet:", example["tweet"])
        print("Gold:", example["gold"], "| Pred:", example["pred"])

    print("\nDay18 prompt-based demo done.")


if __name__ == "__main__":
    main()
