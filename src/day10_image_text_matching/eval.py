"""Evaluation and prediction helpers for Day10.

TODO:
- accuracy_from_logits()
- evaluate()
- predict_match()
"""
import torch
from torch import nn


def accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct, total


def evaluate(model, dataloader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, texts, labels in dataloader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            logits = model(images, texts)
            loss = loss_fn(logits, labels)

            batch_size = images.size(0)
            correct, total = accuracy_from_logits(logits, labels)

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += total

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def predict_match(model, image_tensor, text_tensor, device):
    model.eval()

    image_tensor = image_tensor.unsqueeze(0).to(device)
    text_tensor = text_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(image_tensor, text_tensor)
        prob = torch.sigmoid(logit)

    return prob.item()
