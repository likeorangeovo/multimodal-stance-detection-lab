"""Training loop for Day10.

TODO:
- train_one_epoch()
- train()
- StepLR scheduler
- TensorBoard logging
- best checkpoint saving
"""
import torch
from torch import nn


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for image, text, label in data_loader:
        image = image.to(device)
        text = text.to(device)
        label = label.to(device)

        logits = model(image,text)
        loss = loss_fn(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * image.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_correct += (preds == label).sum().item()
        total_samples += image.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train(model, train_loader, optimizer, device, epochs=10):
    for epoch in range(epochs):
        loss, accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"loss: {loss:.4f} "
            f"acc: {accuracy:.4f}"
        )
