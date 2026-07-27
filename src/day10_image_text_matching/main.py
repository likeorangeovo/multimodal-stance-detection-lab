"""Entry point for the Day10 image-text matching pipeline.

TODO:
1. Set seed and device.
2. Build vocab and synthetic samples.
3. Create Dataset/DataLoader.
4. Create ImageTextMatchingModel.
5. Train with scheduler, TensorBoard, and checkpoint saving.
6. Load best checkpoint.
7. Run positive and negative prediction demos.
"""
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from checkpoint import load_checkpoint, save_checkpoint
from dataset import ImageTextMatchDataset, build_samples, build_vocab
from eval import evaluate, predict_match
from model import ImageTextMatchingModel
from train import train_one_epoch


SEED = 42
BATCH_SIZE = 4
EPOCHS = 10
LR = 0.001
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best.pt"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = build_vocab()
    samples = build_samples()
    dataset = ImageTextMatchDataset(samples, vocab)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ImageTextMatchingModel(len(vocab)).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=LR,
    )

    best_acc = -1.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"train_loss: {train_loss:.4f} "
            f"train_acc: {train_acc:.4f} "
            f"val_loss: {val_loss:.4f} "
            f"val_acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                path=CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metric=best_acc,
            )
            print(f"Saved best checkpoint: {CHECKPOINT_PATH}")

    best_epoch, best_metric = load_checkpoint(
        path=CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
        device=device,
    )
    print(f"Loaded best checkpoint from epoch {best_epoch}, acc: {best_metric:.4f}")

    positive_image, positive_text, _ = dataset[0]
    negative_image, negative_text, _ = dataset[1]

    positive_prob = predict_match(model, positive_image, positive_text, device)
    negative_prob = predict_match(model, negative_image, negative_text, device)

    print(f"Positive demo match probability: {positive_prob:.4f}")
    print(f"Negative demo match probability: {negative_prob:.4f}")


if __name__ == "__main__":
    main()
