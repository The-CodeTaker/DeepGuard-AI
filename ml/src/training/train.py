import os

# Cap OpenBLAS / OMP thread pools before torch/numpy touch them.
# Leaves more RAM headroom and avoids the
# "OpenBLAS error: Memory allocation still failed after 10 retries" crash.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import gc
from pathlib import Path
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# FACES_DIR = PROJECT_ROOT / "data" / "processed" / "faces"
FACES_DIR = Path(r"D:\deepguard_faces")
MODEL_DIR = PROJECT_ROOT / "ml" / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
MAX_FACES_PER_VIDEO = 12
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-4

# Lowered from 6. With three DataLoaders and (previously) all of them set
# to persistent_workers=True, num_workers=6 meant 18 spawned processes
# alive at once on a 15GB machine -- each one re-importing torch/numpy/PIL.
# 4 workers on just the train loader is enough to keep the GPU fed.
NUM_WORKERS = 4

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


class FaceDataset(Dataset):
    def __init__(self, root, train=False):
        self.root = Path(root)
        self.train = train
        self.samples = []

        categories = {
            "original": 0,
            "Deepfakes": 1,
            "Face2Face": 1,
            "FaceSwap": 1,
            "NeuralTextures": 1,
        }

        for category, label in categories.items():
            category_dir = self.root / category

            if not category_dir.exists():
                continue

            for video_dir in sorted(category_dir.iterdir()):
                if not video_dir.is_dir():
                    continue

                images = sorted(video_dir.glob("*.jpg"))

                if not images:
                    continue

                # Fixed number of evenly distributed frames per video.
                if len(images) > MAX_FACES_PER_VIDEO:
                    indices = torch.linspace(
                        0,
                        len(images) - 1,
                        MAX_FACES_PER_VIDEO,
                    ).long().tolist()

                    images = [images[i] for i in indices]

                for image in images:
                    self.samples.append((image, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.train:
            image = train_transform(image)
        else:
            image = eval_transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    result = metrics(y_true, y_pred)
    result["loss"] = total_loss / len(loader.dataset)

    return result, confusion_matrix(y_true, y_pred)


def main():
    print("=" * 70)
    print("DeepGuard AI - Controlled EfficientNet-B0 Training")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = FaceDataset(
        FACES_DIR / "train",
        train=True,
    )

    val_dataset = FaceDataset(
        FACES_DIR / "val",
        train=False,
    )

    test_dataset = FaceDataset(
        FACES_DIR / "test",
        train=False,
    )

    def count(dataset):
        real = sum(label == 0 for _, label in dataset.samples)
        fake = sum(label == 1 for _, label in dataset.samples)
        return real, fake

    train_real, train_fake = count(train_dataset)
    val_real, val_fake = count(val_dataset)
    test_real, test_fake = count(test_dataset)

    print("\nDataset:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")

    print(f"train: real={train_real}, fake={train_fake}")
    print(f"val  : real={val_real}, fake={val_fake}")
    print(f"test : real={test_real}, fake={test_fake}")

    # NUM_WORKERS > 0 spins up background processes to decode/augment
    # images in parallel while the GPU trains on the previous batch.
    # With num_workers=0 all that work ran on the main process, so the
    # GPU sat mostly idle (13% util) waiting on the CPU each batch.
    #
    # persistent_workers=True is only set on the TRAIN loader now. It's
    # reused every epoch, so keeping its workers alive avoids re-spawning
    # them each time. The val/test loaders are only iterated once per
    # epoch (val) or once total (test) -- persistent workers there just
    # meant 12 extra processes sitting idle in RAM for the whole run,
    # which is what pushed memory to the edge and crashed OpenBLAS right
    # as the test loader tried to spin up its own workers on top.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    weights = models.EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 2),
    )

    model = model.to(DEVICE)

    # --- Class weighting ---
    # Your training set is ~4x more fake than real (e.g. real=7995,
    # fake=31962), so the model previously had no reason not to lean
    # toward predicting "fake" when unsure -- that's why real-video
    # recall (85.3%) lagged behind fake-video recall (95.3%) last run.
    # Weighting the loss inversely to class frequency makes mistakes
    # on the minority "real" class cost more, pushing the model to
    # actually learn to distinguish real videos rather than default
    # to the majority class.
    class_counts = torch.tensor([train_real, train_fake], dtype=torch.float)
    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights = class_weights.to(DEVICE)
    print(f"\nClass weights (real, fake): {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_f1 = -1.0
    best_epoch = -1
    history = []

    model_path = MODEL_DIR / "deepguard_efficientnet_b0.pth"

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        y_true = []
        y_pred = []

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

        scheduler.step()

        train_metrics = metrics(y_true, y_pred)
        train_metrics["loss"] = (
            running_loss / len(train_loader.dataset)
        )

        val_metrics, val_cm = evaluate(
            model,
            val_loader,
            criterion,
        )

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        print(
            f"Train | Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"F1: {train_metrics['f1']:.4f}"
        )

        print(
            f"Val   | Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch + 1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "efficientnet_b0",
                    "num_classes": 2,
                    "best_val_f1": best_f1,
                },
                model_path,
            )

            print("✓ Best model saved")

    print(f"\nBest epoch: {best_epoch}/{EPOCHS} (val F1: {best_f1:.4f})")

    # Drop the train/val loaders (and their worker processes) before the
    # final test pass so the test loader's own workers aren't spinning up
    # on top of 8 (train + val) that are still resident in RAM.
    del train_loader, val_loader
    gc.collect()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_cm = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1       : {test_metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(test_cm)

    with (MODEL_DIR / "training_history.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history, f, indent=2)

    with (MODEL_DIR / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                **test_metrics,
                "confusion_matrix": test_cm.tolist(),
            },
            f,
            indent=2,
        )

    print(f"\nSaved model:\n{model_path}")


if __name__ == "__main__":
    main()