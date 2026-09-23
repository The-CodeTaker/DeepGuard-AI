from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
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
FACES_DIR = PROJECT_ROOT / "data" / "processed" / "faces"
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "deepguard_efficientnet_b0.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def build_model():
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 2),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


def main():
    print("=" * 70)
    print("DeepGuard AI - Video-Level Evaluation")
    print("=" * 70)

    model = build_model()

    video_predictions = []

    for category, label in {
        "original": 0,
        "Deepfakes": 1,
        "Face2Face": 1,
        "FaceSwap": 1,
        "NeuralTextures": 1,
    }.items():

        category_dir = FACES_DIR / "test" / category

        if not category_dir.exists():
            continue

        for video_dir in sorted(category_dir.iterdir()):

            if not video_dir.is_dir():
                continue

            image_paths = sorted(video_dir.glob("*.jpg"))

            if not image_paths:
                continue

            probabilities = []

            with torch.no_grad():

                for image_path in image_paths:

                    image = Image.open(
                        image_path
                    ).convert("RGB")

                    image = transform(image)
                    image = image.unsqueeze(0).to(DEVICE)

                    output = model(image)

                    probability = torch.softmax(
                        output,
                        dim=1,
                    )[0, 1].item()

                    probabilities.append(probability)

            # Average frame-level fake probability.
            video_probability = sum(probabilities) / len(probabilities)

            prediction = int(video_probability >= 0.5)

            video_predictions.append({
                "video": f"{category}/{video_dir.name}",
                "true": label,
                "pred": prediction,
                "probability": video_probability,
                "frames": len(probabilities),
            })

    y_true = [
        item["true"]
        for item in video_predictions
    ]

    y_pred = [
        item["pred"]
        for item in video_predictions
    ]

    print(f"\nVideos evaluated: {len(video_predictions)}")

    print("\nVIDEO-LEVEL RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : "
        f"{accuracy_score(y_true, y_pred):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"F1       : "
        f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nPER-MANIPULATION RESULTS")
    print("=" * 70)

    categories = [
        "original",
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    ]

    for category in categories:

        items = [
            item
            for item in video_predictions
            if item["video"].startswith(category + "/")
        ]

        if not items:
            continue

        correct = sum(
            item["true"] == item["pred"]
            for item in items
        )

        accuracy = correct / len(items)

        print(
            f"{category:<20} "
            f"{correct}/{len(items)} "
            f"({accuracy:.2%})"
        )


if __name__ == "__main__":
    main()