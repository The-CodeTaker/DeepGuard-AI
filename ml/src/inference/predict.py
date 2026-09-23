from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "deepguard_efficientnet_b0.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

_model = None


def load_model():
    global _model

    if _model is not None:
        return _model

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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    _model = model

    return _model


def extract_features(image_path):
    """Return the 1280-dimensional EfficientNet-B0 feature vector."""
    model = load_model()

    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model.features(tensor)
        pooled_features = model.avgpool(features)
        embedding = torch.flatten(pooled_features, start_dim=1)[0]

    return embedding.detach().cpu().numpy().astype("float32")


def predict_image(image_path):
    model = load_model()

    image = Image.open(image_path).convert("RGB")

    tensor = _transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

    real_probability = float(
        probabilities[0].item()
    )

    fake_probability = float(
        probabilities[1].item()
    )

    if fake_probability >= real_probability:
        label = "FAKE"
        confidence = fake_probability
    else:
        label = "REAL"
        confidence = real_probability

    return {
        "label": label,
        "confidence": confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "model": "efficientnet-b0",
        "device": str(DEVICE),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python -m "
            "ml.src.inference.predict <image_path>"
        )
        raise SystemExit(1)

    result = predict_image(sys.argv[1])

    print("=" * 40)
    print("DeepGuard AI")
    print("=" * 40)
    print(f"Assessment : {result['label']}")
    print(
        f"Confidence : "
        f"{result['confidence']:.2%}"
    )
    print(
        f"Real       : "
        f"{result['real_probability']:.2%}"
    )
    print(
        f"Fake       : "
        f"{result['fake_probability']:.2%}"
    )
    print(
        "\nModel: "
        f"{result['model']}"
    )