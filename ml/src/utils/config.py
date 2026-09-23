from pathlib import Path
import torch


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ML directory
ML_DIR = PROJECT_ROOT / "ml"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info() -> dict:
    """Return information about the available compute device."""

    info = {
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
            }
        )

    return info