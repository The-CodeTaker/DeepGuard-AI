import torch

from ml.src.utils.config import DEVICE, get_device_info


def main():
    print("=" * 50)
    print("DeepGuard AI - GPU Diagnostic")
    print("=" * 50)

    info = get_device_info()

    for key, value in info.items():
        print(f"{key}: {value}")

    print("\nRunning GPU tensor computation...")

    x = torch.randn(2048, 2048, device=DEVICE)
    y = torch.randn(2048, 2048, device=DEVICE)

    result = x @ y

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    print(f"Tensor device: {result.device}")
    print(f"Tensor shape: {result.shape}")

    if DEVICE.type == "cuda":
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    print("\nGPU diagnostic completed successfully.")


if __name__ == "__main__":
    main()