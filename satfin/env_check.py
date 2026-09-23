"""Print environment info and pick the compute device: `python -m satfin.env_check`."""
import platform
import sys

import torch


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    print(f"Python   {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print(f"PyTorch  {torch.__version__}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}    {p.name}, {p.total_memory / 2**30:.1f} GB, CUDA {torch.version.cuda}")
    else:
        print("GPU      none found -> CPU mode (use tiny settings)")
    print(f"Device   {get_device()}")


if __name__ == "__main__":
    main()
