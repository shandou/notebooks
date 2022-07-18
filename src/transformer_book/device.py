import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def display_device_information():
    gpu_is_available: bool = torch.cuda.is_available()
    if gpu_is_available:
        gpu_count: int = torch.cuda.device_count()
        current_device: int = torch.cuda.current_device()
        gpu_name: str = torch.cuda.get_device_name(current_device)
        print(
            f"GPU information:\n - Number of devicies = {gpu_count}\n"
            f" - Selected GPU name is {gpu_name}"
        )
    else:
        print("GPU is not available")


if __name__ == "__main__":
    display_device_information()
