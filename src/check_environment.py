import sys
import torch
import torchvision


print("=" * 60)
print("       METAL DEFECT DETECTION")
print("       ENVIRONMENT CHECK")
print("=" * 60)

print()

# Python version
print("Python Version     :", sys.version)

# PyTorch version
print("PyTorch Version    :", torch.__version__)

# Torchvision version
print("Torchvision Version:", torchvision.__version__)

print()

# Check CUDA
print("CUDA Available     :", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU                :", torch.cuda.get_device_name(0))

else:

    print("GPU                : Not Available")
    print("Training will use CPU.")

print()

print("=" * 60)
print("Environment check completed.")
print("=" * 60)