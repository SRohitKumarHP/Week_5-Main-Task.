import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD PRETRAINED RESNET18
# ============================================================

print("=" * 70)
print("             RESNET18 MODEL SETUP")
print("=" * 70)

print()

print("Loading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)

print("Pretrained ResNet18 loaded successfully.")

print()


# ============================================================
# ORIGINAL CLASSIFIER
# ============================================================

print("Original classifier:")

print(model.fc)

print()


# ============================================================
# NUMBER OF INPUT FEATURES
# ============================================================

input_features = model.fc.in_features

print(
    "Classifier input features:",
    input_features
)

print()


# ============================================================
# REPLACE FINAL CLASSIFIER
# ============================================================

number_of_classes = 2

model.fc = nn.Linear(
    input_features,
    number_of_classes
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(DEVICE)


# ============================================================
# PRINT NEW CLASSIFIER
# ============================================================

print("New classifier:")

print(model.fc)

print()


# ============================================================
# DEVICE INFORMATION
# ============================================================

print("Device:")

print(DEVICE)

print()


# ============================================================
# TEST MODEL WITH RANDOM IMAGE TENSOR
# ============================================================

print("=" * 70)
print("TESTING MODEL")
print("=" * 70)

print()

# Create a fake batch:
# 1 image
# 3 RGB channels
# 224 x 224 pixels

test_input = torch.randn(
    1,
    3,
    224,
    224
).to(DEVICE)


# Model prediction

with torch.no_grad():

    output = model(
        test_input
    )


print("Input shape:")

print(test_input.shape)

print()

print("Output shape:")

print(output.shape)

print()

print("Raw model output:")

print(output)

print()


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("RESNET18 MODEL SETUP COMPLETED")
print("=" * 70)