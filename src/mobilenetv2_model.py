import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 2


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("MOBILENETV2 METAL DEFECT CLASSIFICATION MODEL")
print("=" * 70)

print()
print("Device:", device)


# ============================================================
# LOAD PRETRAINED MOBILENETV2
# ============================================================

print()
print("Loading pretrained MobileNetV2...")


weights = MobileNet_V2_Weights.DEFAULT


model = mobilenet_v2(
    weights=weights
)


print(
    "Pretrained MobileNetV2 loaded successfully."
)


# ============================================================
# ORIGINAL CLASSIFIER
# ============================================================

print()
print("Original classifier:")

print(
    model.classifier
)


# ============================================================
# GET CLASSIFIER INPUT FEATURES
# ============================================================

classifier_input_features = (
    model.classifier[1].in_features
)


print()
print(
    "Classifier input features:",
    classifier_input_features
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

model.classifier = nn.Sequential(

    nn.Dropout(
        p=0.2
    ),

    nn.Linear(
        classifier_input_features,
        NUM_CLASSES
    )
)


# ============================================================
# DISPLAY NEW CLASSIFIER
# ============================================================

print()
print("New classifier:")

print(
    model.classifier
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(device)


# ============================================================
# TEST INPUT
# ============================================================

print()
print("Testing model with sample input...")


sample_input = torch.randn(
    1,
    3,
    224,
    224
).to(device)


with torch.no_grad():

    output = model(
        sample_input
    )


# ============================================================
# OUTPUT
# ============================================================

print()
print(
    "Input shape :",
    sample_input.shape
)

print(
    "Output shape:",
    output.shape
)

print()
print(
    "Raw model output:"
)

print(
    output
)


# ============================================================
# PARAMETER COUNT
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print()
print("Model parameters:")

print(
    "Total parameters     :",
    total_parameters
)

print(
    "Trainable parameters :",
    trainable_parameters
)


# ============================================================
# STATUS
# ============================================================

print()
print("=" * 70)

print(
    "STATUS: MOBILENETV2 MODEL READY"
)

print("=" * 70)