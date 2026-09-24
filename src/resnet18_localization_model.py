import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 2
NUM_DEFECT_TYPES = 5

IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("RESNET18 LOCALIZATION MODEL")
print("=" * 60)

print("\nDevice:", device)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "defective",
    "good"
]

DEFECT_NAMES = [
    "scratch",
    "dent",
    "rust",
    "crack",
    "hole"
]


# ============================================================
# RESNET18 MULTI-TASK MODEL
# ============================================================

class ResNet18Localization(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Load pretrained ResNet18
        # ----------------------------------------------------

        weights = ResNet18_Weights.DEFAULT

        resnet = resnet18(weights=weights)

        # ----------------------------------------------------
        # Remove original ImageNet classifier
        # ----------------------------------------------------

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # ResNet18 produces 512 features
        self.feature_size = 512

        # ----------------------------------------------------
        # Classification Head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                self.feature_size,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

        # ----------------------------------------------------
        # Defect Type Head
        # ----------------------------------------------------

        self.defect_classifier = nn.Sequential(
            nn.Linear(
                self.feature_size,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                NUM_DEFECT_TYPES
            )
        )

        # ----------------------------------------------------
        # Bounding Box Head
        # ----------------------------------------------------

        self.bbox_regressor = nn.Sequential(
            nn.Linear(
                self.feature_size,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                4
            ),

            nn.Sigmoid()
        )

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):

        # Extract ResNet18 features
        features = self.backbone(x)

        # Flatten
        features = torch.flatten(
            features,
            start_dim=1
        )

        # Classification
        class_output = self.classifier(
            features
        )

        # Defect type
        defect_output = self.defect_classifier(
            features
        )

        # Bounding box
        bbox_output = self.bbox_regressor(
            features
        )

        return (
            class_output,
            defect_output,
            bbox_output
        )


# ============================================================
# CREATE MODEL
# ============================================================

print("\nLoading pretrained ResNet18...")

model = ResNet18Localization()

model = model.to(device)

print("Pretrained ResNet18 loaded successfully.")


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\nModel configuration:")

print("Backbone       : ResNet18")
print("Feature size   :", model.feature_size)

print("Classification  :", NUM_CLASSES)
print("Defect types    :", NUM_DEFECT_TYPES)
print("Bounding box    : 4 values")


# ============================================================
# TEST MODEL WITH RANDOM IMAGE
# ============================================================

print("\nTesting model with sample input...")

sample_input = torch.randn(
    1,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE
).to(device)


model.eval()

with torch.no_grad():

    class_output, defect_output, bbox_output = model(
        sample_input
    )


# ============================================================
# OUTPUT SHAPES
# ============================================================

print("\nOutput shapes:")

print(
    "Classification output:",
    class_output.shape
)

print(
    "Defect type output    :",
    defect_output.shape
)

print(
    "Bounding box output   :",
    bbox_output.shape
)


# ============================================================
# SAMPLE OUTPUTS
# ============================================================

print("\nSample outputs:")

print(
    "Classification logits:",
    class_output
)

print(
    "Defect type logits:",
    defect_output
)

print(
    "Bounding box:",
    bbox_output
)


# ============================================================
# VERIFY BOUNDING BOX RANGE
# ============================================================

bbox_min = bbox_output.min().item()
bbox_max = bbox_output.max().item()

print("\nBounding box range:")

print("Minimum:", bbox_min)
print("Maximum:", bbox_max)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)

if (
    class_output.shape == (1, NUM_CLASSES)
    and defect_output.shape == (1, NUM_DEFECT_TYPES)
    and bbox_output.shape == (1, 4)
    and 0 <= bbox_min <= 1
    and 0 <= bbox_max <= 1
):

    print("STATUS: RESNET18 LOCALIZATION MODEL READY")

else:

    print("STATUS: MODEL CHECK FAILED")

print("=" * 60)