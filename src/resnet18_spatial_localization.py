import os
import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


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
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("IMPROVED RESNET18 SPATIAL LOCALIZATION MODEL")
print("=" * 70)

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
# MODEL
# ============================================================

class ResNet18SpatialLocalization(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Load pretrained ResNet18
        # ----------------------------------------------------

        weights = ResNet18_Weights.DEFAULT

        resnet = resnet18(
            weights=weights
        )

        # ----------------------------------------------------
        # Keep convolutional layers
        #
        # We intentionally remove:
        # - avgpool
        # - fc
        #
        # because we want to preserve spatial information.
        # ----------------------------------------------------

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # ResNet18 final convolutional feature map:
        #
        # [batch, 512, 7, 7]
        #
        self.feature_channels = 512

        # ====================================================
        # CLASSIFICATION HEAD
        # ====================================================

        self.classifier_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                self.feature_channels,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

        # ====================================================
        # DEFECT TYPE HEAD
        # ====================================================

        self.defect_classifier_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.defect_classifier = nn.Sequential(

            nn.Linear(
                self.feature_channels,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                NUM_DEFECT_TYPES
            )
        )

        # ====================================================
        # SPATIAL LOCALIZATION HEAD
        # ====================================================

        self.localization = nn.Sequential(

            nn.Conv2d(
                self.feature_channels,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (4, 4)
            )
        )

        # 128 × 4 × 4 = 2048
        localization_features = (
            128 * 4 * 4
        )

        self.bbox_regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                localization_features,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                4
            ),

            nn.Sigmoid()
        )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Extract spatial feature map
        # ----------------------------------------------------

        features = self.backbone(x)

        # Expected:
        #
        # [batch, 512, 7, 7]
        #

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        class_features = self.classifier_pool(
            features
        )

        class_features = torch.flatten(
            class_features,
            start_dim=1
        )

        class_output = self.classifier(
            class_features
        )

        # ====================================================
        # DEFECT TYPE
        # ====================================================

        defect_features = self.defect_classifier_pool(
            features
        )

        defect_features = torch.flatten(
            defect_features,
            start_dim=1
        )

        defect_output = self.defect_classifier(
            defect_features
        )

        # ====================================================
        # BOUNDING BOX
        # ====================================================

        localization_features = self.localization(
            features
        )

        bbox_output = self.bbox_regressor(
            localization_features
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

model = ResNet18SpatialLocalization()

model = model.to(device)

print(
    "Pretrained ResNet18 loaded successfully."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\nModel configuration:")

print(
    "Backbone          : ResNet18"
)

print(
    "Backbone output   : [512, 7, 7]"
)

print(
    "Classification    : 2 classes"
)

print(
    "Defect types      : 5 classes"
)

print(
    "Bounding box      : 4 values"
)


# ============================================================
# TEST INPUT
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

    # Get spatial features
    spatial_features = model.backbone(
        sample_input
    )

    (
        class_output,
        defect_output,
        bbox_output
    ) = model(
        sample_input
    )


# ============================================================
# PRINT FEATURE SHAPE
# ============================================================

print("\nSpatial feature map shape:")

print(
    spatial_features.shape
)


# ============================================================
# PRINT OUTPUT SHAPES
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
# SAMPLE OUTPUT
# ============================================================

print("\nSample outputs:")

print(
    "Classification logits:"
)

print(
    class_output
)


print(
    "\nDefect type logits:"
)

print(
    defect_output
)


print(
    "\nBounding box:"
)

print(
    bbox_output
)


# ============================================================
# VERIFY BOUNDING BOX
# ============================================================

bbox_min = bbox_output.min().item()

bbox_max = bbox_output.max().item()


print("\nBounding box range:")

print(
    "Minimum:",
    bbox_min
)

print(
    "Maximum:",
    bbox_max
)


# ============================================================
# PARAMETER COUNT
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print("\nModel parameters:")

print(
    "Total parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)


if (
    spatial_features.shape
    == (1, 512, 7, 7)

    and class_output.shape
    == (1, NUM_CLASSES)

    and defect_output.shape
    == (1, NUM_DEFECT_TYPES)

    and bbox_output.shape
    == (1, 4)

    and bbox_min >= 0

    and bbox_max <= 1
):

    print(
        "STATUS: IMPROVED RESNET18 "
        "SPATIAL LOCALIZATION MODEL READY"
    )

else:

    print(
        "STATUS: MODEL CHECK FAILED"
    )


print("=" * 70)