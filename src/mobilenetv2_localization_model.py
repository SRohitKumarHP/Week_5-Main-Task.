import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2SpatialLocalization(nn.Module):

    def __init__(self, num_classes=2, num_defect_types=5):

        super().__init__()

        # --------------------------------------------------
        # Load pretrained MobileNetV2
        # --------------------------------------------------

        weights = MobileNet_V2_Weights.DEFAULT

        backbone = mobilenet_v2(weights=weights)

        # Keep convolutional feature extractor
        self.features = backbone.features

        # MobileNetV2 produces 1280 feature channels
        feature_channels = 1280

        # --------------------------------------------------
        # Classification Head
        # Good / Defective
        # --------------------------------------------------

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # --------------------------------------------------
        # Defect Type Head
        # Scratch / Dent / Rust / Crack / Hole
        # --------------------------------------------------

        self.defect_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_defect_types)
        )

        # --------------------------------------------------
        # Bounding Box Localization Head
        #
        # Output:
        # x_center
        # y_center
        # width
        # height
        # --------------------------------------------------

        self.localization_head = nn.Sequential(

            nn.Conv2d(
                feature_channels,
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

            nn.AdaptiveAvgPool2d((4, 4)),

            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 256),

            nn.ReLU(),

            nn.Linear(256, 4),

            nn.Sigmoid()
        )


    def forward(self, x):

        # Extract spatial feature map
        features = self.features(x)

        # Classification
        class_output = self.classification_head(features)

        # Defect type
        defect_output = self.defect_head(features)

        # Bounding box
        bbox_output = self.localization_head(features)

        return (
            class_output,
            defect_output,
            bbox_output
        )


# ==========================================================
# MODEL TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MOBILENETV2 SPATIAL LOCALIZATION MODEL")
    print("=" * 70)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("Device:", device)

    # Create model
    model = MobileNetV2SpatialLocalization(
        num_classes=2,
        num_defect_types=5
    )

    model = model.to(device)

    print()
    print("Pretrained MobileNetV2 loaded successfully.")

    # --------------------------------------------------
    # Test input
    # --------------------------------------------------

    test_input = torch.randn(
        1,
        3,
        224,
        224
    ).to(device)

    # Forward pass
    with torch.no_grad():

        class_output, defect_output, bbox_output = model(
            test_input
        )

    # --------------------------------------------------
    # Display output shapes
    # --------------------------------------------------

    print()
    print("Input shape:", test_input.shape)

    print(
        "Classification output:",
        class_output.shape
    )

    print(
        "Defect type output:",
        defect_output.shape
    )

    print(
        "Bounding box output:",
        bbox_output.shape
    )

    # --------------------------------------------------
    # Parameter count
    # --------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print("Total parameters:", f"{total_parameters:,}")

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print()
    print("STATUS: MOBILENETV2 LOCALIZATION MODEL READY")
    print("=" * 70)