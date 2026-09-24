import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms
from torchvision.models import mobilenet_v2


# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_DIRECTORY = "input"

OUTPUT_DIRECTORY = "output"

MODEL_PATH = (
    "models/"
    "mobilenetv2_spatial_localization.pth"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ==========================================================
# MODEL
# ==========================================================

class MobileNetV2SpatialLocalization(
    nn.Module
):

    def __init__(
        self,
        num_classes=2,
        num_defect_types=5
    ):

        super().__init__()

        # No need to download pretrained weights here.
        # The trained checkpoint will provide the weights.

        backbone = mobilenet_v2(
            weights=None
        )

        self.features = backbone.features

        feature_channels = 1280

        # --------------------------------------------------
        # Classification Head
        # --------------------------------------------------

        self.classification_head = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),

            nn.Flatten(),

            nn.Linear(
                feature_channels,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                num_classes
            )
        )

        # --------------------------------------------------
        # Defect Type Head
        # --------------------------------------------------

        self.defect_head = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),

            nn.Flatten(),

            nn.Linear(
                feature_channels,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                num_defect_types
            )
        )

        # --------------------------------------------------
        # Bounding Box Head
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

            nn.AdaptiveAvgPool2d(
                (4, 4)
            ),

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                4
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        features = self.features(x)

        class_output = (
            self.classification_head(
                features
            )
        )

        defect_output = (
            self.defect_head(
                features
            )
        )

        bbox_output = (
            self.localization_head(
                features
            )
        )

        return (
            class_output,
            defect_output,
            bbox_output
        )


# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 70)
print(
    "MOBILENETV2 METAL DEFECT INSPECTION"
)
print("=" * 70)

print()
print(
    "Device:",
    DEVICE
)

print()
print(
    "Loading model..."
)

model = MobileNetV2SpatialLocalization(
    num_classes=2,
    num_defect_types=5
)

model = model.to(
    DEVICE
)


if not os.path.exists(
    MODEL_PATH
):

    print()
    print(
        "ERROR: Trained model not found."
    )

    print(
        "Expected model:"
    )

    print(
        os.path.abspath(
            MODEL_PATH
        )
    )

    raise SystemExit


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if (
    isinstance(
        checkpoint,
        dict
    )
    and
    "model_state_dict"
    in checkpoint
):

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()

print(
    "Model loaded successfully."
)


# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


# ==========================================================
# FIND INPUT IMAGES
# ==========================================================

image_files = []

for filename in sorted(
    os.listdir(
        INPUT_DIRECTORY
    )
):

    if filename.lower().endswith(
        SUPPORTED_EXTENSIONS
    ):

        image_files.append(
            filename
        )


print()
print(
    "Input directory:"
)

print(
    os.path.abspath(
        INPUT_DIRECTORY
    )
)

print()
print(
    "Images found:",
    len(image_files)
)


if len(image_files) == 0:

    print()
    print(
        "No images found in input directory."
    )

    print(
        "Put JPG, JPEG, PNG, BMP or WEBP images "
        "inside the input folder."
    )

    raise SystemExit


# ==========================================================
# SUMMARY VARIABLES
# ==========================================================

total_images = 0

good_count = 0

defective_count = 0

defect_counts = {
    "scratch": 0,
    "dent": 0,
    "rust": 0,
    "crack": 0,
    "hole": 0
}


# ==========================================================
# PROCESS IMAGES
# ==========================================================

print()
print("=" * 70)
print(
    "PROCESSING IMAGES"
)
print("=" * 70)


for image_filename in image_files:

    image_path = os.path.join(
        INPUT_DIRECTORY,
        image_filename
    )

    print()
    print("-" * 70)

    print(
        "Image:",
        image_filename
    )

    # ------------------------------------------------------
    # Read original image
    # ------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "ERROR: Unable to read image."
        )

        continue


    original_image = image.copy()

    image_height, image_width = (
        image.shape[:2]
    )

    print(
        "Image size:",
        f"{image_width} x {image_height}"
    )

    # ------------------------------------------------------
    # Convert BGR -> RGB
    # ------------------------------------------------------

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ------------------------------------------------------
    # Prepare tensor
    # ------------------------------------------------------

    tensor = transform(
        rgb_image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    with torch.no_grad():

        (
            class_output,
            defect_output,
            bbox_output
        ) = model(tensor)

    # ------------------------------------------------------
    # Classification probabilities
    # ------------------------------------------------------

    class_probabilities = torch.softmax(
        class_output,
        dim=1
    )

    predicted_class = torch.argmax(
        class_probabilities,
        dim=1
    ).item()

    class_confidence = (
        class_probabilities[
            0,
            predicted_class
        ].item()
        * 100
    )

    # ------------------------------------------------------
    # GOOD IMAGE
    # ------------------------------------------------------

    if predicted_class == 1:

        good_count += 1

        label = (
            f"GOOD "
            f"({class_confidence:.2f}%)"
        )

        print()
        print(
            "Prediction: GOOD"
        )

        print(
            f"Confidence: "
            f"{class_confidence:.2f}%"
        )

        # --------------------------------------------------
        # Draw green-style inspection box around image
        # using default OpenCV color values.
        # --------------------------------------------------

        cv2.rectangle(
            original_image,
            (5, 5),
            (
                image_width - 5,
                image_height - 5
            ),
            (0, 255, 0),
            3
        )

        cv2.putText(
            original_image,
            label,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


    # ------------------------------------------------------
    # DEFECTIVE IMAGE
    # ------------------------------------------------------

    else:

        defective_count += 1

        # --------------------------------------------------
        # Defect type probabilities
        # --------------------------------------------------

        defect_probabilities = (
            torch.softmax(
                defect_output,
                dim=1
            )
        )

        predicted_defect = torch.argmax(
            defect_probabilities,
            dim=1
        ).item()

        defect_confidence = (
            defect_probabilities[
                0,
                predicted_defect
            ].item()
            * 100
        )

        defect_name = (
            DEFECT_NAMES[
                predicted_defect
            ]
        )

        defect_counts[
            defect_name
        ] += 1

        # --------------------------------------------------
        # Bounding box
        # --------------------------------------------------

        bbox = (
            bbox_output[
                0
            ]
            .cpu()
            .numpy()
        )

        x_center = float(
            bbox[0]
        )

        y_center = float(
            bbox[1]
        )

        box_width = float(
            bbox[2]
        )

        box_height = float(
            bbox[3]
        )

        # --------------------------------------------------
        # Convert normalized bbox to pixels
        # --------------------------------------------------

        x1 = int(
            (
                x_center
                - box_width / 2
            )
            * image_width
        )

        y1 = int(
            (
                y_center
                - box_height / 2
            )
            * image_height
        )

        x2 = int(
            (
                x_center
                + box_width / 2
            )
            * image_width
        )

        y2 = int(
            (
                y_center
                + box_height / 2
            )
            * image_height
        )

        # --------------------------------------------------
        # Clamp coordinates
        # --------------------------------------------------

        x1 = max(
            0,
            min(
                x1,
                image_width - 1
            )
        )

        y1 = max(
            0,
            min(
                y1,
                image_height - 1
            )
        )

        x2 = max(
            0,
            min(
                x2,
                image_width - 1
            )
        )

        y2 = max(
            0,
            min(
                y2,
                image_height - 1
            )
        )

        # --------------------------------------------------
        # Print results
        # --------------------------------------------------

        print()
        print(
            "Prediction: DEFECTIVE"
        )

        print(
            f"Classification Confidence: "
            f"{class_confidence:.2f}%"
        )

        print(
            f"Defect Type: "
            f"{defect_name.upper()}"
        )

        print(
            f"Defect Confidence: "
            f"{defect_confidence:.2f}%"
        )

        print()
        print(
            "Bounding Box:"
        )

        print(
            f"x1: {x1}"
        )

        print(
            f"y1: {y1}"
        )

        print(
            f"x2: {x2}"
        )

        print(
            f"y2: {y2}"
        )

        # --------------------------------------------------
        # Draw bounding box
        # --------------------------------------------------

        cv2.rectangle(
            original_image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        # --------------------------------------------------
        # Create label
        # --------------------------------------------------

        label = (
            f"{defect_name.upper()} "
            f"{defect_confidence:.1f}%"
        )

        # --------------------------------------------------
        # Calculate text position
        # --------------------------------------------------

        text_x = x1

        text_y = max(
            30,
            y1 - 10
        )

        # --------------------------------------------------
        # Draw text
        # --------------------------------------------------

        cv2.putText(
            original_image,
            label,
            (
                text_x,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # --------------------------------------------------
        # Draw classification label
        # --------------------------------------------------

        cv2.putText(
            original_image,
            "DEFECTIVE",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )


    # ======================================================
    # SAVE RESULT
    # ======================================================

    base_name = os.path.splitext(
        image_filename
    )[0]

    output_filename = (
        base_name
        + "_result.jpg"
    )

    output_path = os.path.join(
        OUTPUT_DIRECTORY,
        output_filename
    )

    success = cv2.imwrite(
        output_path,
        original_image
    )

    if success:

        print()
        print(
            "Result saved:"
        )

        print(
            os.path.abspath(
                output_path
            )
        )

    else:

        print()
        print(
            "ERROR: Could not save result."
        )

    total_images += 1


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()
print("=" * 70)
print(
    "METAL INSPECTION SUMMARY"
)
print("=" * 70)

print()
print(
    f"Total Images Processed : "
    f"{total_images}"
)

print(
    f"Good Images            : "
    f"{good_count}"
)

print(
    f"Defective Images       : "
    f"{defective_count}"
)

print()

print(
    "DEFECT COUNTS"
)

print(
    f"Scratch : "
    f"{defect_counts['scratch']}"
)

print(
    f"Dent    : "
    f"{defect_counts['dent']}"
)

print(
    f"Rust    : "
    f"{defect_counts['rust']}"
)

print(
    f"Crack   : "
    f"{defect_counts['crack']}"
)

print(
    f"Hole    : "
    f"{defect_counts['hole']}"
)

print()
print(
    "Output directory:"
)

print(
    os.path.abspath(
        OUTPUT_DIRECTORY
    )
)

print()
print(
    "STATUS: METAL INSPECTION COMPLETED"
)

print("=" * 70)