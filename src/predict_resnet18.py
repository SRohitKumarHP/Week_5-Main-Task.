from pathlib import Path

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import resnet18

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_classifier.pth"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "input"
)


IMAGE_SIZE = 224

NUM_CLASSES = 2


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = [
    "defective",
    "good"
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("             METAL DEFECT INSPECTION")
print("=" * 70)

print()

print("Device:", DEVICE)

print()


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    print("ERROR: Trained model not found.")

    print()

    print(
        "Expected model:"
    )

    print(MODEL_PATH)

    exit()


# ============================================================
# FIND INPUT IMAGE
# ============================================================

image_extensions = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]


images = [
    file
    for file in INPUT_DIR.iterdir()
    if file.is_file()
    and file.suffix.lower()
    in image_extensions
]


if len(images) == 0:

    print("ERROR: No image found.")

    print()

    print(
        "Place a metal image inside:"
    )

    print(INPUT_DIR)

    print()

    print(
        "Example:"
    )

    print(
        "input/test_image.jpg"
    )

    exit()


# Use the first image

image_path = images[0]


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        image_path
    ).convert("RGB")

except Exception as error:

    print(
        "ERROR: Could not open image."
    )

    print(error)

    exit()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

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


input_tensor = transform(
    image
)


# Add batch dimension

input_tensor = input_tensor.unsqueeze(
    0
)


input_tensor = input_tensor.to(
    DEVICE
)


# ============================================================
# CREATE RESNET18
# ============================================================

model = resnet18(
    weights=None
)


input_features = (
    model.fc.in_features
)


model.fc = nn.Linear(
    input_features,
    NUM_CLASSES
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)


model = model.to(
    DEVICE
)


model.eval()


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(
        input_tensor
    )


# ============================================================
# CONVERT OUTPUT TO PROBABILITIES
# ============================================================

probabilities = torch.softmax(
    outputs,
    dim=1
)


# Get highest probability

confidence, predicted_index = torch.max(
    probabilities,
    dim=1
)


predicted_index = (
    predicted_index.item()
)

confidence = (
    confidence.item()
)


predicted_class = (
    CLASS_NAMES[predicted_index]
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)

print("INSPECTION RESULT")

print("=" * 70)

print()

print(
    "Image      :",
    image_path.name
)

print(
    "Prediction :",
    predicted_class.upper()
)

print(
    "Confidence :",
    f"{confidence * 100:.2f}%"
)

print()

# Show individual probabilities

for index, class_name in enumerate(
    CLASS_NAMES
):

    probability = (
        probabilities[0][index]
        .item()
    )

    print(
        f"{class_name.capitalize():12}: "
        f"{probability * 100:.2f}%"
    )


print()

print("=" * 70)

if predicted_class == "defective":

    print(
        "RESULT: DEFECTIVE PART DETECTED"
    )

else:

    print(
        "RESULT: GOOD PART"
    )


print("=" * 70)