from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from torchvision.models import resnet18

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "classification"
    / "test"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_classifier.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_CLASSES = 2


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
print("             RESNET18 MODEL EVALUATION")
print("=" * 70)

print()

print("Device:", DEVICE)

print()

# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([

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


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=test_transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("Classes:")
print(test_dataset.classes)

print()

print("Class mapping:")
print(test_dataset.class_to_idx)

print()

print("Test images:")
print(len(test_dataset))

print()


# ============================================================
# CREATE RESNET18
# ============================================================

print("=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

print()


model = resnet18(
    weights=None
)


input_features = model.fc.in_features


model.fc = nn.Linear(
    input_features,
    NUM_CLASSES
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)


model = model.to(DEVICE)


model.eval()


print("Model loaded successfully.")

print()


# ============================================================
# MAKE PREDICTIONS
# ============================================================

all_labels = []

all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(
            images
        )

        _, predictions = torch.max(
            outputs,
            1
        )


        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


precision = precision_score(
    all_labels,
    all_predictions,
    pos_label=0,
    zero_division=0
)


recall = recall_score(
    all_labels,
    all_predictions,
    pos_label=0,
    zero_division=0
)


f1 = f1_score(
    all_labels,
    all_predictions,
    pos_label=0,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 70)
print("MODEL EVALUATION RESULTS")
print("=" * 70)

print()

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(
    "Rows    = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    "Classes:",
    test_dataset.classes
)

print()

print(cm)

print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        zero_division=0
    )
)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "ResNet18 Confusion Matrix"
)

plt.colorbar()

tick_marks = range(
    len(test_dataset.classes)
)

plt.xticks(
    tick_marks,
    test_dataset.classes
)

plt.yticks(
    tick_marks,
    test_dataset.classes
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)


# Add numbers

for i in range(
    len(test_dataset.classes)
):

    for j in range(
        len(test_dataset.classes)
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


confusion_matrix_path = (
    RESULTS_DIR
    / "confusion_matrix.png"
)


plt.savefig(
    confusion_matrix_path
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("=" * 70)

print(
    "Confusion matrix saved to:"
)

print(
    confusion_matrix_path
)

print()

print("=" * 70)

print("MODEL EVALUATION COMPLETED")

print("=" * 70)