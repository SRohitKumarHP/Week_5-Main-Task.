import os
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "dataset/classification/test"

MODEL_PATH = "models/mobilenetv2_classifier.pth"

RESULTS_DIR = "results"

IMAGE_SIZE = 224

BATCH_SIZE = 8


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "defective",
    "good"
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("MOBILENETV2 MODEL EVALUATION")
print("=" * 70)

print()
print("Device:", device)


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

print()
print("Loading test dataset...")


test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print()
print("Test images:", len(test_dataset))

print(
    "Classes:",
    test_dataset.classes
)

print(
    "Class mapping:",
    test_dataset.class_to_idx
)


# ============================================================
# CREATE MOBILENETV2
# ============================================================

print()
print("Creating MobileNetV2 model...")


weights = MobileNet_V2_Weights.DEFAULT


model = mobilenet_v2(
    weights=None
)


classifier_input_features = (
    model.classifier[1].in_features
)


model.classifier = nn.Sequential(

    nn.Dropout(
        p=0.2
    ),

    nn.Linear(
        classifier_input_features,
        2
    )
)


model = model.to(device)


# ============================================================
# LOAD TRAINED CHECKPOINT
# ============================================================

print()
print("Loading trained model...")


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
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


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []

all_labels = []

all_probabilities = []


print()
print("=" * 70)
print("EVALUATING TEST DATASET")
print("=" * 70)


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(
            images
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        _, predictions = torch.max(
            probabilities,
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_labels.extend(
            labels.cpu().numpy()
        )


        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


# defective = class 0
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

print()
print("=" * 70)
print("MOBILENETV2 EVALUATION RESULTS")
print("=" * 70)

print()

print(
    "Accuracy  :",
    f"{accuracy * 100:.2f}%"
)

print(
    "Precision :",
    f"{precision * 100:.2f}%"
)

print(
    "Recall    :",
    f"{recall * 100:.2f}%"
)

print(
    "F1 Score  :",
    f"{f1 * 100:.2f}%"
)


print()
print("Confusion Matrix:")

print(
    cm
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report:")

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=2,
    zero_division=0
)


print(
    report
)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

import matplotlib.pyplot as plt


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    cm
)


plt.title(
    "MobileNetV2 Confusion Matrix"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.xticks(
    [0, 1],
    CLASS_NAMES
)


plt.yticks(
    [0, 1],
    CLASS_NAMES
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


confusion_matrix_path = os.path.join(
    RESULTS_DIR,
    "mobilenetv2_confusion_matrix.png"
)


plt.savefig(
    confusion_matrix_path
)


plt.close()


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "mobilenetv2_evaluation.txt"
)


with open(
    report_path,
    "w"
) as file:

    file.write(
        "MOBILENETV2 EVALUATION RESULTS\n"
    )

    file.write(
        "=" * 60
        + "\n\n"
    )

    file.write(
        f"Test Images : {len(test_dataset)}\n\n"
    )

    file.write(
        f"Accuracy  : {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision : {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall    : {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score  : {f1 * 100:.2f}%\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        str(cm)
        + "\n\n"
    )

    file.write(
        "Classification Report:\n"
    )

    file.write(
        report
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)

print(
    "Confusion matrix saved to:"
)

print(
    os.path.abspath(
        confusion_matrix_path
    )
)

print()
print(
    "Evaluation report saved to:"
)

print(
    os.path.abspath(
        report_path
    )
)

print()
print(
    "STATUS: MOBILENETV2 EVALUATION COMPLETED"
)

print("=" * 70)