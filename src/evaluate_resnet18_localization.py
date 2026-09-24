import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "detection"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet18_localization.pth"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 8

NUM_CLASSES = 2
NUM_DEFECT_TYPES = 5

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
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("RESNET18 LOCALIZATION MODEL EVALUATION")
print("=" * 70)

print("\nDevice:", device)


# ============================================================
# DATASET
# ============================================================

class MetalDefectTestDataset(Dataset):

    def __init__(
        self,
        transform=None
    ):

        self.image_dir = os.path.join(
            DATASET_DIR,
            "images",
            "test"
        )

        self.label_dir = os.path.join(
            DATASET_DIR,
            "labels",
            "test"
        )

        self.transform = transform

        self.samples = []

        image_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        )

        image_files = sorted([
            file
            for file in os.listdir(
                self.image_dir
            )
            if file.lower().endswith(
                image_extensions
            )
        ])

        for image_file in image_files:

            image_path = os.path.join(
                self.image_dir,
                image_file
            )

            label_file = (
                os.path.splitext(image_file)[0]
                + ".txt"
            )

            label_path = os.path.join(
                self.label_dir,
                label_file
            )

            # Default = Good
            class_id = 1

            defect_type = -1

            bbox = [
                0.0,
                0.0,
                0.0,
                0.0
            ]

            has_defect = False

            if os.path.exists(label_path):

                with open(
                    label_path,
                    "r"
                ) as file:

                    lines = [
                        line.strip()
                        for line in file
                        if line.strip()
                    ]

                if len(lines) > 0:

                    values = lines[0].split()

                    if len(values) == 5:

                        defect_type = int(
                            values[0]
                        )

                        bbox = [
                            float(values[1]),
                            float(values[2]),
                            float(values[3]),
                            float(values[4])
                        ]

                        class_id = 0

                        has_defect = True

            self.samples.append({
                "image": image_path,
                "class_id": class_id,
                "defect_type": defect_type,
                "bbox": bbox,
                "has_defect": has_defect
            })

        print(
            "Test images:",
            len(self.samples)
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = cv2.imread(
            sample["image"]
        )

        if image is None:

            raise RuntimeError(
                f"Unable to read: "
                f"{sample['image']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        from PIL import Image

        image = Image.fromarray(image)

        if self.transform:

            image = self.transform(image)

        class_label = torch.tensor(
            sample["class_id"],
            dtype=torch.long
        )

        defect_label = torch.tensor(
            sample["defect_type"],
            dtype=torch.long
        )

        bbox = torch.tensor(
            sample["bbox"],
            dtype=torch.float32
        )

        has_defect = torch.tensor(
            sample["has_defect"],
            dtype=torch.bool
        )

        return (
            image,
            class_label,
            defect_label,
            bbox,
            has_defect
        )


# ============================================================
# TRANSFORM
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
# CREATE DATASET
# ============================================================

test_dataset = MetalDefectTestDataset(
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

class ResNet18Localization(nn.Module):

    def __init__(self):

        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        resnet = resnet18(
            weights=weights
        )

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.feature_size = 512

        # Binary classifier
        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

        # Defect type classifier
        self.defect_classifier = nn.Sequential(

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                NUM_DEFECT_TYPES
            )
        )

        # Bounding box regressor
        self.bbox_regressor = nn.Sequential(

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                4
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        features = self.backbone(x)

        features = torch.flatten(
            features,
            start_dim=1
        )

        class_output = self.classifier(
            features
        )

        defect_output = self.defect_classifier(
            features
        )

        bbox_output = self.bbox_regressor(
            features
        )

        return (
            class_output,
            defect_output,
            bbox_output
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = ResNet18Localization()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ============================================================
# EVALUATION STORAGE
# ============================================================

true_classes = []
pred_classes = []

true_defects = []
pred_defects = []

bbox_ious = []

bbox_correct = 0
bbox_total = 0


# ============================================================
# IOU FUNCTION
# ============================================================

def calculate_iou(
    box1,
    box2
):

    # box format:
    # x_center, y_center, width, height

    x1_center, y1_center, w1, h1 = box1

    x2_center, y2_center, w2, h2 = box2

    # Convert to corner coordinates

    box1_x1 = x1_center - w1 / 2
    box1_y1 = y1_center - h1 / 2

    box1_x2 = x1_center + w1 / 2
    box1_y2 = y1_center + h1 / 2

    box2_x1 = x2_center - w2 / 2
    box2_y1 = y2_center - h2 / 2

    box2_x2 = x2_center + w2 / 2
    box2_y2 = y2_center + h2 / 2

    # Intersection

    intersection_x1 = max(
        box1_x1,
        box2_x1
    )

    intersection_y1 = max(
        box1_y1,
        box2_y1
    )

    intersection_x2 = min(
        box1_x2,
        box2_x2
    )

    intersection_y2 = min(
        box1_y2,
        box2_y2
    )

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    # Areas

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = (
        area1
        + area2
        - intersection_area
    )

    if union_area <= 0:

        return 0.0

    return (
        intersection_area
        / union_area
    )


# ============================================================
# RUN TEST SET
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING TEST DATASET")
print("=" * 70)


with torch.no_grad():

    for (
        images,
        class_labels,
        defect_labels,
        bbox_targets,
        has_defect
    ) in test_loader:

        images = images.to(device)

        (
            class_output,
            defect_output,
            bbox_output
        ) = model(images)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        class_predictions = torch.argmax(
            class_output,
            dim=1
        )

        true_classes.extend(
            class_labels.numpy()
        )

        pred_classes.extend(
            class_predictions.cpu().numpy()
        )

        # ----------------------------------------------------
        # Defect type
        # ----------------------------------------------------

        defect_predictions = torch.argmax(
            defect_output,
            dim=1
        )

        for i in range(
            len(has_defect)
        ):

            if has_defect[i]:

                true_defects.append(
                    defect_labels[i].item()
                )

                pred_defects.append(
                    defect_predictions[i].item()
                )

                # --------------------------------------------
                # Bounding box IoU
                # --------------------------------------------

                predicted_box = (
                    bbox_output[i]
                    .cpu()
                    .numpy()
                )

                actual_box = (
                    bbox_targets[i]
                    .cpu()
                    .numpy()
                )

                iou = calculate_iou(
                    predicted_box,
                    actual_box
                )

                bbox_ious.append(
                    iou
                )

                bbox_total += 1

                if iou >= 0.50:

                    bbox_correct += 1


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    true_classes,
    pred_classes
)

precision = precision_score(
    true_classes,
    pred_classes,
    pos_label=0,
    zero_division=0
)

recall = recall_score(
    true_classes,
    pred_classes,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    true_classes,
    pred_classes,
    pos_label=0,
    zero_division=0
)


# ============================================================
# DEFECT TYPE METRICS
# ============================================================

defect_accuracy = accuracy_score(
    true_defects,
    pred_defects
)

defect_report = classification_report(
    true_defects,
    pred_defects,
    labels=list(range(NUM_DEFECT_TYPES)),
    target_names=DEFECT_NAMES,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_classes,
    pred_classes,
    labels=[0, 1]
)


# ============================================================
# LOCALIZATION METRICS
# ============================================================

if len(bbox_ious) > 0:

    mean_iou = np.mean(
        bbox_ious
    )

    bbox_accuracy = (
        bbox_correct
        / bbox_total
    )

else:

    mean_iou = 0.0
    bbox_accuracy = 0.0


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION RESULTS")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
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


print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        true_classes,
        pred_classes,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# DEFECT TYPE RESULTS
# ============================================================

print("=" * 70)
print("DEFECT TYPE RESULTS")
print("=" * 70)

print(
    f"\nDefect Type Accuracy: "
    f"{defect_accuracy * 100:.2f}%"
)

print("\nPer-class report:")

print(defect_report)


# ============================================================
# LOCALIZATION RESULTS
# ============================================================

print("=" * 70)
print("BOUNDING BOX LOCALIZATION RESULTS")
print("=" * 70)

print(
    f"\nDefective test images: "
    f"{bbox_total}"
)

print(
    f"Mean IoU: "
    f"{mean_iou:.4f}"
)

print(
    f"Mean IoU: "
    f"{mean_iou * 100:.2f}%"
)

print(
    f"Bounding Box Accuracy "
    f"(IoU >= 0.50): "
    f"{bbox_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm
)

plt.title(
    "ResNet18 Classification Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    CLASS_NAMES
)

plt.yticks(
    [0, 1],
    CLASS_NAMES
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
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

cm_path = os.path.join(
    RESULTS_DIR,
    "localization_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=150
)

plt.close()


# ============================================================
# IOU DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    bbox_ious,
    bins=10
)

plt.axvline(
    0.50,
    linestyle="--",
    label="IoU = 0.50"
)

plt.xlabel(
    "IoU"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "Bounding Box IoU Distribution"
)

plt.legend()

plt.grid(True)

iou_path = os.path.join(
    RESULTS_DIR,
    "bbox_iou_distribution.png"
)

plt.savefig(
    iou_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "resnet18_localization_evaluation.txt"
)

with open(
    report_path,
    "w"
) as file:

    file.write(
        "RESNET18 LOCALIZATION EVALUATION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Test Images: {len(test_dataset)}\n\n"
    )

    file.write(
        "CLASSIFICATION\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall: {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score: {f1 * 100:.2f}%\n\n"
    )

    file.write(
        "DEFECT TYPE\n"
    )

    file.write(
        f"Accuracy: "
        f"{defect_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        defect_report
    )

    file.write(
        "\nBOUNDING BOX LOCALIZATION\n"
    )

    file.write(
        f"Mean IoU: {mean_iou:.4f}\n"
    )

    file.write(
        f"IoU >= 0.50 Accuracy: "
        f"{bbox_accuracy * 100:.2f}%\n"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

print(
    "Confusion matrix saved to:"
)

print(cm_path)

print(
    "\nIoU graph saved to:"
)

print(iou_path)

print(
    "\nEvaluation report saved to:"
)

print(report_path)

print("\n" + "=" * 70)

print(
    "STATUS: RESNET18 LOCALIZATION EVALUATION COMPLETED"
)

print("=" * 70)