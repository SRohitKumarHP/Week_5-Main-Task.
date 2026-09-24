import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# CONFIGURATION
# ==========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_ROOT = "dataset/detection/images"
LABEL_ROOT = "dataset/detection/labels"

MODEL_PATH = (
    "models/"
    "mobilenetv2_spatial_localization.pth"
)

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
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


# ==========================================================
# DATASET
# ==========================================================

class MetalDefectDataset(Dataset):

    def __init__(self, split):

        self.image_dir = os.path.join(
            IMAGE_ROOT,
            split
        )

        self.label_dir = os.path.join(
            LABEL_ROOT,
            split
        )

        self.image_files = []

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        )

        for filename in sorted(
            os.listdir(self.image_dir)
        ):

            if filename.lower().endswith(
                valid_extensions
            ):

                self.image_files.append(
                    filename
                )

        self.transform = transforms.Compose([
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


    def __len__(self):

        return len(
            self.image_files
        )


    def __getitem__(self, index):

        filename = self.image_files[index]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        label_filename = (
            os.path.splitext(filename)[0]
            + ".txt"
        )

        label_path = os.path.join(
            self.label_dir,
            label_filename
        )

        # --------------------------------------------------
        # Read image
        # --------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise RuntimeError(
                f"Unable to read image: "
                f"{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------------------------
        # Default = GOOD
        # --------------------------------------------------

        classification_label = 1

        defect_label = 0

        bbox = np.array(
            [0.0, 0.0, 0.0, 0.0],
            dtype=np.float32
        )

        # --------------------------------------------------
        # Read annotation
        # --------------------------------------------------

        if os.path.exists(
            label_path
        ):

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

                parts = lines[0].split()

                if len(parts) == 5:

                    defect_label = int(
                        parts[0]
                    )

                    x_center = float(
                        parts[1]
                    )

                    y_center = float(
                        parts[2]
                    )

                    width = float(
                        parts[3]
                    )

                    height = float(
                        parts[4]
                    )

                    classification_label = 0

                    bbox = np.array(
                        [
                            x_center,
                            y_center,
                            width,
                            height
                        ],
                        dtype=np.float32
                    )

        # --------------------------------------------------
        # Transform
        # --------------------------------------------------

        image = self.transform(
            image
        )

        return (
            image,
            torch.tensor(
                classification_label,
                dtype=torch.long
            ),
            torch.tensor(
                defect_label,
                dtype=torch.long
            ),
            torch.tensor(
                bbox,
                dtype=torch.float32
            ),
            filename
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

        backbone = mobilenet_v2(
            weights=None
        )

        self.features = (
            backbone.features
        )

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
# IoU CALCULATION
# ==========================================================

def calculate_iou(
    predicted,
    target
):

    # ------------------------------------------------------
    # YOLO format:
    # x_center, y_center, width, height
    # ------------------------------------------------------

    px, py, pw, ph = predicted

    tx, ty, tw, th = target

    # ------------------------------------------------------
    # Predicted corners
    # ------------------------------------------------------

    pred_x1 = px - pw / 2
    pred_y1 = py - ph / 2

    pred_x2 = px + pw / 2
    pred_y2 = py + ph / 2

    # ------------------------------------------------------
    # Target corners
    # ------------------------------------------------------

    target_x1 = tx - tw / 2
    target_y1 = ty - th / 2

    target_x2 = tx + tw / 2
    target_y2 = ty + th / 2

    # ------------------------------------------------------
    # Intersection
    # ------------------------------------------------------

    intersection_x1 = max(
        pred_x1,
        target_x1
    )

    intersection_y1 = max(
        pred_y1,
        target_y1
    )

    intersection_x2 = min(
        pred_x2,
        target_x2
    )

    intersection_y2 = min(
        pred_y2,
        target_y2
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

    # ------------------------------------------------------
    # Areas
    # ------------------------------------------------------

    predicted_area = (
        max(0, pred_x2 - pred_x1)
        *
        max(0, pred_y2 - pred_y1)
    )

    target_area = (
        max(0, target_x2 - target_x1)
        *
        max(0, target_y2 - target_y1)
    )

    union_area = (
        predicted_area
        + target_area
        - intersection_area
    )

    if union_area <= 0:

        return 0.0

    return (
        intersection_area
        / union_area
    )


# ==========================================================
# START
# ==========================================================

print("=" * 70)
print(
    "MOBILENETV2 SPATIAL LOCALIZATION "
    "EVALUATION"
)
print("=" * 70)

print()
print(
    "Device:",
    DEVICE
)

print()
print(
    "Loading test dataset..."
)

test_dataset = MetalDefectDataset(
    "test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print(
    "Test images:",
    len(test_dataset)
)

print(
    "Classes:",
    CLASS_NAMES
)

print(
    "Defect types:",
    DEFECT_NAMES
)


# ==========================================================
# LOAD MODEL
# ==========================================================

print()
print(
    "Creating MobileNetV2 model..."
)

model = MobileNetV2SpatialLocalization(
    num_classes=2,
    num_defect_types=5
)

model = model.to(DEVICE)

print()
print(
    "Loading trained model..."
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(
    checkpoint,
    dict
) and "model_state_dict" in checkpoint:

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
# EVALUATION VARIABLES
# ==========================================================

true_classes = []
predicted_classes = []

true_defects = []
predicted_defects = []

ious = []

iou_50_count = 0

defect_iou = {
    name: []
    for name in DEFECT_NAMES
}

defect_correct = {
    name: 0
    for name in DEFECT_NAMES
}

defect_total = {
    name: 0
    for name in DEFECT_NAMES
}


# ==========================================================
# EVALUATION
# ==========================================================

print()
print("=" * 70)
print(
    "EVALUATING TEST DATASET"
)
print("=" * 70)

with torch.no_grad():

    for (
        images,
        class_labels,
        defect_labels,
        bbox_targets,
        filenames
    ) in test_loader:

        images = images.to(
            DEVICE
        )

        class_labels = class_labels.to(
            DEVICE
        )

        defect_labels = defect_labels.to(
            DEVICE
        )

        bbox_targets = bbox_targets.to(
            DEVICE
        )

        (
            class_output,
            defect_output,
            bbox_output
        ) = model(images)

        # --------------------------------------------------
        # Classification prediction
        # --------------------------------------------------

        class_predictions = torch.argmax(
            class_output,
            dim=1
        )

        # --------------------------------------------------
        # Defect prediction
        # --------------------------------------------------

        defect_predictions = torch.argmax(
            defect_output,
            dim=1
        )

        # --------------------------------------------------
        # Store classification results
        # --------------------------------------------------

        true_classes.extend(
            class_labels.cpu().numpy()
        )

        predicted_classes.extend(
            class_predictions.cpu().numpy()
        )

        # --------------------------------------------------
        # Store defective-image results
        # --------------------------------------------------

        for i in range(
            images.size(0)
        ):

            actual_class = (
                class_labels[i].item()
            )

            if actual_class == 0:

                actual_defect = (
                    defect_labels[i].item()
                )

                predicted_defect = (
                    defect_predictions[i].item()
                )

                true_defects.append(
                    actual_defect
                )

                predicted_defects.append(
                    predicted_defect
                )

                defect_name = (
                    DEFECT_NAMES[
                        actual_defect
                    ]
                )

                defect_total[
                    defect_name
                ] += 1

                if (
                    actual_defect
                    ==
                    predicted_defect
                ):

                    defect_correct[
                        defect_name
                    ] += 1

                # --------------------------------------------------
                # IoU
                # --------------------------------------------------

                predicted_bbox = (
                    bbox_output[i]
                    .cpu()
                    .numpy()
                )

                target_bbox = (
                    bbox_targets[i]
                    .cpu()
                    .numpy()
                )

                iou = calculate_iou(
                    predicted_bbox,
                    target_bbox
                )

                ious.append(
                    iou
                )

                defect_iou[
                    defect_name
                ].append(
                    iou
                )

                if iou >= 0.50:

                    iou_50_count += 1


# ==========================================================
# CLASSIFICATION METRICS
# ==========================================================

accuracy = accuracy_score(
    true_classes,
    predicted_classes
)

precision = precision_score(
    true_classes,
    predicted_classes,
    pos_label=0,
    zero_division=0
)

recall = recall_score(
    true_classes,
    predicted_classes,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    true_classes,
    predicted_classes,
    pos_label=0,
    zero_division=0
)

confusion = confusion_matrix(
    true_classes,
    predicted_classes
)

classification_report_text = (
    classification_report(
        true_classes,
        predicted_classes,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ==========================================================
# DEFECT TYPE METRICS
# ==========================================================

if len(true_defects) > 0:

    defect_accuracy = accuracy_score(
        true_defects,
        predicted_defects
    )

    defect_report = classification_report(
        true_defects,
        predicted_defects,
        labels=list(range(5)),
        target_names=DEFECT_NAMES,
        zero_division=0
    )

else:

    defect_accuracy = 0.0

    defect_report = (
        "No defective images found."
    )


# ==========================================================
# IOU METRICS
# ==========================================================

if len(ious) > 0:

    mean_iou = np.mean(
        ious
    )

    iou_50_percentage = (
        iou_50_count
        / len(ious)
        * 100
    )

else:

    mean_iou = 0.0

    iou_50_percentage = 0.0


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print()
print("=" * 70)
print(
    "MOBILENETV2 EVALUATION RESULTS"
)
print("=" * 70)

print()
print(
    "CLASSIFICATION RESULTS"
)

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
print(
    "Confusion Matrix:"
)

print(
    confusion
)

print()
print(
    "Classification Report:"
)

print(
    classification_report_text
)


# ==========================================================
# DEFECT TYPE RESULTS
# ==========================================================

print()
print("=" * 70)
print(
    "DEFECT TYPE RESULTS"
)
print("=" * 70)

print()
print(
    f"Defect Type Accuracy: "
    f"{defect_accuracy * 100:.2f}%"
)

print()
print(
    "Defect Classification Report:"
)

print(
    defect_report
)


# ==========================================================
# PER DEFECT ACCURACY
# ==========================================================

print()
print(
    "PER-DEFECT ACCURACY"
)

for defect_name in DEFECT_NAMES:

    total = defect_total[
        defect_name
    ]

    correct = defect_correct[
        defect_name
    ]

    if total > 0:

        defect_acc = (
            correct
            / total
            * 100
        )

    else:

        defect_acc = 0.0

    print(
        f"{defect_name:<10}: "
        f"{defect_acc:.2f}% "
        f"({correct}/{total})"
    )


# ==========================================================
# BOUNDING BOX RESULTS
# ==========================================================

print()
print("=" * 70)
print(
    "BOUNDING BOX LOCALIZATION RESULTS"
)
print("=" * 70)

print()
print(
    "Defective test images:",
    len(ious)
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
    f"IoU >= 0.50: "
    f"{iou_50_percentage:.2f}%"
)

print()
print(
    "Per-defect Mean IoU:"
)

for defect_name in DEFECT_NAMES:

    values = defect_iou[
        defect_name
    ]

    if len(values) > 0:

        mean_defect_iou = np.mean(
            values
        )

        print(
            f"{defect_name:<10}: "
            f"{mean_defect_iou:.4f}"
        )

    else:

        print(
            f"{defect_name:<10}: "
            "N/A"
        )


# ==========================================================
# SAVE REPORT
# ==========================================================

report_path = os.path.join(
    RESULTS_DIR,
    "mobilenetv2_spatial_localization_evaluation.txt"
)

with open(
    report_path,
    "w"
) as file:

    file.write(
        "MOBILENETV2 SPATIAL LOCALIZATION "
        "EVALUATION\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        "CLASSIFICATION RESULTS\n"
    )

    file.write(
        f"Accuracy  : "
        f"{accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision : "
        f"{precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall    : "
        f"{recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score  : "
        f"{f1 * 100:.2f}%\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        str(confusion)
        + "\n\n"
    )

    file.write(
        "Classification Report:\n"
    )

    file.write(
        classification_report_text
    )

    file.write(
        "\n\n"
    )

    file.write(
        "DEFECT TYPE RESULTS\n"
    )

    file.write(
        f"Defect Type Accuracy: "
        f"{defect_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        defect_report
    )

    file.write(
        "\n\n"
    )

    file.write(
        "BOUNDING BOX RESULTS\n"
    )

    file.write(
        f"Defective Images: "
        f"{len(ious)}\n"
    )

    file.write(
        f"Mean IoU: "
        f"{mean_iou:.4f} "
        f"({mean_iou * 100:.2f}%)\n"
    )

    file.write(
        f"IoU >= 0.50: "
        f"{iou_50_percentage:.2f}%\n\n"
    )

    file.write(
        "Per-defect Mean IoU:\n"
    )

    for defect_name in DEFECT_NAMES:

        values = defect_iou[
            defect_name
        ]

        if len(values) > 0:

            value = np.mean(
                values
            )

            file.write(
                f"{defect_name:<10}: "
                f"{value:.4f}\n"
            )

        else:

            file.write(
                f"{defect_name:<10}: "
                "N/A\n"
            )


# ==========================================================
# FINAL MESSAGE
# ==========================================================

print()
print("=" * 70)

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
    "STATUS: MOBILENETV2 SPATIAL "
    "LOCALIZATION EVALUATION COMPLETED"
)

print("=" * 70)