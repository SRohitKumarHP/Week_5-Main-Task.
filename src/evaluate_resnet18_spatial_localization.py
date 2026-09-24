import os
import cv2
import torch
import numpy as np

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
    "resnet18_spatial_localization.pth"
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
print("IMPROVED RESNET18 LOCALIZATION EVALUATION")
print("=" * 70)

print("\nDevice:", device)


# ============================================================
# DATASET
# ============================================================

class MetalDefectTestDataset(Dataset):

    def __init__(self, transform=None):

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

        extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        )

        image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(extensions)
        ])

        for image_file in image_files:

            image_path = os.path.join(
                self.image_dir,
                image_file
            )

            label_path = os.path.join(
                self.label_dir,
                os.path.splitext(image_file)[0] + ".txt"
            )

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

                with open(label_path, "r") as file:

                    lines = [
                        line.strip()
                        for line in file
                        if line.strip()
                    ]

                if len(lines) > 0:

                    values = lines[0].split()

                    if len(values) == 5:

                        defect_type = int(values[0])

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
# DATASET
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

class ResNet18SpatialLocalization(nn.Module):

    def __init__(self):

        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        resnet = resnet18(
            weights=weights
        )

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        self.classifier_pool = (
            nn.AdaptiveAvgPool2d((1, 1))
        )

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

        self.defect_classifier_pool = (
            nn.AdaptiveAvgPool2d((1, 1))
        )

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

        self.localization = nn.Sequential(

            nn.Conv2d(
                512,
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

        self.bbox_regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
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

    def forward(self, x):

        features = self.backbone(x)

        class_features = (
            self.classifier_pool(features)
        )

        class_features = torch.flatten(
            class_features,
            start_dim=1
        )

        class_output = self.classifier(
            class_features
        )

        defect_features = (
            self.defect_classifier_pool(features)
        )

        defect_features = torch.flatten(
            defect_features,
            start_dim=1
        )

        defect_output = self.defect_classifier(
            defect_features
        )

        localization_features = (
            self.localization(features)
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
# LOAD MODEL
# ============================================================

print("\nLoading trained spatial model...")

model = ResNet18SpatialLocalization()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("Spatial model loaded successfully.")


# ============================================================
# IOU FUNCTION
# ============================================================

def calculate_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x1 = x1 - w1 / 2
    box1_y1 = y1 - h1 / 2
    box1_x2 = x1 + w1 / 2
    box1_y2 = y1 + h1 / 2

    box2_x1 = x2 - w2 / 2
    box2_y1 = y2 - h2 / 2
    box2_x2 = x2 + w2 / 2
    box2_y2 = y2 + h2 / 2

    inter_x1 = max(
        box1_x1,
        box2_x1
    )

    inter_y1 = max(
        box1_y1,
        box2_y1
    )

    inter_x2 = min(
        box1_x2,
        box2_x2
    )

    inter_y2 = min(
        box1_y2,
        box2_y2
    )

    inter_width = max(
        0,
        inter_x2 - inter_x1
    )

    inter_height = max(
        0,
        inter_y2 - inter_y1
    )

    intersection = (
        inter_width
        * inter_height
    )

    area1 = w1 * h1
    area2 = w2 * h2

    union = (
        area1
        + area2
        - intersection
    )

    if union <= 0:

        return 0.0

    return intersection / union


# ============================================================
# EVALUATION STORAGE
# ============================================================

true_classes = []
pred_classes = []

true_defects = []
pred_defects = []

ious = []

per_class_ious = {
    name: []
    for name in DEFECT_NAMES
}


# ============================================================
# EVALUATION
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

        class_predictions = torch.argmax(
            class_output,
            dim=1
        )

        defect_predictions = torch.argmax(
            defect_output,
            dim=1
        )

        true_classes.extend(
            class_labels.numpy()
        )

        pred_classes.extend(
            class_predictions.cpu().numpy()
        )

        for i in range(
            len(has_defect)
        ):

            if has_defect[i]:

                true_defect = (
                    defect_labels[i].item()
                )

                predicted_defect = (
                    defect_predictions[i].item()
                )

                true_defects.append(
                    true_defect
                )

                pred_defects.append(
                    predicted_defect
                )

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

                ious.append(iou)

                per_class_ious[
                    DEFECT_NAMES[true_defect]
                ].append(iou)


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


# ============================================================
# LOCALIZATION METRICS
# ============================================================

mean_iou = np.mean(
    ious
)

iou_50_count = sum(
    iou >= 0.50
    for iou in ious
)

iou_50_accuracy = (
    iou_50_count
    / len(ious)
)


# ============================================================
# PRINT CLASSIFICATION
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

print(
    confusion_matrix(
        true_classes,
        pred_classes
    )
)

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
# DEFECT TYPE
# ============================================================

print("=" * 70)
print("DEFECT TYPE RESULTS")
print("=" * 70)

print(
    f"\nDefect Type Accuracy: "
    f"{defect_accuracy * 100:.2f}%"
)

print(
    classification_report(
        true_defects,
        pred_defects,
        labels=list(range(NUM_DEFECT_TYPES)),
        target_names=DEFECT_NAMES,
        zero_division=0
    )
)


# ============================================================
# LOCALIZATION
# ============================================================

print("=" * 70)
print("BOUNDING BOX LOCALIZATION RESULTS")
print("=" * 70)

print(
    f"\nDefective test images: "
    f"{len(ious)}"
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
    f"{iou_50_accuracy * 100:.2f}%"
)


# ============================================================
# PER-DEFECT IOU
# ============================================================

print("\nPer-defect Mean IoU:")

for defect_name in DEFECT_NAMES:

    values = per_class_ious[defect_name]

    if len(values) > 0:

        class_iou = np.mean(values)

        print(
            f"{defect_name:<10}: "
            f"{class_iou:.4f} "
            f"({class_iou * 100:.2f}%)"
        )


# ============================================================
# COMPARISON
# ============================================================

OLD_MEAN_IOU = 0.3476
OLD_IOU_50 = 0.2444

iou_improvement = (
    mean_iou
    - OLD_MEAN_IOU
)

iou50_improvement = (
    iou_50_accuracy
    - OLD_IOU_50
)

print("\n" + "=" * 70)
print("COMPARISON WITH PREVIOUS MODEL")
print("=" * 70)

print(
    f"\nPrevious Mean IoU : "
    f"{OLD_MEAN_IOU * 100:.2f}%"
)

print(
    f"New Mean IoU      : "
    f"{mean_iou * 100:.2f}%"
)

print(
    f"Improvement       : "
    f"{iou_improvement * 100:+.2f} percentage points"
)

print(
    f"\nPrevious IoU >= 0.50 : "
    f"{OLD_IOU_50 * 100:.2f}%"
)

print(
    f"New IoU >= 0.50      : "
    f"{iou_50_accuracy * 100:.2f}%"
)

print(
    f"Improvement          : "
    f"{iou50_improvement * 100:+.2f} percentage points"
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "spatial_localization_evaluation.txt"
)

with open(
    report_path,
    "w"
) as file:

    file.write(
        "IMPROVED RESNET18 LOCALIZATION EVALUATION\n"
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
        "LOCALIZATION\n"
    )

    file.write(
        f"Mean IoU: "
        f"{mean_iou * 100:.2f}%\n"
    )

    file.write(
        f"IoU >= 0.50: "
        f"{iou_50_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "COMPARISON\n"
    )

    file.write(
        f"Previous Mean IoU: "
        f"{OLD_MEAN_IOU * 100:.2f}%\n"
    )

    file.write(
        f"New Mean IoU: "
        f"{mean_iou * 100:.2f}%\n"
    )

    file.write(
        f"Mean IoU Improvement: "
        f"{iou_improvement * 100:+.2f} percentage points\n"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

print(
    "Evaluation report saved to:"
)

print(report_path)

print("\n" + "=" * 70)

print(
    "STATUS: IMPROVED RESNET18 "
    "LOCALIZATION EVALUATION COMPLETED"
)

print("=" * 70)