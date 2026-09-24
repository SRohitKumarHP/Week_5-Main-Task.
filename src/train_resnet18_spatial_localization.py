import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from PIL import Image

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001

NUM_CLASSES = 2
NUM_DEFECT_TYPES = 5

CLASSIFICATION_WEIGHT = 1.0
DEFECT_TYPE_WEIGHT = 1.0
BBOX_WEIGHT = 5.0


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
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("TRAIN IMPROVED RESNET18 SPATIAL LOCALIZATION")
print("=" * 70)

print("\nDevice:", device)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)


# ============================================================
# DATASET
# ============================================================

class MetalDefectDataset(Dataset):

    def __init__(self, split, transform=None):

        self.transform = transform

        self.image_dir = os.path.join(
            DATASET_DIR,
            "images",
            split
        )

        self.label_dir = os.path.join(
            DATASET_DIR,
            "labels",
            split
        )

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
            f"{split.upper()} dataset loaded: "
            f"{len(self.samples)} images"
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
                f"Unable to read: {sample['image']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

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
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomRotation(
        5
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20
    ),

    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0)
            )
        ],
        p=0.20
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


val_transform = transforms.Compose([

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
# DATASETS
# ============================================================

train_dataset = MetalDefectDataset(
    "train",
    train_transform
)

val_dataset = MetalDefectDataset(
    "val",
    val_transform
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = Counter(
    sample["class_id"]
    for sample in train_dataset.samples
)

total = (
    class_counts[0]
    + class_counts[1]
)

defective_weight = (
    total / (2 * class_counts[0])
)

good_weight = (
    total / (2 * class_counts[1])
)

class_weights = torch.tensor(
    [
        defective_weight,
        good_weight
    ],
    dtype=torch.float32
).to(device)

print("\nClassification class weights:")

print(
    f"Defective: {defective_weight:.4f}"
)

print(
    f"Good:      {good_weight:.4f}"
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

        # Keep convolutional feature extractor.
        # Remove avgpool and fc.

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        self.feature_channels = 512

        # ----------------------------------------------------
        # Classification branch
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Defect type branch
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Spatial localization branch
        # ----------------------------------------------------

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

        # Classification
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

        # Defect type
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

        # Localization
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
# CREATE MODEL
# ============================================================

print("\nLoading pretrained ResNet18...")

model = ResNet18SpatialLocalization()

model = model.to(device)

print("Model loaded successfully.")


# ============================================================
# LOSS FUNCTIONS
# ============================================================

classification_loss_fn = nn.CrossEntropyLoss(
    weight=class_weights
)

defect_loss_fn = nn.CrossEntropyLoss()

bbox_loss_fn = nn.SmoothL1Loss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# HISTORY
# ============================================================

train_losses = []
val_losses = []

train_bbox_losses = []
val_bbox_losses = []

train_class_acc = []
val_class_acc = []

train_defect_acc = []
val_defect_acc = []


# ============================================================
# BEST MODEL
# ============================================================

best_val_loss = float("inf")

model_path = os.path.join(
    MODEL_DIR,
    "resnet18_spatial_localization.pth"
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(NUM_EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0
    running_bbox_loss = 0.0

    correct_classes = 0
    total_classes = 0

    correct_defects = 0
    total_defects = 0

    for (
        images,
        class_labels,
        defect_labels,
        bbox_targets,
        has_defect
    ) in train_loader:

        images = images.to(device)
        class_labels = class_labels.to(device)
        defect_labels = defect_labels.to(device)
        bbox_targets = bbox_targets.to(device)
        has_defect = has_defect.to(device)

        optimizer.zero_grad()

        (
            class_output,
            defect_output,
            bbox_output
        ) = model(images)

        # ----------------------------------------------------
        # Classification loss
        # ----------------------------------------------------

        class_loss = classification_loss_fn(
            class_output,
            class_labels
        )

        # ----------------------------------------------------
        # Defect and bounding-box losses
        # ----------------------------------------------------

        if has_defect.any():

            defect_loss = defect_loss_fn(
                defect_output[has_defect],
                defect_labels[has_defect]
            )

            bbox_loss = bbox_loss_fn(
                bbox_output[has_defect],
                bbox_targets[has_defect]
            )

        else:

            defect_loss = torch.tensor(
                0.0,
                device=device
            )

            bbox_loss = torch.tensor(
                0.0,
                device=device
            )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total_loss = (

            CLASSIFICATION_WEIGHT
            * class_loss

            +

            DEFECT_TYPE_WEIGHT
            * defect_loss

            +

            BBOX_WEIGHT
            * bbox_loss
        )

        total_loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Loss statistics
        # ----------------------------------------------------

        running_loss += (
            total_loss.item()
            * images.size(0)
        )

        running_bbox_loss += (
            bbox_loss.item()
            * images.size(0)
        )

        # ----------------------------------------------------
        # Classification accuracy
        # ----------------------------------------------------

        class_predictions = torch.argmax(
            class_output,
            dim=1
        )

        correct_classes += (
            class_predictions == class_labels
        ).sum().item()

        total_classes += (
            class_labels.size(0)
        )

        # ----------------------------------------------------
        # Defect accuracy
        # ----------------------------------------------------

        if has_defect.any():

            defect_predictions = torch.argmax(
                defect_output[has_defect],
                dim=1
            )

            correct_defects += (
                defect_predictions
                ==
                defect_labels[has_defect]
            ).sum().item()

            total_defects += (
                has_defect.sum().item()
            )

    # ========================================================
    # TRAIN METRICS
    # ========================================================

    train_loss = (
        running_loss
        / len(train_dataset)
    )

    train_bbox_loss = (
        running_bbox_loss
        / len(train_dataset)
    )

    train_class_accuracy = (
        correct_classes
        / total_classes
    )

    train_defect_accuracy = (
        correct_defects
        / total_defects
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0
    val_running_bbox_loss = 0.0

    val_correct_classes = 0
    val_total_classes = 0

    val_correct_defects = 0
    val_total_defects = 0

    with torch.no_grad():

        for (
            images,
            class_labels,
            defect_labels,
            bbox_targets,
            has_defect
        ) in val_loader:

            images = images.to(device)
            class_labels = class_labels.to(device)
            defect_labels = defect_labels.to(device)
            bbox_targets = bbox_targets.to(device)
            has_defect = has_defect.to(device)

            (
                class_output,
                defect_output,
                bbox_output
            ) = model(images)

            class_loss = classification_loss_fn(
                class_output,
                class_labels
            )

            if has_defect.any():

                defect_loss = defect_loss_fn(
                    defect_output[has_defect],
                    defect_labels[has_defect]
                )

                bbox_loss = bbox_loss_fn(
                    bbox_output[has_defect],
                    bbox_targets[has_defect]
                )

            else:

                defect_loss = torch.tensor(
                    0.0,
                    device=device
                )

                bbox_loss = torch.tensor(
                    0.0,
                    device=device
                )

            total_loss = (

                CLASSIFICATION_WEIGHT
                * class_loss

                +

                DEFECT_TYPE_WEIGHT
                * defect_loss

                +

                BBOX_WEIGHT
                * bbox_loss
            )

            val_running_loss += (
                total_loss.item()
                * images.size(0)
            )

            val_running_bbox_loss += (
                bbox_loss.item()
                * images.size(0)
            )

            # Classification
            class_predictions = torch.argmax(
                class_output,
                dim=1
            )

            val_correct_classes += (
                class_predictions == class_labels
            ).sum().item()

            val_total_classes += (
                class_labels.size(0)
            )

            # Defect type
            if has_defect.any():

                defect_predictions = torch.argmax(
                    defect_output[has_defect],
                    dim=1
                )

                val_correct_defects += (
                    defect_predictions
                    ==
                    defect_labels[has_defect]
                ).sum().item()

                val_total_defects += (
                    has_defect.sum().item()
                )

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    val_loss = (
        val_running_loss
        / len(val_dataset)
    )

    val_bbox_loss = (
        val_running_bbox_loss
        / len(val_dataset)
    )

    val_class_accuracy = (
        val_correct_classes
        / val_total_classes
    )

    val_defect_accuracy = (
        val_correct_defects
        / val_total_defects
    )


    # ========================================================
    # HISTORY
    # ========================================================

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_bbox_losses.append(
        train_bbox_loss
    )

    val_bbox_losses.append(
        val_bbox_loss
    )

    train_class_acc.append(
        train_class_accuracy
    )

    val_class_acc.append(
        val_class_accuracy
    )

    train_defect_acc.append(
        train_defect_accuracy
    )

    val_defect_acc.append(
        val_defect_accuracy
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n" + "-" * 70)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Val Loss        : {val_loss:.4f}"
    )

    print(
        f"Train Class Acc : "
        f"{train_class_accuracy * 100:.2f}%"
    )

    print(
        f"Val Class Acc   : "
        f"{val_class_accuracy * 100:.2f}%"
    )

    print(
        f"Train Defect Acc: "
        f"{train_defect_accuracy * 100:.2f}%"
    )

    print(
        f"Val Defect Acc  : "
        f"{val_defect_accuracy * 100:.2f}%"
    )

    print(
        f"Train BBox Loss : "
        f"{train_bbox_loss:.4f}"
    )

    print(
        f"Val BBox Loss   : "
        f"{val_bbox_loss:.4f}"
    )

    print(
        f"Learning Rate   : "
        f"{current_lr:.7f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "defect_names":
                    DEFECT_NAMES,

                "image_size":
                    IMAGE_SIZE
            },
            model_path
        )

        print(
            "✓ Best spatial model saved."
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    "\nBest validation loss:",
    f"{best_val_loss:.4f}"
)

print(
    "\nModel saved to:"
)

print(model_path)


# ============================================================
# GRAPH 1 — LOSS
# ============================================================

epochs = range(
    1,
    NUM_EPOCHS + 1
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    train_losses,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Improved ResNet18 Training Loss"
)

plt.legend()
plt.grid(True)

loss_path = os.path.join(
    RESULTS_DIR,
    "spatial_training_loss.png"
)

plt.savefig(
    loss_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 2 — CLASSIFICATION ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    np.array(train_class_acc) * 100,
    label="Training Classification Accuracy"
)

plt.plot(
    epochs,
    np.array(val_class_acc) * 100,
    label="Validation Classification Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "Improved ResNet18 Classification Accuracy"
)

plt.legend()
plt.grid(True)

class_acc_path = os.path.join(
    RESULTS_DIR,
    "spatial_classification_accuracy.png"
)

plt.savefig(
    class_acc_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 3 — DEFECT TYPE ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    np.array(train_defect_acc) * 100,
    label="Training Defect Type Accuracy"
)

plt.plot(
    epochs,
    np.array(val_defect_acc) * 100,
    label="Validation Defect Type Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "Improved ResNet18 Defect Type Accuracy"
)

plt.legend()
plt.grid(True)

defect_acc_path = os.path.join(
    RESULTS_DIR,
    "spatial_defect_accuracy.png"
)

plt.savefig(
    defect_acc_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 4 — BOUNDING BOX LOSS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    train_bbox_losses,
    label="Training Bounding Box Loss"
)

plt.plot(
    epochs,
    val_bbox_losses,
    label="Validation Bounding Box Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Smooth L1 Loss")

plt.title(
    "Improved ResNet18 Bounding Box Loss"
)

plt.legend()
plt.grid(True)

bbox_path = os.path.join(
    RESULTS_DIR,
    "spatial_bbox_loss.png"
)

plt.savefig(
    bbox_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nGraphs saved:")

print(loss_path)
print(class_acc_path)
print(defect_acc_path)
print(bbox_path)

print("\n" + "=" * 70)
print(
    "STATUS: IMPROVED RESNET18 "
    "SPATIAL TRAINING COMPLETED"
)
print("=" * 70)