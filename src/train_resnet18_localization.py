import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# CONFIGURATION
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
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_EPOCHS = 10

LEARNING_RATE = 0.0001

NUM_CLASSES = 2

NUM_DEFECT_TYPES = 5


# Loss weights
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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("RESNET18 LOCALIZATION TRAINING")
print("=" * 70)

print("\nDevice:", device)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)


# ============================================================
# DATASET
# ============================================================

class MetalDefectDetectionDataset(Dataset):

    def __init__(
        self,
        split,
        transform=None
    ):

        self.split = split
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

        image_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        )

        image_files = sorted([
            file
            for file in os.listdir(self.image_dir)
            if file.lower().endswith(image_extensions)
        ])

        # ----------------------------------------------------
        # Read every image and annotation
        # ----------------------------------------------------

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

            # ------------------------------------------------
            # Determine whether image is defective
            # ------------------------------------------------

            has_defect = False
            defect_type = -1
            bbox = [
                0.0,
                0.0,
                0.0,
                0.0
            ]

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

                    # Our current dataset contains
                    # one defect per image.

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

                        has_defect = True

            # ------------------------------------------------
            # Binary class
            # ------------------------------------------------

            if has_defect:

                class_id = 0       # defective

            else:

                class_id = 1       # good

            self.samples.append(
                {
                    "image": image_path,
                    "class_id": class_id,
                    "defect_type": defect_type,
                    "bbox": bbox,
                    "has_defect": has_defect
                }
            )

        print(
            f"{split.upper()} dataset loaded: "
            f"{len(self.samples)} images"
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        sample = self.samples[index]

        image_path = sample["image"]

        image = cv2.imread(image_path)

        if image is None:

            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Convert to PIL Image for torchvision transforms
        # ----------------------------------------------------

        from PIL import Image

        image = Image.fromarray(image)

        if self.transform is not None:

            image = self.transform(image)

        # ----------------------------------------------------
        # Create tensors
        # ----------------------------------------------------

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
        degrees=5
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
# CREATE DATASETS
# ============================================================

train_dataset = MetalDefectDetectionDataset(
    "train",
    transform=train_transform
)

val_dataset = MetalDefectDetectionDataset(
    "val",
    transform=val_transform
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
# DATASET INFORMATION
# ============================================================

train_class_counts = Counter(
    sample["class_id"]
    for sample in train_dataset.samples
)

print("\nTraining class distribution:")

print(
    "Defective:",
    train_class_counts[0]
)

print(
    "Good:",
    train_class_counts[1]
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_train = (
    train_class_counts[0]
    + train_class_counts[1]
)

defective_weight = (
    total_train
    / (2 * train_class_counts[0])
)

good_weight = (
    total_train
    / (2 * train_class_counts[1])
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
    "Defective:",
    f"{defective_weight:.4f}"
)

print(
    "Good:",
    f"{good_weight:.4f}"
)


# ============================================================
# MODEL
# ============================================================

class ResNet18Localization(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Pretrained ResNet18
        # ----------------------------------------------------

        weights = ResNet18_Weights.DEFAULT

        resnet = resnet18(
            weights=weights
        )

        # ----------------------------------------------------
        # Remove original classifier
        # ----------------------------------------------------

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.feature_size = 512

        # ----------------------------------------------------
        # Binary classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                self.feature_size,
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
        # Defect type head
        # ----------------------------------------------------

        self.defect_classifier = nn.Sequential(

            nn.Linear(
                self.feature_size,
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
        # Bounding box head
        # ----------------------------------------------------

        self.bbox_regressor = nn.Sequential(

            nn.Linear(
                self.feature_size,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                4
            ),

            nn.Sigmoid()
        )

    # ========================================================
    # FORWARD
    # ========================================================

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
# CREATE MODEL
# ============================================================

print("\nLoading pretrained ResNet18...")

model = ResNet18Localization()

model = model.to(device)

print("Model loaded successfully.")


# ============================================================
# LOSS FUNCTIONS
# ============================================================

classification_loss_fn = nn.CrossEntropyLoss(
    weight=class_weights
)

defect_loss_fn = nn.CrossEntropyLoss()

bbox_loss_fn = nn.SmoothL1Loss(
    reduction="mean"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []
val_losses = []

train_class_losses = []
val_class_losses = []

train_bbox_losses = []
val_bbox_losses = []

train_class_accuracies = []
val_class_accuracies = []

train_defect_accuracies = []
val_defect_accuracies = []


# ============================================================
# BEST MODEL
# ============================================================

best_val_loss = float("inf")

best_model_path = os.path.join(
    MODEL_DIR,
    "resnet18_localization.pth"
)


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(NUM_EPOCHS):

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    running_loss = 0.0
    running_class_loss = 0.0
    running_bbox_loss = 0.0

    correct_classes = 0
    total_classes = 0

    correct_defects = 0
    total_defects = 0

    # --------------------------------------------------------
    # Batches
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

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
        # Defect type + bounding box loss
        # Only defective images
        # ----------------------------------------------------

        if has_defect.any():

            defect_mask = has_defect

            defect_loss = defect_loss_fn(
                defect_output[defect_mask],
                defect_labels[defect_mask]
            )

            bbox_loss = bbox_loss_fn(
                bbox_output[defect_mask],
                bbox_targets[defect_mask]
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
        # Total loss
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

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        total_loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
            total_loss.item()
            * images.size(0)
        )

        running_class_loss += (
            class_loss.item()
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
        # Defect type accuracy
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
    # TRAINING RESULTS
    # ========================================================

    train_loss = (
        running_loss
        / len(train_dataset)
    )

    train_class_loss = (
        running_class_loss
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

    if total_defects > 0:

        train_defect_accuracy = (
            correct_defects
            / total_defects
        )

    else:

        train_defect_accuracy = 0.0


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0
    val_running_class_loss = 0.0
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

            # ------------------------------------------------
            # Classification loss
            # ------------------------------------------------

            class_loss = classification_loss_fn(
                class_output,
                class_labels
            )

            # ------------------------------------------------
            # Defect + bbox loss
            # ------------------------------------------------

            if has_defect.any():

                defect_mask = has_defect

                defect_loss = defect_loss_fn(
                    defect_output[defect_mask],
                    defect_labels[defect_mask]
                )

                bbox_loss = bbox_loss_fn(
                    bbox_output[defect_mask],
                    bbox_targets[defect_mask]
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

            # ------------------------------------------------
            # Total loss
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            val_running_loss += (
                total_loss.item()
                * images.size(0)
            )

            val_running_class_loss += (
                class_loss.item()
                * images.size(0)
            )

            val_running_bbox_loss += (
                bbox_loss.item()
                * images.size(0)
            )

            # ------------------------------------------------
            # Classification accuracy
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Defect accuracy
            # ------------------------------------------------

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
    # VALIDATION RESULTS
    # ========================================================

    val_loss = (
        val_running_loss
        / len(val_dataset)
    )

    val_class_loss = (
        val_running_class_loss
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

    if val_total_defects > 0:

        val_defect_accuracy = (
            val_correct_defects
            / val_total_defects
        )

    else:

        val_defect_accuracy = 0.0


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_class_losses.append(
        train_class_loss
    )

    val_class_losses.append(
        val_class_loss
    )

    train_bbox_losses.append(
        train_bbox_loss
    )

    val_bbox_losses.append(
        val_bbox_loss
    )

    train_class_accuracies.append(
        train_class_accuracy
    )

    val_class_accuracies.append(
        val_class_accuracy
    )

    train_defect_accuracies.append(
        train_defect_accuracy
    )

    val_defect_accuracies.append(
        val_defect_accuracy
    )


    # ========================================================
    # LEARNING RATE
    # ========================================================

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # PRINT EPOCH
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
            best_model_path
        )

        print(
            "✓ Best model saved."
        )


# ============================================================
# TRAINING COMPLETED
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

print(
    best_model_path
)


# ============================================================
# SAVE LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "ResNet18 Localization Training Loss"
)

plt.legend()

plt.grid(True)

loss_graph = os.path.join(
    RESULTS_DIR,
    "localization_training_loss.png"
)

plt.savefig(
    loss_graph,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE CLASSIFICATION ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    np.array(train_class_accuracies) * 100,
    label="Training Classification Accuracy"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    np.array(val_class_accuracies) * 100,
    label="Validation Classification Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title(
    "ResNet18 Classification Accuracy"
)

plt.legend()

plt.grid(True)

accuracy_graph = os.path.join(
    RESULTS_DIR,
    "localization_classification_accuracy.png"
)

plt.savefig(
    accuracy_graph,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE DEFECT TYPE ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    np.array(train_defect_accuracies) * 100,
    label="Training Defect Type Accuracy"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    np.array(val_defect_accuracies) * 100,
    label="Validation Defect Type Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title(
    "ResNet18 Defect Type Accuracy"
)

plt.legend()

plt.grid(True)

defect_accuracy_graph = os.path.join(
    RESULTS_DIR,
    "localization_defect_accuracy.png"
)

plt.savefig(
    defect_accuracy_graph,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE BOUNDING BOX LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    train_bbox_losses,
    label="Training Bounding Box Loss"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    val_bbox_losses,
    label="Validation Bounding Box Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Smooth L1 Loss")

plt.title(
    "ResNet18 Bounding Box Loss"
)

plt.legend()

plt.grid(True)

bbox_graph = os.path.join(
    RESULTS_DIR,
    "localization_bbox_loss.png"
)

plt.savefig(
    bbox_graph,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL FILES
# ============================================================

print("\nGraphs saved:")

print(
    loss_graph
)

print(
    accuracy_graph
)

print(
    defect_accuracy_graph
)

print(
    bbox_graph
)

print("\n" + "=" * 70)
print("STATUS: RESNET18 LOCALIZATION TRAINING COMPLETED")
print("=" * 70)